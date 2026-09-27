import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from qa_agent.chains import (
    LLMConfigurationError,
    analyze_requirement,
    apply_plan,
    generate_test_suite,
    get_chat_model,
    plan_tests,
)
from qa_agent.chains.test_case_chain import normalize_test_ids
from qa_agent.chains.test_plan_chain import AutomationDecision, TestPlanDraft, finalize_plan
from qa_agent.config import Settings
from qa_agent.domain import Requirement, RequirementAnalysis, TestSuite
from tests.conftest import PASSWORD_RESET_REQUIREMENT, load_fixture
from tests.fakes import FakeStructuredChatModel


def _analysis() -> RequirementAnalysis:
    return RequirementAnalysis.model_validate(
        load_fixture("password_reset", "requirement_analysis.json")
    )


def _suite(fake_llm) -> TestSuite:
    return generate_test_suite(fake_llm, PASSWORD_RESET_REQUIREMENT, _analysis())


# --- llm provider ------------------------------------------------------------


def test_get_chat_model_requires_api_key():
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        get_chat_model(Settings(_env_file=None, openai_api_key=None))


def test_get_chat_model_uses_configured_model():
    settings = Settings(_env_file=None, openai_api_key="sk-test", llm_model="gpt-4o-mini")

    llm = get_chat_model(settings)

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"


def test_get_chat_model_adds_configured_fallbacks():
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test",
        llm_model="primary",
        llm_max_retries=4,
        llm_fallback_models="backup-a, backup-b",
    )

    llm = get_chat_model(settings)

    assert isinstance(llm, RunnableWithFallbacks)
    models = [llm.runnable, *llm.fallbacks]
    assert [m.model_name for m in models] == ["primary", "backup-a", "backup-b"]
    assert all(m.max_retries == 4 for m in models)
    # Chains call with_structured_output(); it must reach every model.
    structured = llm.with_structured_output(TestSuite)
    assert isinstance(structured, RunnableWithFallbacks)
    assert len(structured.fallbacks) == 2


# --- requirement analysis ----------------------------------------------------


def test_analyze_requirement_returns_structured_analysis(fake_llm):
    analysis = analyze_requirement(fake_llm, Requirement(text=PASSWORD_RESET_REQUIREMENT))

    assert isinstance(analysis, RequirementAnalysis)
    assert analysis.feature == "Password reset"
    assert len(analysis.acceptance_criteria) == 4
    assert "Reset link expiry duration is not specified" in analysis.ambiguities


def test_analyze_requirement_sends_requirement_text(fake_llm):
    analyze_requirement(fake_llm, Requirement(text=PASSWORD_RESET_REQUIREMENT))

    [messages] = fake_llm.calls_for("RequirementAnalysis")
    assert messages[0].type == "system"
    assert PASSWORD_RESET_REQUIREMENT in messages[-1].content


def test_invalid_llm_output_is_retried_once():
    valid = load_fixture("password_reset", "requirement_analysis.json")
    invalid = {k: v for k, v in valid.items() if k != "acceptance_criteria"}
    llm = FakeStructuredChatModel(responses={"RequirementAnalysis": [invalid, valid]})

    analysis = analyze_requirement(llm, Requirement(text=PASSWORD_RESET_REQUIREMENT))

    assert len(analysis.acceptance_criteria) == 4
    assert len(llm.calls) == 2


def test_persistently_invalid_llm_output_raises():
    llm = FakeStructuredChatModel(responses={"RequirementAnalysis": [{"summary": "only"}]})

    with pytest.raises((ValidationError, OutputParserException)):
        analyze_requirement(llm, Requirement(text=PASSWORD_RESET_REQUIREMENT))
    assert len(llm.calls) == 2


# --- test case generation ----------------------------------------------------


def test_generated_suite_covers_positive_negative_and_edge_cases(fake_llm):
    suite = _suite(fake_llm)

    by_type = {t: [tc for tc in suite.test_cases if tc.test_type == t] for t in ("positive", "negative", "edge_case")}
    assert all(by_type.values())
    assert len(suite.test_cases) == 8


def test_generated_test_cases_are_fully_populated(fake_llm):
    for tc in _suite(fake_llm).test_cases:
        assert tc.title and tc.description and tc.expected_result
        assert tc.steps
        assert tc.test_data
        assert tc.automation_type in {"api", "ui", "manual"}


def test_generation_prompt_uses_requirement_analysis(fake_llm):
    _suite(fake_llm)

    [messages] = fake_llm.calls_for("TestSuite")
    prompt = messages[-1].content
    assert PASSWORD_RESET_REQUIREMENT in prompt
    assert "Account enumeration" in prompt  # a risk area from the analysis
    assert "Reset link expiry duration is not specified" in prompt


def test_generated_ids_are_renumbered_sequentially(fake_llm):
    suite = _suite(fake_llm)

    assert [tc.test_case_id for tc in suite.test_cases] == [f"TC-{i:03d}" for i in range(1, 9)]
    assert suite.test_cases[0].title == "Reset request with registered email is accepted"


def test_normalize_test_ids_makes_duplicates_unique():
    payload = load_fixture("password_reset", "test_suite.json")
    for tc in payload["test_cases"]:
        tc["test_case_id"] = "DUP"

    suite = normalize_test_ids(TestSuite.model_validate(payload))

    assert len({tc.test_case_id for tc in suite.test_cases}) == len(suite.test_cases)


# --- test planning -----------------------------------------------------------


def test_plan_classifies_each_test_as_api_ui_or_manual(fake_llm):
    plan = plan_tests(fake_llm, _suite(fake_llm).test_cases)

    assert plan.ids_for("api") == ["TC-001", "TC-004", "TC-006", "TC-007", "TC-008"]
    assert plan.ids_for("ui") == ["TC-002", "TC-005"]
    assert plan.ids_for("manual") == ["TC-003"]
    assert plan.warnings == []
    assert all(e.rationale for e in plan.entries)


def test_plan_orders_by_priority_then_automation_type(fake_llm):
    plan = plan_tests(fake_llm, _suite(fake_llm).test_cases)

    ordered = sorted(plan.entries, key=lambda e: e.execution_order)
    assert [e.test_case_id for e in ordered] == [
        "TC-001", "TC-004", "TC-002",  # high: api, api, ui
        "TC-006", "TC-007", "TC-005", "TC-003",  # medium: api, api, ui, manual
        "TC-008",  # low
    ]
    assert [e.execution_order for e in ordered] == list(range(1, 9))


def test_planning_prompt_contains_every_test_case(fake_llm):
    test_cases = _suite(fake_llm).test_cases
    plan_tests(fake_llm, test_cases)

    [messages] = fake_llm.calls_for("TestPlanDraft")
    for tc in test_cases:
        assert tc.test_case_id in messages[-1].content


def test_apply_plan_overrides_generator_automation_suggestion(fake_llm):
    test_cases = _suite(fake_llm).test_cases
    assert test_cases[4].automation_type == "api"  # generator suggested api for TC-005

    updated = apply_plan(test_cases, plan_tests(fake_llm, test_cases))

    assert updated[4].automation_type == "ui"


def test_finalize_plan_reconciles_bad_decisions(fake_llm):
    test_cases = _suite(fake_llm).test_cases[:3]
    draft = TestPlanDraft(
        strategy="s",
        decisions=[
            AutomationDecision(test_case_id="TC-001", automation_type="ui", rationale="first"),
            AutomationDecision(test_case_id="TC-001", automation_type="manual", rationale="dup"),
            AutomationDecision(test_case_id="TC-999", automation_type="api", rationale="unknown"),
            AutomationDecision(test_case_id="TC-003", automation_type="manual", rationale="inbox"),
        ],
    )

    plan = finalize_plan(draft, test_cases)

    entries = {e.test_case_id: e for e in plan.entries}
    assert set(entries) == {"TC-001", "TC-002", "TC-003"}
    assert entries["TC-001"].automation_type == "ui"
    assert entries["TC-002"].automation_type == "ui"  # generator suggestion kept
    assert plan.warnings == [
        "Ignored duplicate decision for TC-001",
        "Ignored decision for unknown test case TC-999",
        "No planner decision for TC-002; kept suggested automation type 'ui'",
    ]


def test_finalize_plan_moves_tests_without_a_configured_target(fake_llm):
    test_cases = _suite(fake_llm).test_cases[:3]
    draft = TestPlanDraft(
        strategy="s",
        decisions=[
            AutomationDecision(test_case_id=test_cases[0].test_case_id, automation_type="api", rationale="r"),
            AutomationDecision(test_case_id=test_cases[1].test_case_id, automation_type="ui", rationale="r"),
            AutomationDecision(test_case_id=test_cases[2].test_case_id, automation_type="manual", rationale="r"),
        ],
    )

    plan = finalize_plan(draft, test_cases, available={"ui", "manual"})

    types = {e.test_case_id: e.automation_type for e in plan.entries}
    assert [types[tc.test_case_id] for tc in test_cases] == ["ui", "ui", "manual"]
    assert any("running it as 'ui'" in w for w in plan.warnings)


def test_finalize_plan_keeps_type_when_no_alternative_is_configured(fake_llm):
    test_cases = _suite(fake_llm).test_cases[:1]
    draft = TestPlanDraft(
        strategy="s",
        decisions=[AutomationDecision(test_case_id=test_cases[0].test_case_id, automation_type="api", rationale="r")],
    )

    plan = finalize_plan(draft, test_cases, available={"manual"})

    assert plan.entries[0].automation_type == "api"
    assert plan.warnings == []
