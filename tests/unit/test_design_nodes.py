"""Node-level tests for the Requirement Analyzer, Test Case Generator and Test Planner."""

import pytest

from qa_agent.config import get_settings
from qa_agent.domain import Requirement, RequirementAnalysis, TestCase
from qa_agent.graph import build_qa_graph, initial_state, run_config
from qa_agent.graph.nodes.requirement_analyzer import make_requirement_analyzer
from qa_agent.graph.nodes.test_case_generator import make_test_case_generator
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from qa_agent.graph.nodes.test_planner import make_test_planner
from tests.conftest import PASSWORD_RESET_REQUIREMENT, load_fixture, repo_knowledge_base


def _requirement() -> Requirement:
    return Requirement(text=PASSWORD_RESET_REQUIREMENT)


def _analysis() -> RequirementAnalysis:
    return RequirementAnalysis.model_validate(
        load_fixture("password_reset", "requirement_analysis.json")
    )


def _test_cases() -> list[TestCase]:
    raw = load_fixture("password_reset", "test_suite.json")["test_cases"]
    return [
        TestCase.model_validate({**tc, "test_case_id": f"TC-{i:03d}"})
        for i, tc in enumerate(raw, start=1)
    ]


def test_requirement_analyzer_writes_analysis(fake_llm):
    update = make_requirement_analyzer(fake_llm)({"requirement": _requirement()})

    assert update["requirement_analysis"] == _analysis()


def test_test_case_generator_uses_analysis_from_state(fake_llm):
    analysis = _analysis().model_copy(update={"summary": "SUMMARY-FROM-STATE"})

    update = make_test_case_generator(fake_llm, repo_knowledge_base)(
        {"requirement": _requirement(), "requirement_analysis": analysis}
    )

    assert len(update["test_cases"]) == 8
    [messages] = fake_llm.calls_for("TestSuite")
    assert "SUMMARY-FROM-STATE" in messages[-1].content


def test_test_case_generator_requires_analysis(fake_llm):
    with pytest.raises(ValueError, match="requirement_analysis"):
        make_test_case_generator(fake_llm, repo_knowledge_base)({"requirement": _requirement()})
    assert fake_llm.calls == []


def test_test_planner_writes_plan_and_syncs_test_cases(fake_llm):
    update = make_test_planner(fake_llm)({"requirement": _requirement(), "test_cases": _test_cases()})

    plan = update["test_plan"]
    assert len(plan.entries) == 8
    planned = {e.test_case_id: e.automation_type for e in plan.entries}
    assert {tc.test_case_id: tc.automation_type for tc in update["test_cases"]} == planned


def test_test_planner_skips_llm_when_there_are_no_test_cases(fake_llm):
    update = make_test_planner(fake_llm)({"requirement": _requirement(), "test_cases": []})

    assert update["test_plan"].entries == []
    assert "test_cases" not in update
    assert fake_llm.calls == []


def test_graph_runs_design_nodes_in_order_and_calls_llm_once_each(fake_llm, knowledge_base):
    requirement = _requirement()

    graph = build_qa_graph(
        {"test_executor": make_mock_test_executor()}, llm=fake_llm, knowledge_base=knowledge_base
    )
    state = graph.invoke(initial_state(requirement), run_config(requirement.id))

    assert [name for name, _ in fake_llm.calls] == [
        "RequirementAnalysis",
        "TestSuite",
        "TestPlanDraft",
    ]
    assert state["errors"] == []
    assert state["test_plan"].ids_for("manual") == ["TC-003"]


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("no_api_key")
def test_missing_llm_configuration_is_recorded_not_raised():
    requirement = _requirement()

    state = build_qa_graph().invoke(initial_state(requirement), run_config(requirement.id))

    assert state["errors"][0].node == "requirement_analyzer"
    assert state["errors"][0].error_type == "LLMConfigurationError"
    assert state["final_report"].status == "error"


def test_configured_automation_types_follows_test_targets():
    from qa_agent.config import Settings
    from qa_agent.graph.nodes.test_planner import configured_automation_types

    def types(**kw):
        return configured_automation_types(Settings(_env_file=None, **kw))

    assert types() is None
    assert types(ui_test_base_url="http://127.0.0.1:8765") == {"ui", "manual"}
    assert types(api_test_base_url="http://127.0.0.1:9000") == {"api", "manual"}
