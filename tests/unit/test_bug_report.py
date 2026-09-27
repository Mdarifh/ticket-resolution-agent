"""Bug Report Generator: facts from recorded data, narrative from the LLM, uncertainty marked."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from qa_agent.analysis.failure_facts import analyze_failure
from qa_agent.chains.bug_report_chain import (
    BugReportContext,
    generate_bug_reports,
    render_bug_report_markdown,
)
from qa_agent.chains.root_cause_chain import RootCauseDraft, finalize_root_cause
from qa_agent.domain import BugEnvironment, BugReport, HumanReview, ProbableCause, Requirement, RequirementAnalysis
from qa_agent.graph import build_qa_graph, initial_state, run_config
from qa_agent.graph.nodes.bug_report_generator import BUG_REPORT_DOC_TYPES, make_bug_report_generator
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from tests.conftest import PASSWORD_RESET_REQUIREMENT, load_fixture, password_reset_responses, repo_knowledge_base
from tests.fakes import FakeStructuredChatModel
from tests.scenarios import api_500, make_draft, make_finding, requirement_mismatch, ui_element_missing

TOKEN_REUSE_BUG = "previous_bugs/BUG-1042-reset-token-reusable.md"
REQUIRED_FIELDS = {
    "title",
    "summary",
    "environment",
    "preconditions",
    "reproduction_steps",
    "expected_result",
    "actual_result",
    "evidence",
    "probable_root_cause",
    "severity",
    "priority",
    "affected_component",
    "confidence",
}


def _narrative(finding_id="RC-1", **overrides) -> dict:
    return {
        "finding_id": finding_id,
        "title": "Password reset request returns HTTP 500",
        "summary": "Reset requests fail with HTTP 500; the cause is possibly an unhandled exception.",
        "severity": "critical",
        "severity_rationale": "Core account recovery flow is blocked",
        "priority": "p0",
        "priority_rationale": "Blocks all users",
        "related_bugs": [],
        **overrides,
    }


def _context(scenario=api_500, finding_overrides=None, **context_overrides):
    """Run a scenario through the Failure Analyzer and a guarded RCA; return a report context."""
    result, test_case = scenario()
    failure = analyze_failure(result, test_case)
    finding = make_finding([test_case.test_case_id], **(finding_overrides or {}))
    knowledge = repo_knowledge_base().search("password reset", k=6)
    rca, _ = finalize_root_cause(RootCauseDraft.model_validate(make_draft(finding)), [failure], knowledge)
    context = BugReportContext(
        requirement_text=PASSWORD_RESET_REQUIREMENT,
        requirement_analysis=RequirementAnalysis.model_validate(load_fixture("password_reset", "requirement_analysis.json")),
        root_cause=rca,
        failures=[failure],
        test_cases=[test_case],
        execution_results=[result],
        app_env="staging",
        run_id="run-123",
        **context_overrides,
    )
    return context, failure, test_case


def _generate(context, *narratives, knowledge=None):
    llm = FakeStructuredChatModel(responses={"BugNarrativeDraft": [{"reports": list(narratives) or [_narrative()]}]})
    knowledge = knowledge or repo_knowledge_base().search("password reset token used twice", doc_types=BUG_REPORT_DOC_TYPES)
    reports, discarded, fallback = generate_bug_reports(llm, context, knowledge)
    return reports, discarded, fallback, llm


# --- model ---------------------------------------------------------------------


def test_bug_report_has_every_required_field():
    assert REQUIRED_FIELDS <= set(BugReport.model_fields)


def test_bug_report_rejects_invalid_values():
    context, _, _ = _context()
    [report] = _generate(context)[0]
    payload = report.model_dump()

    for field, value in [("severity", "urgent"), ("priority", "p9"), ("confidence", 1.5), ("report_type", "feature")]:
        with pytest.raises(ValidationError):
            BugReport.model_validate({**payload, field: value})


# --- facts come only from recorded data ---------------------------------------


def test_report_from_api_500_is_complete():
    context, failure, test_case = _context()

    [report], _, fallback, _ = _generate(context)

    assert fallback is None
    assert all(getattr(report, field) not in (None, "", []) for field in REQUIRED_FIELDS - {"preconditions"})
    assert report.id == "BR-1"
    assert report.generated_by == "llm"
    assert report.related_test_case_ids == ["TC-001"]
    assert report.preconditions == test_case.preconditions
    assert report.expected_result == test_case.expected_result
    assert report.actual_result == failure.actual_result == "HTTP 500; Expected status 202, got 500"


def test_evidence_is_exactly_the_recorded_facts_plus_validated_knowledge():
    context, failure, _ = _context(
        finding_overrides={"evidence": [{"reference": "TC-001:F1", "interpretation": "server error"}, {"reference": "api_docs/auth-password-reset-api.md", "interpretation": "contract"}]}
    )

    [report] = _generate(context)[0]

    observed = [e for e in report.evidence if e.kind == "observed"]
    assert [(e.reference, e.detail) for e in observed] == [(f.id, f.detail) for f in failure.observed_facts]
    assert observed[0].interpretation == "server error"
    assert [e.reference for e in report.evidence if e.kind == "knowledge_base"] == ["api_docs/auth-password-reset-api.md"]


def test_llm_narrative_cannot_add_evidence_or_change_facts():
    context, failure, test_case = _context()
    narrative = _narrative(summary="Server logs show a NullPointerException in ResetService.")

    [report] = _generate(context, narrative)[0]

    # The narrative is only title/summary/severity/priority; facts stay recorded values.
    assert all(e.detail in {f.detail for f in failure.observed_facts} for e in report.evidence if e.kind == "observed")
    assert report.expected_result == test_case.expected_result
    assert "NullPointerException" not in " ".join(e.detail for e in report.evidence)


def test_environment_is_derived_from_evidence_and_settings():
    context, _, _ = _context()

    [report] = _generate(context)[0]

    env = report.environment
    assert env.app_env == "staging"
    assert env.run_id == "run-123"
    assert env.targets == ["https://sut.test.local"]
    assert env.automation_types == ["api"]
    assert "Build or version of the system under test was not recorded" in env.unknowns


def test_ui_reproduction_steps_are_the_executed_browser_steps():
    context, _, _ = _context(ui_element_missing)

    [report] = _generate(context)[0]

    assert report.reproduction_source == "executed_steps"
    assert report.reproduction_steps == [
        "navigate /forgot-password.html",
        "fill data-testid=email-input with value 'not-an-email'",
        "click data-testid=send-reset-link  <- fails here",
        "Test data: email=not-an-email",
    ]
    assert report.environment.targets == ["http://127.0.0.1:8765"]


def test_api_reproduction_uses_rca_steps_plus_test_data():
    context, _, _ = _context(finding_overrides={"reproduction_steps": ["POST /api/v2/password-reset", "Observe HTTP 500"]})

    [report] = _generate(context)[0]

    assert report.reproduction_source == "root_cause_analysis"
    assert report.reproduction_steps == [
        "POST /api/v2/password-reset",
        "Observe HTTP 500",
        "Test data: email=registered.user@example.com",
    ]


def test_reproduction_falls_back_to_test_case_steps():
    context, _, test_case = _context(finding_overrides={"reproduction_steps": []})

    [report] = _generate(context)[0]

    assert report.reproduction_source == "test_case"
    assert report.reproduction_steps[: len(test_case.steps)] == test_case.steps


# --- uncertainty ---------------------------------------------------------------


def test_probable_root_cause_keeps_likelihood_and_is_marked_unconfirmed():
    context, _, _ = _context()

    [report] = _generate(context)[0]

    assert report.probable_root_cause.likelihood == "likely"
    assert report.uncertainties[0] == "Probable root cause is not confirmed (assessed as likely)"
    assert "Server logs are not available" in report.uncertainties  # RCA unknowns carried over
    assert report.confidence == 0.8


def test_low_confidence_and_unknown_component_are_flagged():
    context, _, _ = _context(finding_overrides={"confidence": 0.35, "affected_component": ""})

    [report] = _generate(context)[0]

    assert report.affected_component == "unknown"
    assert "Root cause analysis confidence is low (0.35); verify before acting" in report.uncertainties
    assert "Affected component could not be determined" in report.uncertainties


def test_speculative_cause_stays_speculative_in_the_report():
    finding = {"probable_root_cause": {"description": "Maybe DNS", "reasoning": "guess", "likelihood": "likely", "supporting_evidence": []}}
    context, _, _ = _context(finding_overrides=finding)

    [report] = _generate(context)[0]

    assert report.probable_root_cause.likelihood == "speculative"
    assert report.confidence == 0.3
    assert "No observed evidence directly supports the probable cause" in report.uncertainties


@pytest.mark.parametrize(
    ("origin", "report_type"),
    [
        ("product_defect", "product_bug"),
        ("test_defect", "test_issue"),
        ("requirement_mismatch", "requirement_issue"),
        ("environment", "environment_issue"),
        ("unknown", "needs_investigation"),
    ],
)
def test_report_type_follows_suspected_origin(origin, report_type):
    context, _, _ = _context(finding_overrides={"suspected_origin": origin})

    [report] = _generate(context)[0]

    assert report.report_type == report_type


def test_requirement_mismatch_is_reported_as_a_requirement_issue():
    context, _, test_case = _context(requirement_mismatch, finding_overrides={"suspected_origin": "requirement_mismatch"})

    [report] = _generate(context, _narrative(title="TC-004 expects 404 but the requirement demands a generic 202", severity="low", priority="p3"))[0]

    assert report.report_type == "requirement_issue"
    assert report.expected_result == "Response is 404 Not Found for an unregistered email"
    assert report.actual_result == "HTTP 202; Expected status 404, got 202"


# --- LLM narrative and RAG -----------------------------------------------------


def test_narrative_fields_come_from_llm():
    context, _, _ = _context()

    [report] = _generate(context, _narrative(severity="high", priority="p1", severity_rationale="because"))[0]

    assert (report.title, report.severity, report.priority) == ("Password reset request returns HTTP 500", "high", "p1")
    assert report.severity_rationale == "because"


def test_related_bugs_must_be_retrieved_previous_bugs():
    context, _, _ = _context()
    narrative = _narrative(related_bugs=[TOKEN_REUSE_BUG, "previous_bugs/BUG-0000-invented.md", "api_docs/auth-password-reset-api.md"])

    [report], discarded, _, _ = _generate(context, narrative)

    assert report.related_bugs == [TOKEN_REUSE_BUG]
    assert discarded == ["previous_bugs/BUG-0000-invented.md", "api_docs/auth-password-reset-api.md"]


def test_prompt_contains_requirement_finding_facts_and_previous_bugs():
    context, failure, _ = _context()

    _, _, _, llm = _generate(context)

    [messages] = llm.calls_for("BugNarrativeDraft")
    system, human = messages[0].content, messages[-1].content
    assert PASSWORD_RESET_REQUIREMENT in human
    assert "- User can request a password reset by submitting their registered email" in human
    assert '"finding_id": "RC-1"' in human
    assert failure.observed_facts[0].detail in human
    assert f"[source: {TOKEN_REUSE_BUG}" in human
    assert "Never introduce facts" in system
    assert "hedge it" in system


# --- fallback --------------------------------------------------------------------


def test_missing_narrative_uses_deterministic_fallback():
    context, _, _ = _context()

    [report], _, fallback, _ = _generate(context, _narrative(finding_id="RC-9"))

    assert report.generated_by == "fallback"
    assert report.severity == "high"  # RCA severity suggestion
    assert report.priority == "p1"
    assert report.title == "TC-001: summary"
    assert "Probable cause (likely, unconfirmed)" in report.summary
    assert "Title, summary, severity and priority were generated without the LLM" in report.uncertainties


def test_llm_failure_still_produces_reports():
    context, _, _ = _context()
    broken = FakeStructuredChatModel(responses={})

    reports, _, fallback = generate_bug_reports(broken, context, repo_knowledge_base().search("reset"))

    assert [r.generated_by for r in reports] == ["fallback"]
    assert fallback.startswith("Narrative generation failed")


def test_no_findings_means_no_reports_and_no_llm_call():
    context, _, _ = _context()
    empty = context.model_copy(update={"root_cause": context.root_cause.model_copy(update={"findings": []})})
    llm = FakeStructuredChatModel(responses={})

    assert generate_bug_reports(llm, empty, repo_knowledge_base().search("reset")) == ([], [], None)
    assert llm.calls == []


# --- node ------------------------------------------------------------------------


def _node_state(context) -> dict:
    return {
        "requirement": Requirement(id="run-123", text=PASSWORD_RESET_REQUIREMENT),
        "requirement_analysis": context.requirement_analysis,
        "root_cause_analysis": context.root_cause,
        "failures": context.failures,
        "test_cases": context.test_cases,
        "execution_results": context.execution_results,
    }


def test_node_retrieves_previous_bugs_and_records_usage():
    context, _, _ = _context()
    llm = FakeStructuredChatModel(responses={"BugNarrativeDraft": [{"reports": [_narrative(related_bugs=[TOKEN_REUSE_BUG])]}]})

    update = make_bug_report_generator(llm, repo_knowledge_base)(_node_state(context))

    [usage] = update["retrieved_knowledge"]
    assert usage.node == "bug_report_generator"
    assert all(s.startswith("previous_bugs/") for s in usage.retrieved_sources)
    assert len(update["bug_reports"]) == 1


def test_node_without_rca_or_failures_produces_nothing():
    llm = FakeStructuredChatModel(responses={})

    assert make_bug_report_generator(llm, repo_knowledge_base)({"requirement": Requirement(text="x")}) == {"bug_reports": []}
    assert llm.calls == []


def test_node_regenerates_with_feedback_on_reanalysis():
    context, _, _ = _context()
    llm = FakeStructuredChatModel(responses={"BugNarrativeDraft": [{"reports": [_narrative()]}]})
    node = make_bug_report_generator(llm, repo_knowledge_base)
    state = _node_state(context)
    first = node(state)["bug_reports"]

    decided = HumanReview(
        review_id="run-123-review-1", reason="r", ai_recommendation="a", confidence=0.5, proposed_action="p",
        requested_at=datetime.now(UTC), human_decision="REQUEST_REANALYSIS", reviewer_comment="Add the trace id",
    )
    state.update(bug_reports=first, human_review=decided, review_history=[decided])
    second = node(state)["bug_reports"]

    assert [r.revision for r in first] == [0] and [r.revision for r in second] == [1]
    assert second[0].reviewer_feedback == ["Add the trace id"]
    assert "- Add the trace id" in llm.calls_for("BugNarrativeDraft")[-1][-1].content


def test_node_without_llm_configuration_falls_back(monkeypatch):
    context, _, _ = _context()

    def no_llm():
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr("qa_agent.graph.nodes.bug_report_generator.get_chat_model", no_llm)

    update = make_bug_report_generator(None, repo_knowledge_base)(_node_state(context))

    assert [r.generated_by for r in update["bug_reports"]] == ["fallback"]


def test_deterministic_mode_uses_no_llm_or_retrieval():
    context, _, _ = _context()

    def must_not_retrieve():
        raise AssertionError("no retrieval expected")

    update = make_bug_report_generator(None, must_not_retrieve, use_llm=False)(_node_state(context))

    assert [r.generated_by for r in update["bug_reports"]] == ["fallback"]
    assert "retrieved_knowledge" not in update


# --- rendering ---------------------------------------------------------------------


def test_markdown_marks_uncertain_information():
    context, _, _ = _context()
    [report] = _generate(context)[0]

    text = render_bug_report_markdown(report)

    assert text.startswith("### BR-1: Password reset request returns HTTP 500")
    assert "- **Actual (observed):** HTTP 500; Expected status 202, got 500" in text
    assert "- **Probable root cause (UNCONFIRMED, likely):** Possibly X" in text
    assert "- **Uncertain / unknown:**\n  - Probable root cause is not confirmed (assessed as likely)" in text
    assert "[observed] TC-001:F1:" in text


# --- graph -------------------------------------------------------------------------


def test_graph_produces_bug_reports_and_sends_them_to_review():
    graph = build_qa_graph(
        {"test_executor": make_mock_test_executor({"TC-007": "failed"}), "confidence_checker": make_confidence_checker(threshold=0.7)},
        llm=FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
    )
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)
    config = run_config(requirement.id)

    paused = graph.invoke(initial_state(requirement), config)

    [bug] = paused["bug_reports"]
    assert bug.title == "Password reset confirm does not reject an already-used reset token"
    assert bug.severity == "high" and bug.priority == "p1"
    assert bug.related_bugs == [TOKEN_REUSE_BUG]  # the invented BUG-9999 was dropped
    [interrupt] = paused["__interrupt__"]
    assert interrupt.value["bug_reports"][0]["id"] == "BR-1"
    assert "Sensitive action requires approval: report_high_severity_bugs (BR-1)" in paused["confidence_score"].reasons

    state = graph.invoke(Command(resume={"decision": "approve"}), config)
    report = state["final_report"]
    assert [b.id for b in report.bug_reports] == ["BR-1"]
    assert "## Bug reports" in report.markdown
    assert "Probable root cause (UNCONFIRMED" in report.markdown


def test_bug_environment_model():
    env = BugEnvironment(app_env="local", recorded_at=datetime.now(UTC))

    assert env.targets == [] and env.unknowns == []
    assert ProbableCause(description="d", reasoning="r", likelihood="possible").supporting_evidence == []
