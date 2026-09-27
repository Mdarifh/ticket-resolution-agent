"""Human-in-the-loop: pause on low confidence or sensitive actions, resume on a human decision."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from qa_agent.analysis.review_request import parse_sensitive_actions
from qa_agent.config import get_settings
from qa_agent.domain import Requirement
from qa_agent.domain.review import HumanReview, HumanReviewDecision
from qa_agent.graph import build_qa_graph, initial_state, run_config
from qa_agent.graph.checkpointer import create_default_checkpointer
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from qa_agent.graph.review import get_pending_review, submit_review_decision
from tests.conftest import PASSWORD_RESET_REQUIREMENT, password_reset_responses, repo_knowledge_base
from tests.fakes import FakeStructuredChatModel

TOKEN_REUSE_BUG = "previous_bugs/BUG-1042-reset-token-reusable.md"
REVIEW_FIELDS = {
    "review_id",
    "reason",
    "ai_recommendation",
    "evidence",
    "confidence",
    "proposed_action",
    "human_decision",
    "reviewer_comment",
}


def _graph(threshold=0.7, sensitive=None, checkpointer=None, llm=None):
    """TC-007 fails; the RCA (fake LLM) has confidence 0.62 and the bug report is high severity."""
    nodes = {
        "test_executor": make_mock_test_executor({"TC-007": "failed"}),
        "confidence_checker": make_confidence_checker(threshold=threshold, sensitive_actions=sensitive),
    }
    return build_qa_graph(
        nodes,
        llm=llm or FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
        checkpointer=checkpointer,
    )


def _start(graph):
    run_id = uuid4().hex
    state = graph.invoke(initial_state(Requirement(id=run_id, text=PASSWORD_RESET_REQUIREMENT)), run_config(run_id))
    return run_id, state


# --- model ---------------------------------------------------------------------


def test_human_review_has_the_required_fields():
    assert REVIEW_FIELDS <= set(HumanReview.model_fields)


@pytest.mark.parametrize("raw", ["approve", " Approve ", "APPROVE"])
def test_decisions_are_normalized(raw):
    assert HumanReviewDecision(decision=raw).decision == "APPROVE"


def test_only_the_three_decisions_are_accepted():
    for decision in ["APPROVE", "REJECT"]:
        HumanReviewDecision(decision=decision)
    HumanReviewDecision(decision="REQUEST_REANALYSIS", comment="check the cache")
    for bad in ["maybe", "request_changes", ""]:
        with pytest.raises(ValidationError):
            HumanReviewDecision(decision=bad)


def test_reanalysis_requires_a_comment():
    for comment in [None, "", "   "]:
        with pytest.raises(ValidationError, match="requires a comment"):
            HumanReviewDecision(decision="REQUEST_REANALYSIS", comment=comment)


def test_sensitive_actions_setting_is_validated():
    assert parse_sensitive_actions(" report_bugs , report_high_severity_bugs,") == {
        "report_bugs",
        "report_high_severity_bugs",
    }
    assert parse_sensitive_actions("") == set()
    with pytest.raises(ValueError, match="Unknown sensitive action"):
        parse_sensitive_actions("delete_database")


# --- when the workflow pauses ----------------------------------------------------


def test_low_confidence_alone_pauses():
    graph = _graph(threshold=0.7, sensitive=set())

    run_id, state = _start(graph)

    review = get_pending_review(graph, run_id)
    assert review.reasons == ["Confidence 0.62 is below threshold 0.70"]
    assert review.sensitive_actions == []
    assert "final_report" not in state


def test_sensitive_action_alone_pauses():
    graph = _graph(threshold=0.5, sensitive={"report_high_severity_bugs"})

    run_id, _ = _start(graph)

    review = get_pending_review(graph, run_id)
    assert review.reasons == ["Sensitive action requires approval: report_high_severity_bugs (BR-1)"]
    assert review.sensitive_actions == ["report_high_severity_bugs"]


def test_no_trigger_means_no_pause():
    graph = _graph(threshold=0.5, sensitive=set())

    run_id, state = _start(graph)

    assert get_pending_review(graph, run_id) is None
    assert state["human_review"] is None
    assert state["final_report"].status == "failed"
    assert state["final_report"].human_decision is None
    assert "human_review" not in state["execution_metadata"].nodes_visited


def test_threshold_and_sensitive_actions_come_from_settings(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("SENSITIVE_ACTIONS", "report_bugs")
    get_settings.cache_clear()
    try:
        graph = build_qa_graph(
            {"test_executor": make_mock_test_executor({"TC-007": "failed"})},
            llm=FakeStructuredChatModel(responses=password_reset_responses()),
            knowledge_base=repo_knowledge_base(),
        )
        run_id, _ = _start(graph)
    finally:
        get_settings.cache_clear()

    review = get_pending_review(graph, run_id)
    # 0.62 >= 0.5 so confidence is fine, but every bug report now needs approval.
    assert review.reasons == ["Sensitive action requires approval: report_bugs (BR-1)"]


# --- the review request ----------------------------------------------------------


def test_review_request_explains_recommendation_evidence_and_action():
    graph = _graph()

    run_id, state = _start(graph)

    review = get_pending_review(graph, run_id)
    assert REVIEW_FIELDS <= set(review.model_dump())
    assert review.review_id == f"{run_id}-review-1"
    assert review.reason == (
        "Confidence 0.62 is below threshold 0.70; "
        "Sensitive action requires approval: report_high_severity_bugs (BR-1)"
    )
    assert review.confidence == 0.62
    assert review.ai_recommendation.startswith(
        "Report 1 issue(s): BR-1 [product bug, high/p1] Password reset confirm does not reject an already-used reset token"
    )
    assert "Root cause analysis (confidence 0.62, unconfirmed)" in review.ai_recommendation
    assert review.proposed_action == "Include 1 bug report(s) in the final QA report; flag BR-1 as high/critical severity"
    assert review.actions == ["report_bugs", "report_high_severity_bugs"]
    assert review.evidence and review.evidence[0].kind == "observed"
    assert review.evidence[0].detail.startswith("Test reported status 'failed'")
    assert review.human_decision is None and review.reviewer_comment is None


def test_interrupt_payload_carries_review_and_bug_reports():
    graph = _graph()

    _, state = _start(graph)

    [pending] = state["__interrupt__"]
    assert pending.value["review"]["reason"].startswith("Confidence 0.62")
    assert [b["id"] for b in pending.value["bug_reports"]] == ["BR-1"]


def test_paused_state_is_checkpointed():
    graph = _graph()

    run_id, _ = _start(graph)

    snapshot = graph.get_state(run_config(run_id))
    assert snapshot.next == ("human_review",)
    assert snapshot.values["human_review"].human_decision is None
    assert snapshot.values["execution_metadata"].status == "running"


# --- decisions --------------------------------------------------------------------


def test_approve_finishes_the_run_with_the_decision_recorded():
    graph = _graph()
    run_id, _ = _start(graph)

    state = submit_review_decision(graph, run_id, {"decision": "APPROVE", "comment": "Matches BUG-1042", "reviewer": "qa-lead"})

    report = state["final_report"]
    assert report.status == "failed"
    assert report.human_decision == "APPROVE"
    assert report.human_review.reviewer == "qa-lead"
    assert report.human_review.reviewer_comment == "Matches BUG-1042"
    assert report.human_review.decided_at is not None
    assert [b.id for b in report.bug_reports] == ["BR-1"]
    assert [r.human_decision for r in state["review_history"]] == ["APPROVE"]
    assert f"- {run_id}-review-1: **APPROVE** by qa-lead" in report.markdown
    assert get_pending_review(graph, run_id) is None
    assert graph.get_state(run_config(run_id)).next == ()


def test_reject_finishes_the_run_as_rejected():
    graph = _graph()
    run_id, _ = _start(graph)

    state = submit_review_decision(graph, run_id, HumanReviewDecision(decision="REJECT", comment="Test data was stale"))

    report = state["final_report"]
    assert report.status == "rejected"
    assert report.human_decision == "REJECT"
    assert report.human_review.reviewer_comment == "Test data was stale"
    assert state["execution_metadata"].nodes_visited[-2:] == ["human_review", "final_report_generator"]


def test_reanalysis_reruns_root_cause_analysis_with_the_comment_and_pauses_again():
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    graph = _graph(llm=llm)
    run_id, _ = _start(graph)

    paused = submit_review_decision(
        graph, run_id, {"decision": "REQUEST_REANALYSIS", "comment": "Check whether the fixture token was reset overnight"}
    )

    visited = paused["execution_metadata"].nodes_visited
    assert visited[-4:] == ["human_review", "root_cause_analyzer", "bug_report_generator", "confidence_checker"]
    rca_prompts = llm.calls_for("RootCauseDraft")
    assert len(rca_prompts) == 2
    assert "it does not count as evidence):\nnone" in rca_prompts[0][-1].content
    assert "- Check whether the fixture token was reset overnight" in rca_prompts[1][-1].content
    bug_prompt = llm.calls_for("BugNarrativeDraft")[-1][-1].content
    assert "- Check whether the fixture token was reset overnight" in bug_prompt
    [bug] = paused["bug_reports"]
    assert bug.revision == 1
    assert bug.reviewer_feedback == ["Check whether the fixture token was reset overnight"]

    second = get_pending_review(graph, run_id)
    assert second.review_id == f"{run_id}-review-2"
    assert second.human_decision is None

    state = submit_review_decision(graph, run_id, {"decision": "approve", "reviewer": "qa-lead"})

    assert [r.human_decision for r in state["review_history"]] == ["REQUEST_REANALYSIS", "APPROVE"]
    assert [r.review_id for r in state["review_history"]] == [f"{run_id}-review-1", f"{run_id}-review-2"]
    assert state["final_report"].human_decision == "APPROVE"
    assert "**REQUEST_REANALYSIS**" in state["final_report"].markdown


def test_reviewer_comment_is_not_treated_as_evidence():
    graph = _graph()
    run_id, _ = _start(graph)

    paused = submit_review_decision(graph, run_id, {"decision": "REQUEST_REANALYSIS", "comment": "Redis was down"})

    evidence = [e.detail for f in paused["root_cause_analysis"].findings for e in f.evidence]
    assert all("Redis was down" not in detail for detail in evidence)


# --- resume -----------------------------------------------------------------------


def test_resume_from_a_fresh_graph_sharing_the_checkpointer():
    """Simulates a restart: the paused state lives in the checkpointer, not the graph object."""
    saver = create_default_checkpointer()
    run_id, _ = _start(_graph(checkpointer=saver))

    restarted = _graph(checkpointer=saver)
    review = get_pending_review(restarted, run_id)
    state = submit_review_decision(restarted, run_id, {"decision": "APPROVE"})

    assert review.review_id == f"{run_id}-review-1"
    assert state["final_report"].human_decision == "APPROVE"
    visited = state["execution_metadata"].nodes_visited
    assert visited.count("root_cause_analyzer") == 1  # nothing before the pause re-ran
    assert visited[-2:] == ["human_review", "final_report_generator"]


def test_other_runs_are_unaffected_by_a_decision():
    graph = _graph()
    first, _ = _start(graph)
    second, _ = _start(graph)

    submit_review_decision(graph, first, {"decision": "APPROVE"})

    assert get_pending_review(graph, first) is None
    assert get_pending_review(graph, second).review_id == f"{second}-review-1"


def test_invalid_decision_is_rejected_and_run_stays_paused():
    graph = _graph()
    run_id, _ = _start(graph)

    with pytest.raises(ValidationError):
        submit_review_decision(graph, run_id, {"decision": "REQUEST_REANALYSIS"})  # no comment

    assert get_pending_review(graph, run_id) is not None


def test_resuming_a_run_that_is_not_paused_fails():
    graph = _graph(threshold=0.5, sensitive=set())
    finished, _ = _start(graph)

    with pytest.raises(ValueError, match="not waiting for a human review"):
        submit_review_decision(graph, finished, {"decision": "APPROVE"})
    with pytest.raises(ValueError, match="not waiting"):
        submit_review_decision(graph, "unknown-run", {"decision": "APPROVE"})
