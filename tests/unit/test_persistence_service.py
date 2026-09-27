"""RunService + PersistenceService: workflow state recorded through the repositories."""

import pytest
from sqlalchemy import func, select

from qa_agent.graph import build_qa_graph
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from qa_agent.persistence.db import session_scope
from qa_agent.persistence.models import AgentRun, BugReportRecord, HumanReviewRecord, TestCaseRecord, TestResultRecord
from qa_agent.persistence.repositories import (
    AgentRunRepository,
    BugReportRepository,
    FailureRepository,
    HumanReviewRepository,
    TestCaseRepository,
)
from qa_agent.persistence.service import PersistenceService
from qa_agent.services.run_service import RunService
from tests.conftest import PASSWORD_RESET_REQUIREMENT, password_reset_responses, repo_knowledge_base
from tests.fakes import FakeStructuredChatModel


def _service(db_sessions, *, failing=("TC-007",), threshold=0.7, sensitive=None, nodes=None) -> tuple[RunService, PersistenceService]:
    graph = build_qa_graph(
        {
            "test_executor": make_mock_test_executor({tc: "failed" for tc in failing}),
            "confidence_checker": make_confidence_checker(threshold=threshold, sensitive_actions=sensitive),
            **(nodes or {}),
        },
        llm=FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
    )
    persistence = PersistenceService(db_sessions)
    return RunService(graph, persistence), persistence


def _count(db_sessions, model) -> int:
    with session_scope(db_sessions) as session:
        return session.scalar(select(func.count()).select_from(model))


def test_paused_run_is_recorded_as_awaiting_review(db_sessions):
    service, persistence = _service(db_sessions)

    outcome = service.start(PASSWORD_RESET_REQUIREMENT, project="shop", metadata={"ticket": "QA-1"})

    assert outcome.status == "awaiting_review"
    summary = persistence.get_run_summary(outcome.run_id)
    assert (summary.project, summary.requirement, summary.status) == ("shop", PASSWORD_RESET_REQUIREMENT, "awaiting_review")
    assert summary.test_counts == {"total": 8, "passed": 6, "failed": 1, "errored": 0, "skipped": 1}
    assert summary.failures == ["TC-007"]
    assert summary.bug_reports == [
        {"id": "BR-1", "revision": 0, "severity": "high", "status": "draft",
         "title": "Password reset confirm does not reject an already-used reset token"}
    ]
    assert summary.pending_review == f"{outcome.run_id}-review-1"
    assert summary.confidence == 0.62
    with session_scope(db_sessions) as session:
        run = AgentRunRepository(session).get(outcome.run_id)
        assert run.requirement_analysis["feature"] == "Password reset"
        assert run.root_cause_analysis["findings"][0]["test_case_ids"] == ["TC-007"]
        assert run.final_report is None and run.finished_at is None
        [(_, failure)] = FailureRepository(session).list_for_run(outcome.run_id)
        assert failure.observed_facts[0]["id"] == "TC-007:F1"


def test_approve_completes_the_run_and_approves_bug_reports(db_sessions):
    service, persistence = _service(db_sessions)
    run_id = service.start(PASSWORD_RESET_REQUIREMENT).run_id

    outcome = service.decide(run_id, {"decision": "APPROVE", "reviewer": "qa-lead", "comment": "Confirmed"})

    assert outcome.status == "completed"
    summary = persistence.get_run_summary(run_id)
    assert (summary.status, summary.outcome) == ("completed", "failed")
    assert [b["status"] for b in summary.bug_reports] == ["approved"]
    assert summary.reviews == [{"review_id": f"{run_id}-review-1", "decision": "APPROVE", "reviewer": "qa-lead"}]
    assert summary.pending_review is None
    with session_scope(db_sessions) as session:
        run = AgentRunRepository(session).get(run_id)
        assert run.finished_at is not None
        assert run.final_report["human_decision"] == "APPROVE"
        assert run.nodes_visited[-1] == "final_report_generator"


def test_reject_marks_bug_reports_rejected(db_sessions):
    service, persistence = _service(db_sessions)
    run_id = service.start(PASSWORD_RESET_REQUIREMENT).run_id

    service.decide(run_id, {"decision": "REJECT", "comment": "stale data"})

    summary = persistence.get_run_summary(run_id)
    assert summary.outcome == "rejected"
    assert [b["status"] for b in summary.bug_reports] == ["rejected"]


def test_reanalysis_stores_both_reviews_and_both_revisions(db_sessions):
    service, _ = _service(db_sessions)
    run_id = service.start(PASSWORD_RESET_REQUIREMENT).run_id

    paused = service.decide(run_id, {"decision": "REQUEST_REANALYSIS", "comment": "check fixture token"})
    service.decide(run_id, {"decision": "APPROVE"})

    assert paused.status == "awaiting_review"
    with session_scope(db_sessions) as session:
        reviews = HumanReviewRepository(session).list_for_run(run_id)
        assert [(r.review_key, r.human_decision) for r in reviews] == [
            (f"{run_id}-review-1", "REQUEST_REANALYSIS"),
            (f"{run_id}-review-2", "APPROVE"),
        ]
        assert reviews[0].reviewer_comment == "check fixture token"
        assert [(b.revision, b.status) for b in BugReportRepository(session).list_all(run_id)] == [
            (0, "superseded"),
            (1, "approved"),
        ]


def test_successful_run_is_completed_without_bugs_or_reviews(db_sessions):
    service, persistence = _service(db_sessions, failing=())

    outcome = service.start(PASSWORD_RESET_REQUIREMENT)

    summary = persistence.get_run_summary(outcome.run_id)
    assert (summary.status, summary.outcome) == ("completed", "passed")
    assert summary.bug_reports == [] and summary.reviews == []
    assert _count(db_sessions, TestCaseRecord) == 8
    assert _count(db_sessions, TestResultRecord) == 8


def test_sync_is_idempotent(db_sessions):
    service, persistence = _service(db_sessions)
    run_id = service.start(PASSWORD_RESET_REQUIREMENT).run_id
    state = service.graph.get_state({"configurable": {"thread_id": run_id}}).values

    persistence.sync_state(run_id, state, awaiting_review=True)
    persistence.sync_state(run_id, state, awaiting_review=True)

    assert _count(db_sessions, TestCaseRecord) == 8
    assert _count(db_sessions, TestResultRecord) == 8
    assert _count(db_sessions, BugReportRecord) == 1
    assert _count(db_sessions, HumanReviewRecord) == 1


def test_runs_are_kept_apart(db_sessions):
    service, _ = _service(db_sessions)
    first = service.start(PASSWORD_RESET_REQUIREMENT, project="shop").run_id
    second = service.start(PASSWORD_RESET_REQUIREMENT, project="shop").run_id

    service.decide(first, {"decision": "APPROVE"})

    with session_scope(db_sessions) as session:
        runs = AgentRunRepository(session)
        assert (runs.get(first).status, runs.get(second).status) == ("completed", "awaiting_review")
        assert len(TestCaseRepository(session).list_for_run(first)) == 8
        assert len(TestCaseRepository(session).list_for_run(second)) == 8
        assert len(HumanReviewRepository(session).pending()) == 1


class Crash(Exception):
    pass


def test_crash_marks_the_run_failed(db_sessions, monkeypatch):
    service, _ = _service(db_sessions)

    def crash_stream(*args, **kwargs):  # e.g. the worker process dies mid-run
        raise Crash("worker died")

    monkeypatch.setattr(service.graph, "stream", crash_stream)

    with pytest.raises(Crash):
        service.start(PASSWORD_RESET_REQUIREMENT)

    with session_scope(db_sessions) as session:
        [run] = session.scalars(select(AgentRun)).all()
        assert run.status == "failed"
        assert run.error_message == "Crash: worker died"
        assert run.finished_at is not None


def test_invalid_decision_keeps_the_run_awaiting_review(db_sessions):
    service, persistence = _service(db_sessions)
    run_id = service.start(PASSWORD_RESET_REQUIREMENT).run_id

    with pytest.raises(ValueError):
        service.decide(run_id, {"decision": "REQUEST_REANALYSIS"})  # comment required

    assert persistence.get_run_summary(run_id).status == "awaiting_review"


def test_run_service_works_without_persistence():
    graph = build_qa_graph(
        {"test_executor": make_mock_test_executor()},
        llm=FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
    )

    outcome = RunService(graph).start(PASSWORD_RESET_REQUIREMENT)

    assert outcome.status == "completed"
    assert outcome.final_report.status == "passed"


def test_unknown_run_summary_is_none(db_sessions):
    assert PersistenceService(db_sessions).get_run_summary("nope") is None


def test_run_paused_before_execution_is_recorded_as_awaiting_execution(db_sessions):
    service, persistence = _service(db_sessions)

    outcome = service.start(PASSWORD_RESET_REQUIREMENT, pause_before_execution=True)

    assert outcome.status == "awaiting_execution"
    summary = persistence.get_run_summary(outcome.run_id)
    assert summary.status == "awaiting_execution"
    assert summary.test_counts == {}
    assert _count(db_sessions, TestCaseRecord) == 8

    resumed = service.execute(outcome.run_id)

    assert resumed.status == "awaiting_review"
    assert persistence.get_run_summary(outcome.run_id).test_counts["total"] == 8
    with pytest.raises(ValueError, match="not waiting to execute"):
        service.execute(outcome.run_id)
