"""Repository operations on SQLite, and on PostgreSQL when TEST_DATABASE_URL is set."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from qa_agent.domain import ExecutionResult, FailureDetail, ObservedFact, TestCase
from qa_agent.domain.review import HumanReview
from qa_agent.persistence.db import session_scope
from qa_agent.persistence.models import (
    AgentRun,
    BugReportRecord,
    FailureRecord,
    HumanReviewRecord,
    Project,
    TestCaseRecord,
    TestResultRecord,
)
from qa_agent.persistence.repositories import (
    AgentRunRepository,
    BugReportRepository,
    FailureRepository,
    HumanReviewRepository,
    ProjectRepository,
    RequirementRepository,
    TestCaseRepository,
    TestResultRepository,
    TestRunRepository,
)
from tests.conftest import load_fixture


def _test_cases() -> list[TestCase]:
    raw = load_fixture("password_reset", "test_suite.json")["test_cases"]
    return [TestCase.model_validate({**tc, "test_case_id": f"TC-{i:03d}"}) for i, tc in enumerate(raw, start=1)]


def _new_run(session, run_id="run-1", project="shop") -> AgentRun:
    project_row = ProjectRepository(session).get_or_create(project)
    requirement = RequirementRepository(session).create(project_row, "Reset password via email", {"source": "jira"})
    return AgentRunRepository(session).create(run_id, project_row, requirement)


def _bug_report(key="BR-1", revision=0, severity="high"):
    from tests.unit.test_bug_report import _context, _generate

    context, _, _ = _context()
    [report] = _generate(context)[0]
    return report.model_copy(update={"id": key, "revision": revision, "severity": severity})


def _review(key="run-1-review-1", decision=None, comment=None) -> HumanReview:
    return HumanReview(
        review_id=key,
        reason="Confidence 0.62 is below threshold 0.70",
        ai_recommendation="Report 1 issue(s)",
        confidence=0.62,
        proposed_action="Include 1 bug report(s)",
        actions=["report_bugs"],
        requested_at=datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
        human_decision=decision,
        reviewer_comment=comment,
        reviewer="qa-lead" if decision else None,
        decided_at=datetime(2026, 9, 26, 12, 5, tzinfo=UTC) if decision else None,
    )


# --- projects, requirements, agent runs ------------------------------------------


def test_project_get_or_create_is_idempotent(db_sessions):
    with session_scope(db_sessions) as session:
        first = ProjectRepository(session).get_or_create("shop", "Demo shop")
        second = ProjectRepository(session).get_or_create("shop")

        assert first.id == second.id
        assert session.scalar(select(func.count()).select_from(Project)) == 1


def test_project_names_are_unique(db_sessions):
    with pytest.raises(IntegrityError):
        with session_scope(db_sessions) as session:
            session.add_all([Project(name="dup"), Project(name="dup")])
            session.flush()


def test_requirement_keeps_text_and_metadata(db_sessions):
    with session_scope(db_sessions) as session:
        project = ProjectRepository(session).get_or_create("shop")
        created = RequirementRepository(session).create(project, "Reset password", {"priority": "high"})
    with session_scope(db_sessions) as session:
        stored = RequirementRepository(session).get(created.id)
        assert (stored.text, stored.metadata_) == ("Reset password", {"priority": "high"})
        assert [r.id for r in RequirementRepository(session).list_for_project(stored.project)] == [created.id]


def test_agent_run_create_update_and_list(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session, "run-1")
        _new_run(session, "run-2")
        AgentRunRepository(session).update("run-2", status="awaiting_review", confidence=0.62)

    with session_scope(db_sessions) as session:
        runs = AgentRunRepository(session)
        assert runs.get("run-1").status == "running"
        assert [r.id for r in runs.list(status="awaiting_review")] == ["run-2"]
        assert {r.id for r in runs.list(project=ProjectRepository(session).get_by_name("shop"))} == {"run-1", "run-2"}
        assert runs.get("missing") is None
        with pytest.raises(LookupError):
            runs.require("missing")


def test_agent_run_status_is_constrained(db_sessions):
    with pytest.raises(IntegrityError):
        with session_scope(db_sessions) as session:
            _new_run(session)
            AgentRunRepository(session).update("run-1", status="exploded")


def test_failed_unit_of_work_is_rolled_back(db_sessions):
    with pytest.raises(RuntimeError):
        with session_scope(db_sessions) as session:
            _new_run(session)
            raise RuntimeError("boom")

    with session_scope(db_sessions) as session:
        assert AgentRunRepository(session).get("run-1") is None


# --- test cases, test runs, results, failures ------------------------------------


def test_test_cases_upsert_by_run_and_key(db_sessions):
    cases = _test_cases()
    with session_scope(db_sessions) as session:
        run = _new_run(session)
        TestCaseRepository(session).upsert_for_run(run, cases)
        changed = [cases[0].model_copy(update={"automation_type": "ui", "title": "Renamed"})]
        TestCaseRepository(session).upsert_for_run(run, changed)

    with session_scope(db_sessions) as session:
        stored = TestCaseRepository(session).list_for_run("run-1")
        assert [c.case_key for c in stored] == [f"TC-{i:03d}" for i in range(1, 9)]
        assert (stored[0].title, stored[0].automation_type) == ("Renamed", "ui")
        assert stored[0].test_data == [{"name": "email", "value": "registered.user@example.com"}]


def test_results_counts_and_failures(db_sessions):
    results = [
        ExecutionResult(test_case_id="TC-001", status="passed", duration_ms=12),
        ExecutionResult(test_case_id="TC-002", status="failed", message="HTTP 500", evidence={"status_code": 500}),
        ExecutionResult(test_case_id="TC-003", status="skipped"),
        ExecutionResult(test_case_id="TC-004", status="error", message="timeout"),
    ]
    failure = FailureDetail(
        test_case_id="TC-002",
        status="failed",
        category="server_error",
        actual_result="HTTP 500",
        observed_facts=[ObservedFact(id="TC-002:F1", source="api.status_code", detail="returned HTTP 500")],
    )
    with session_scope(db_sessions) as session:
        run = _new_run(session)
        cases = TestCaseRepository(session).upsert_for_run(run, _test_cases())
        test_runs = TestRunRepository(session)
        test_run = test_runs.get_or_create(run.id)
        test_runs.update_counts(test_run, results)
        stored = TestResultRepository(session).upsert(test_run, cases, results)
        FailureRepository(session).upsert(stored["TC-002"], failure)

    with session_scope(db_sessions) as session:
        test_run = TestRunRepository(session).get_for_run("run-1")
        assert (test_run.total, test_run.passed, test_run.failed, test_run.errored, test_run.skipped) == (4, 1, 1, 1, 1)
        by_key = dict(TestResultRepository(session).list_for_run("run-1"))
        assert by_key["TC-002"].evidence == {"status_code": 500}
        [(key, stored_failure)] = FailureRepository(session).list_for_run("run-1")
        assert key == "TC-002"
        assert stored_failure.category == "server_error"
        assert stored_failure.observed_facts[0]["detail"] == "returned HTTP 500"


def test_result_and_failure_upserts_do_not_duplicate(db_sessions):
    result = ExecutionResult(test_case_id="TC-001", status="failed", message="first")
    failure = FailureDetail(test_case_id="TC-001", status="failed", category="assertion_mismatch")
    with session_scope(db_sessions) as session:
        run = _new_run(session)
        cases = TestCaseRepository(session).upsert_for_run(run, _test_cases())
        test_run = TestRunRepository(session).get_or_create(run.id)
        for message in ["first", "second"]:
            stored = TestResultRepository(session).upsert(test_run, cases, [result.model_copy(update={"message": message})])
            FailureRepository(session).upsert(stored["TC-001"], failure)
        assert TestRunRepository(session).get_or_create(run.id).id == test_run.id

    with session_scope(db_sessions) as session:
        assert session.scalar(select(func.count()).select_from(TestResultRecord)) == 1
        assert session.scalar(select(func.count()).select_from(FailureRecord)) == 1
        assert session.scalar(select(TestResultRecord.message)) == "second"


def test_result_for_unknown_test_case_is_rejected(db_sessions):
    with session_scope(db_sessions) as session:
        run = _new_run(session)
        test_run = TestRunRepository(session).get_or_create(run.id)
        with pytest.raises(LookupError, match="TC-404"):
            TestResultRepository(session).upsert(test_run, {}, [ExecutionResult(test_case_id="TC-404", status="passed")])


def test_invalid_result_status_is_rejected_by_the_database(db_sessions):
    with pytest.raises(IntegrityError):
        with session_scope(db_sessions) as session:
            run = _new_run(session)
            cases = TestCaseRepository(session).upsert_for_run(run, _test_cases())
            test_run = TestRunRepository(session).get_or_create(run.id)
            session.add(TestResultRecord(test_run_id=test_run.id, test_case_id=cases["TC-001"].id, status="green"))
            session.flush()


# --- bug reports -------------------------------------------------------------------


def test_bug_report_revisions_supersede_older_ones(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session)
        bugs = BugReportRepository(session)
        bugs.save_revision("run-1", [_bug_report(revision=0)])
        bugs.save_revision("run-1", [_bug_report(revision=1, severity="critical")])

    with session_scope(db_sessions) as session:
        bugs = BugReportRepository(session)
        assert [(b.revision, b.status) for b in bugs.list_all("run-1")] == [(0, "superseded"), (1, "draft")]
        [latest] = bugs.list_latest("run-1")
        assert (latest.revision, latest.severity) == (1, "critical")
        assert latest.payload["id"] == "BR-1" and latest.payload["revision"] == 1


def test_bug_report_status_follows_review_and_is_queryable(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session)
        bugs = BugReportRepository(session)
        bugs.save_revision("run-1", [_bug_report("BR-1", severity="high"), _bug_report("BR-2", severity="low")])
        assert bugs.set_status_of_latest("run-1", "approved") == 2

    with session_scope(db_sessions) as session:
        bugs = BugReportRepository(session)
        assert {b.status for b in bugs.list_latest("run-1")} == {"approved"}
        assert [b.report_key for b in bugs.find_by_severity(["high", "critical"])] == ["BR-1"]


def test_saving_the_same_revision_twice_updates_in_place(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session)
        bugs = BugReportRepository(session)
        bugs.save_revision("run-1", [_bug_report()])
        bugs.save_revision("run-1", [_bug_report().model_copy(update={"title": "Better title"})])

    with session_scope(db_sessions) as session:
        [stored] = BugReportRepository(session).list_all("run-1")
        assert stored.title == "Better title"


def test_bug_report_severity_is_constrained(db_sessions):
    with pytest.raises(IntegrityError):
        with session_scope(db_sessions) as session:
            _new_run(session)
            [record] = BugReportRepository(session).save_revision("run-1", [_bug_report()])
            record.severity = "catastrophic"
            session.flush()


# --- human reviews -----------------------------------------------------------------


def test_review_is_created_pending_then_decided(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session)
        HumanReviewRepository(session).upsert("run-1", _review())
        assert [r.review_key for r in HumanReviewRepository(session).pending()] == ["run-1-review-1"]

    with session_scope(db_sessions) as session:
        HumanReviewRepository(session).upsert("run-1", _review(decision="REQUEST_REANALYSIS", comment="check cache"))

    with session_scope(db_sessions) as session:
        reviews = HumanReviewRepository(session)
        stored = reviews.get("run-1-review-1")
        assert (stored.human_decision, stored.reviewer, stored.reviewer_comment) == ("REQUEST_REANALYSIS", "qa-lead", "check cache")
        assert stored.decided_at is not None
        assert reviews.pending(run_id="run-1") == []
        assert session.scalar(select(func.count()).select_from(HumanReviewRecord)) == 1


def test_recorded_decision_is_never_erased(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session)
        reviews = HumanReviewRepository(session)
        reviews.upsert("run-1", _review(decision="APPROVE"))
        reviews.upsert("run-1", _review())  # a stale, undecided copy of the same review

    with session_scope(db_sessions) as session:
        assert HumanReviewRepository(session).get("run-1-review-1").human_decision == "APPROVE"


def test_reviews_are_listed_in_request_order(db_sessions):
    with session_scope(db_sessions) as session:
        _new_run(session)
        reviews = HumanReviewRepository(session)
        second = _review("run-1-review-2").model_copy(update={"requested_at": datetime(2026, 9, 26, 13, 0, tzinfo=UTC)})
        reviews.upsert("run-1", second)
        reviews.upsert("run-1", _review("run-1-review-1", decision="REQUEST_REANALYSIS", comment="x"))

        assert [r.review_key for r in reviews.list_for_run("run-1")] == ["run-1-review-1", "run-1-review-2"]


def test_invalid_decision_is_rejected_by_the_database(db_sessions):
    with pytest.raises(IntegrityError):
        with session_scope(db_sessions) as session:
            _new_run(session)
            record = HumanReviewRepository(session).upsert("run-1", _review())
            record.human_decision = "MAYBE"
            session.flush()


# --- cascades ----------------------------------------------------------------------


def test_deleting_an_agent_run_cascades_to_its_records(db_sessions):
    with session_scope(db_sessions) as session:
        run = _new_run(session)
        cases = TestCaseRepository(session).upsert_for_run(run, _test_cases())
        test_run = TestRunRepository(session).get_or_create(run.id)
        stored = TestResultRepository(session).upsert(test_run, cases, [ExecutionResult(test_case_id="TC-001", status="failed")])
        FailureRepository(session).upsert(stored["TC-001"], FailureDetail(test_case_id="TC-001", status="failed"))
        BugReportRepository(session).save_revision(run.id, [_bug_report()])
        HumanReviewRepository(session).upsert(run.id, _review())

    with session_scope(db_sessions) as session:
        session.delete(session.get(AgentRun, "run-1"))

    with session_scope(db_sessions) as session:
        for model in (TestCaseRecord, TestResultRecord, FailureRecord, BugReportRecord, HumanReviewRecord):
            assert session.scalar(select(func.count()).select_from(model)) == 0, model.__tablename__
