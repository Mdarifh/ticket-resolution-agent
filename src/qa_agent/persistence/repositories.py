"""Repositories: the only place that issues database queries.

Each repository works on a ``Session`` owned by the caller, so a service can
group several repository calls into one transaction. Domain models go in,
ORM records come out.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from qa_agent.domain import BugReport, ExecutionResult, FailureDetail, TestCase
from qa_agent.domain.review import HumanReview
from qa_agent.persistence.models import (
    AgentRun,
    BugReportRecord,
    FailureRecord,
    HumanReviewRecord,
    Project,
    RequirementRecord,
    TestCaseRecord,
    TestResultRecord,
    TestRunRecord,
)


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, project_id: uuid.UUID) -> Project | None:
        return self.session.get(Project, project_id)

    def get_by_name(self, name: str) -> Project | None:
        return self.session.scalar(select(Project).where(Project.name == name))

    def list_all(self) -> Sequence[Project]:
        return self.session.scalars(select(Project).order_by(Project.name)).all()

    def get_or_create(self, name: str, description: str | None = None) -> Project:
        project = self.get_by_name(name)
        if project is None:
            project = Project(name=name, description=description)
            self.session.add(project)
            self.session.flush()
        return project


class RequirementRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, project: Project, text: str, metadata: dict[str, Any] | None = None) -> RequirementRecord:
        record = RequirementRecord(project_id=project.id, text=text, metadata_=metadata or {})
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, requirement_id: uuid.UUID) -> RequirementRecord | None:
        return self.session.get(RequirementRecord, requirement_id)

    def list_for_project(self, project: Project) -> Sequence[RequirementRecord]:
        return self.session.scalars(
            select(RequirementRecord).where(RequirementRecord.project_id == project.id).order_by(RequirementRecord.created_at)
        ).all()


class AgentRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, run_id: str, project: Project, requirement: RequirementRecord) -> AgentRun:
        run = AgentRun(id=run_id, project_id=project.id, requirement_id=requirement.id, status="running")
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: str) -> AgentRun | None:
        return self.session.get(AgentRun, run_id)

    def require(self, run_id: str) -> AgentRun:
        run = self.get(run_id)
        if run is None:
            raise LookupError(f"Agent run {run_id!r} not found")
        return run

    def update(self, run_id: str, **fields: Any) -> AgentRun:
        run = self.require(run_id)
        for name, value in fields.items():
            setattr(run, name, value)
        self.session.flush()
        return run

    def list(self, *, status: str | None = None, project: Project | None = None, limit: int = 50) -> Sequence[AgentRun]:
        query = select(AgentRun).order_by(AgentRun.started_at.desc(), AgentRun.id).limit(limit)
        if status:
            query = query.where(AgentRun.status == status)
        if project:
            query = query.where(AgentRun.project_id == project.id)
        return self.session.scalars(query).all()


class TestCaseRepository:
    __test__ = False  # not a pytest test class

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_for_run(self, run: AgentRun, test_cases: Sequence[TestCase]) -> dict[str, TestCaseRecord]:
        """Insert or update test cases by (run, case key); returns records by case key."""
        existing = {r.case_key: r for r in self.list_for_run(run.id)}
        for case in test_cases:
            record = existing.get(case.test_case_id)
            fields = {
                "title": case.title,
                "description": case.description,
                "preconditions": list(case.preconditions),
                "test_data": [item.model_dump() for item in case.test_data],
                "steps": list(case.steps),
                "expected_result": case.expected_result,
                "priority": case.priority,
                "test_type": case.test_type,
                "automation_type": case.automation_type,
            }
            if record is None:
                record = TestCaseRecord(
                    agent_run_id=run.id, requirement_id=run.requirement_id, case_key=case.test_case_id, **fields
                )
                self.session.add(record)
                existing[case.test_case_id] = record
            else:
                for name, value in fields.items():
                    setattr(record, name, value)
        self.session.flush()
        return existing

    def list_for_run(self, run_id: str) -> Sequence[TestCaseRecord]:
        return self.session.scalars(
            select(TestCaseRecord).where(TestCaseRecord.agent_run_id == run_id).order_by(TestCaseRecord.case_key)
        ).all()


class TestRunRepository:
    __test__ = False  # not a pytest test class

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_run(self, run_id: str) -> TestRunRecord | None:
        return self.session.scalar(select(TestRunRecord).where(TestRunRecord.agent_run_id == run_id))

    def get_or_create(self, run_id: str) -> TestRunRecord:
        test_run = self.get_for_run(run_id)
        if test_run is None:
            test_run = TestRunRecord(agent_run_id=run_id)
            self.session.add(test_run)
            self.session.flush()
        return test_run

    def update_counts(self, test_run: TestRunRecord, results: Sequence[ExecutionResult]) -> TestRunRecord:
        statuses = [r.status for r in results]
        test_run.total = len(statuses)
        test_run.passed = statuses.count("passed")
        test_run.failed = statuses.count("failed")
        test_run.errored = statuses.count("error")
        test_run.skipped = statuses.count("skipped")
        self.session.flush()
        return test_run


class TestResultRepository:
    __test__ = False  # not a pytest test class

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self, test_run: TestRunRecord, cases: dict[str, TestCaseRecord], results: Sequence[ExecutionResult]
    ) -> dict[str, TestResultRecord]:
        """Insert or update results by (test run, test case); returns records by case key."""
        existing = self._by_case_key(test_run.id)
        for result in results:
            case = cases.get(result.test_case_id)
            if case is None:
                raise LookupError(f"Test case {result.test_case_id!r} is not stored for this run")
            fields = {
                "status": result.status,
                "duration_ms": result.duration_ms,
                "message": result.message,
                "evidence": result.evidence,
            }
            record = existing.get(result.test_case_id)
            if record is None:
                record = TestResultRecord(test_run_id=test_run.id, test_case_id=case.id, **fields)
                self.session.add(record)
                existing[result.test_case_id] = record
            else:
                for name, value in fields.items():
                    setattr(record, name, value)
        self.session.flush()
        return existing

    def list_for_run(self, run_id: str) -> Sequence[tuple[str, TestResultRecord]]:
        rows = self.session.execute(
            select(TestCaseRecord.case_key, TestResultRecord)
            .join(TestCaseRecord, TestResultRecord.test_case_id == TestCaseRecord.id)
            .join(TestRunRecord, TestResultRecord.test_run_id == TestRunRecord.id)
            .where(TestRunRecord.agent_run_id == run_id)
            .order_by(TestCaseRecord.case_key)
        ).all()
        return [(key, record) for key, record in rows]

    def _by_case_key(self, test_run_id: uuid.UUID) -> dict[str, TestResultRecord]:
        rows = self.session.execute(
            select(TestCaseRecord.case_key, TestResultRecord)
            .join(TestCaseRecord, TestResultRecord.test_case_id == TestCaseRecord.id)
            .where(TestResultRecord.test_run_id == test_run_id)
        ).all()
        return {key: record for key, record in rows}


class FailureRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, result: TestResultRecord, failure: FailureDetail) -> FailureRecord:
        record = self.session.scalar(select(FailureRecord).where(FailureRecord.test_result_id == result.id))
        fields = {
            "category": failure.category,
            "message": failure.message,
            "signal": failure.signal,
            "expected_result": failure.expected_result,
            "actual_result": failure.actual_result,
            "observed_facts": [fact.model_dump() for fact in failure.observed_facts],
        }
        if record is None:
            record = FailureRecord(test_result_id=result.id, **fields)
            self.session.add(record)
        else:
            for name, value in fields.items():
                setattr(record, name, value)
        self.session.flush()
        return record

    def list_for_run(self, run_id: str) -> Sequence[tuple[str, FailureRecord]]:
        rows = self.session.execute(
            select(TestCaseRecord.case_key, FailureRecord)
            .join(TestResultRecord, FailureRecord.test_result_id == TestResultRecord.id)
            .join(TestCaseRecord, TestResultRecord.test_case_id == TestCaseRecord.id)
            .where(TestCaseRecord.agent_run_id == run_id)
            .order_by(TestCaseRecord.case_key)
        ).all()
        return [(key, record) for key, record in rows]


class BugReportRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save_revision(self, run_id: str, reports: Sequence[BugReport]) -> list[BugReportRecord]:
        """Store the reports of one revision; older revisions become ``superseded``."""
        saved = []
        for report in reports:
            record = self.session.scalar(
                select(BugReportRecord).where(
                    BugReportRecord.agent_run_id == run_id,
                    BugReportRecord.report_key == report.id,
                    BugReportRecord.revision == report.revision,
                )
            )
            fields = {
                "title": report.title,
                "summary": report.summary,
                "report_type": report.report_type,
                "severity": report.severity,
                "priority": report.priority,
                "affected_component": report.affected_component,
                "confidence": report.confidence,
                "related_test_case_ids": list(report.related_test_case_ids),
                "payload": report.model_dump(mode="json"),
            }
            if record is None:
                record = BugReportRecord(agent_run_id=run_id, report_key=report.id, revision=report.revision, **fields)
                self.session.add(record)
            else:
                for name, value in fields.items():
                    setattr(record, name, value)
            saved.append(record)
        self.session.flush()

        if reports:
            latest = max(r.revision for r in reports)
            self.session.execute(
                update(BugReportRecord)
                .where(BugReportRecord.agent_run_id == run_id, BugReportRecord.revision < latest)
                .values(status="superseded")
            )
        return saved

    def set_status_of_latest(self, run_id: str, status: str) -> int:
        """Set the status of the current (non-superseded) reports; returns how many changed."""
        result = self.session.execute(
            update(BugReportRecord)
            .where(BugReportRecord.agent_run_id == run_id, BugReportRecord.status != "superseded")
            .values(status=status)
        )
        return result.rowcount

    def list_latest(self, run_id: str) -> Sequence[BugReportRecord]:
        return self.session.scalars(
            select(BugReportRecord)
            .where(BugReportRecord.agent_run_id == run_id, BugReportRecord.status != "superseded")
            .order_by(BugReportRecord.report_key)
        ).all()

    def list_all(self, run_id: str) -> Sequence[BugReportRecord]:
        return self.session.scalars(
            select(BugReportRecord)
            .where(BugReportRecord.agent_run_id == run_id)
            .order_by(BugReportRecord.revision, BugReportRecord.report_key)
        ).all()

    def find_by_severity(self, severities: Sequence[str], *, limit: int = 100) -> Sequence[BugReportRecord]:
        return self.session.scalars(
            select(BugReportRecord)
            .where(BugReportRecord.severity.in_(severities), BugReportRecord.status != "superseded")
            .order_by(BugReportRecord.created_at.desc())
            .limit(limit)
        ).all()


class HumanReviewRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, run_id: str, review: HumanReview) -> HumanReviewRecord:
        """Create the review when requested; fill in the decision when it is made."""
        record = self.get(review.review_id)
        fields = {
            "reason": review.reason,
            "ai_recommendation": review.ai_recommendation,
            "evidence": [item.model_dump(mode="json") for item in review.evidence],
            "confidence": review.confidence,
            "proposed_action": review.proposed_action,
            "actions": list(review.actions),
            "sensitive_actions": list(review.sensitive_actions),
            "requested_at": review.requested_at,
        }
        if record is None:
            record = HumanReviewRecord(agent_run_id=run_id, review_key=review.review_id, **fields)
            self.session.add(record)
        else:
            for name, value in fields.items():
                setattr(record, name, value)
        if review.human_decision is not None:  # never erase a recorded decision
            record.human_decision = review.human_decision
            record.reviewer = review.reviewer
            record.reviewer_comment = review.reviewer_comment
            record.decided_at = review.decided_at or datetime.now(UTC)
        self.session.flush()
        return record

    def get(self, review_key: str) -> HumanReviewRecord | None:
        return self.session.scalar(select(HumanReviewRecord).where(HumanReviewRecord.review_key == review_key))

    def list_for_run(self, run_id: str) -> Sequence[HumanReviewRecord]:
        return self.session.scalars(
            select(HumanReviewRecord)
            .where(HumanReviewRecord.agent_run_id == run_id)
            .order_by(HumanReviewRecord.requested_at, HumanReviewRecord.review_key)
        ).all()

    def pending(self, *, run_id: str | None = None) -> Sequence[HumanReviewRecord]:
        query = select(HumanReviewRecord).where(HumanReviewRecord.human_decision.is_(None))
        if run_id:
            query = query.where(HumanReviewRecord.agent_run_id == run_id)
        return self.session.scalars(query.order_by(HumanReviewRecord.requested_at)).all()
