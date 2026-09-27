"""PersistenceService: maps workflow state onto the database through the repositories.

Graph nodes never touch the database. The run orchestrator calls this service
when a run starts, pauses for review, finishes or fails. ``sync_state`` is
idempotent: syncing the same state twice leaves the same rows.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from qa_agent.domain import Requirement
from qa_agent.persistence.db import session_scope
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

logger = logging.getLogger(__name__)

_BUG_STATUS_FOR_DECISION = {"APPROVE": "approved", "REJECT": "rejected"}


class RunRecordSummary(BaseModel):
    """Read model of a stored run (for APIs and reports)."""

    run_id: str
    project: str
    requirement: str
    status: str
    outcome: str | None = None
    confidence: float | None = None
    test_counts: dict[str, int] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    bug_reports: list[dict[str, Any]] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    pending_review: str | None = None


class PersistenceService:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    def start_run(
        self, requirement: Requirement, *, project_name: str = "default", project_description: str | None = None
    ) -> str:
        """Create project (if new), requirement and agent run rows; returns the run id."""
        with session_scope(self.factory) as session:
            project = ProjectRepository(session).get_or_create(project_name, project_description)
            record = RequirementRepository(session).create(project, requirement.text, requirement.metadata)
            AgentRunRepository(session).create(requirement.id, project, record)
        logger.info("DB: agent_runs %s created (project=%s)", requirement.id, project_name)
        return requirement.id

    def list_projects(self) -> list[dict[str, Any]]:
        with session_scope(self.factory) as session:
            return [
                {"name": p.name, "description": p.description, "created_at": p.created_at}
                for p in ProjectRepository(session).list_all()
            ]

    def create_project(self, name: str, description: str | None = None) -> None:
        with session_scope(self.factory) as session:
            ProjectRepository(session).get_or_create(name, description)

    def sync_state(
        self,
        run_id: str,
        state: Mapping[str, Any],
        *,
        awaiting_review: bool = False,
        awaiting_execution: bool = False,
    ) -> None:
        with session_scope(self.factory) as session:
            runs = AgentRunRepository(session)
            run = runs.require(run_id)

            cases = TestCaseRepository(session).upsert_for_run(run, state.get("test_cases") or [])
            results = state.get("execution_results") or []
            if results:
                test_runs = TestRunRepository(session)
                test_run = test_runs.get_or_create(run_id)
                test_runs.update_counts(test_run, results)
                stored = TestResultRepository(session).upsert(test_run, cases, results)
                failures = FailureRepository(session)
                for failure in state.get("failures") or []:
                    if failure.test_case_id in stored:
                        failures.upsert(stored[failure.test_case_id], failure)

            bug_repo = BugReportRepository(session)
            bug_reports = state.get("bug_reports") or []
            if bug_reports:
                bug_repo.save_revision(run_id, bug_reports)

            review_repo = HumanReviewRepository(session)
            for review in state.get("review_history") or []:
                review_repo.upsert(run_id, review)
            current = state.get("human_review")
            if current is not None:
                review_repo.upsert(run_id, current)

            final_report = state.get("final_report")
            decision = current.human_decision if current else None
            if final_report is not None and decision in _BUG_STATUS_FOR_DECISION:
                bug_repo.set_status_of_latest(run_id, _BUG_STATUS_FOR_DECISION[decision])

            metadata = state.get("execution_metadata")
            analysis = state.get("requirement_analysis")
            root_cause = state.get("root_cause_analysis")
            fields: dict[str, Any] = {
                "nodes_visited": list(metadata.nodes_visited) if metadata else [],
                "error_count": len(state.get("errors") or []),
                "requirement_analysis": analysis.model_dump(mode="json") if analysis else None,
                "root_cause_analysis": root_cause.model_dump(mode="json") if root_cause else None,
                "confidence": root_cause.confidence if root_cause else None,
            }
            if final_report is not None:
                fields.update(
                    status="completed",
                    outcome=final_report.status,
                    summary=final_report.summary,
                    final_report=final_report.model_dump(mode="json"),
                    finished_at=datetime.now(UTC),
                )
            else:
                if awaiting_review:
                    fields["status"] = "awaiting_review"
                elif awaiting_execution:
                    fields["status"] = "awaiting_execution"
                else:
                    fields["status"] = "running"
            runs.update(run_id, **fields)
        logger.info(
            "DB: agent_runs %s synced status=%s test_cases=%d results=%d bug_reports=%d",
            run_id,
            fields["status"],
            len(state.get("test_cases") or []),
            len(results),
            len(bug_reports),
        )

    def mark_failed(self, run_id: str, error: str) -> None:
        with session_scope(self.factory) as session:
            AgentRunRepository(session).update(
                run_id, status="failed", error_message=error, finished_at=datetime.now(UTC)
            )
        logger.error("DB: agent_runs %s marked failed: %s", run_id, error)

    def get_run_summary(self, run_id: str) -> RunRecordSummary | None:
        with session_scope(self.factory) as session:
            runs = AgentRunRepository(session)
            run = runs.get(run_id)
            if run is None:
                return None
            project = ProjectRepository(session).get(run.project_id)
            requirement = RequirementRepository(session).get(run.requirement_id)
            test_run = TestRunRepository(session).get_for_run(run_id)
            reviews = HumanReviewRepository(session).list_for_run(run_id)
            pending = [r.review_key for r in reviews if r.human_decision is None]
            return RunRecordSummary(
                run_id=run.id,
                project=project.name,
                requirement=requirement.text,
                status=run.status,
                outcome=run.outcome,
                confidence=run.confidence,
                test_counts=(
                    {
                        "total": test_run.total,
                        "passed": test_run.passed,
                        "failed": test_run.failed,
                        "errored": test_run.errored,
                        "skipped": test_run.skipped,
                    }
                    if test_run
                    else {}
                ),
                failures=[key for key, _ in FailureRepository(session).list_for_run(run_id)],
                bug_reports=[
                    {"id": b.report_key, "revision": b.revision, "severity": b.severity, "status": b.status, "title": b.title}
                    for b in BugReportRepository(session).list_latest(run_id)
                ],
                reviews=[
                    {"review_id": r.review_key, "decision": r.human_decision, "reviewer": r.reviewer}
                    for r in reviews
                ],
                pending_review=pending[-1] if pending else None,
            )
