"""ORM tables. Schema changes go through Alembic migrations (``migrations/``)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qa_agent.persistence.db import Base, JsonType

AGENT_RUN_STATUSES = ("running", "awaiting_execution", "awaiting_review", "completed", "failed")
TEST_RESULT_STATUSES = ("passed", "failed", "error", "skipped")
SEVERITIES = ("low", "medium", "high", "critical")
PRIORITIES = ("p0", "p1", "p2", "p3")
BUG_STATUSES = ("draft", "approved", "rejected", "superseded")
DECISIONS = ("APPROVE", "REJECT", "REQUEST_REANALYSIS")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    requirements: Mapped[list["RequirementRecord"]] = relationship(back_populates="project", passive_deletes=True)


class RequirementRecord(Base):
    __tablename__ = "requirements"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    project: Mapped[Project] = relationship(back_populates="requirements")


class AgentRun(Base):
    """One LangGraph workflow execution; ``id`` is the graph thread id (run id)."""

    __tablename__ = "agent_runs"
    __table_args__ = (CheckConstraint(_in("status", AGENT_RUN_STATUSES), name="status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    requirement_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("requirements.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running", index=True)
    outcome: Mapped[str | None] = mapped_column(String(32), comment="Final report status once completed")
    confidence: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[str | None] = mapped_column(Text)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    nodes_visited: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    requirement_analysis: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    root_cause_analysis: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    final_report: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    started_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TestCaseRecord(Base):
    __tablename__ = "test_cases"
    __test__ = False  # not a pytest test class
    __table_args__ = (UniqueConstraint("agent_run_id", "case_key"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    requirement_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("requirements.id", ondelete="CASCADE"), index=True)
    case_key: Mapped[str] = mapped_column(String(32), nullable=False, comment="e.g. TC-001")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    preconditions: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    test_data: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list, nullable=False)
    steps: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    expected_result: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False)
    test_type: Mapped[str] = mapped_column(String(16), nullable=False)
    automation_type: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class TestRunRecord(Base):
    """The test execution batch of one agent run."""

    __tablename__ = "test_runs"
    __test__ = False  # not a pytest test class

    id: Mapped[uuid.UUID] = _uuid_pk()
    agent_run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    errored: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class TestResultRecord(Base):
    __tablename__ = "test_results"
    __test__ = False  # not a pytest test class
    __table_args__ = (
        UniqueConstraint("test_run_id", "test_case_id"),
        CheckConstraint(_in("status", TEST_RESULT_STATUSES), name="status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    test_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("test_runs.id", ondelete="CASCADE"), index=True)
    test_case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("test_cases.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class FailureRecord(Base):
    __tablename__ = "failures"

    id: Mapped[uuid.UUID] = _uuid_pk()
    test_result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("test_results.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    category: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text)
    signal: Mapped[str | None] = mapped_column(Text)
    expected_result: Mapped[str | None] = mapped_column(Text)
    actual_result: Mapped[str | None] = mapped_column(Text)
    observed_facts: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class BugReportRecord(Base):
    __tablename__ = "bug_reports"
    __table_args__ = (
        UniqueConstraint("agent_run_id", "report_key", "revision"),
        CheckConstraint(_in("severity", SEVERITIES), name="severity"),
        CheckConstraint(_in("priority", PRIORITIES), name="priority"),
        CheckConstraint(_in("status", BUG_STATUSES), name="status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    report_key: Mapped[str] = mapped_column(String(32), nullable=False, comment="e.g. BR-1")
    revision: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    report_type: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[str] = mapped_column(String(8), nullable=False)
    affected_component: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="draft", nullable=False)
    related_test_case_ids: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False, comment="Full BugReport")
    created_at: Mapped[datetime] = _created_at()


class HumanReviewRecord(Base):
    __tablename__ = "human_reviews"
    __table_args__ = (
        CheckConstraint(f"human_decision IS NULL OR {_in('human_decision', DECISIONS)}", name="human_decision"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    review_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, comment="HumanReview.review_id")
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    ai_recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    proposed_action: Mapped[str] = mapped_column(Text, nullable=False)
    actions: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    sensitive_actions: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    human_decision: Mapped[str | None] = mapped_column(String(32))
    reviewer: Mapped[str | None] = mapped_column(String(200))
    reviewer_comment: Mapped[str | None] = mapped_column(Text)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
