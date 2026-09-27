"""Structured bug report.

Factual sections (environment, preconditions, reproduction steps, expected and
actual results, evidence) are assembled from recorded test data, never written
by the LLM. The probable root cause keeps its likelihood, and everything the
analysis could not establish is listed under ``uncertainties``.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from qa_agent.domain.analysis import EvidenceItem, ProbableCause, Severity

__all__ = ["BugEnvironment", "BugPriority", "BugReport", "ReportType", "Severity"]

BugPriority = Literal["p0", "p1", "p2", "p3"]
ReportType = Literal[
    "product_bug", "test_issue", "requirement_issue", "environment_issue", "needs_investigation"
]


class BugEnvironment(BaseModel):
    app_env: str = Field(description="Environment the agent ran in (APP_ENV).")
    run_id: str | None = None
    targets: list[str] = Field(
        default_factory=list, description="Origins of the system under test seen in the evidence."
    )
    automation_types: list[str] = Field(default_factory=list)
    recorded_at: datetime
    unknowns: list[str] = Field(default_factory=list)


class BugReport(BaseModel):
    id: str
    title: str
    summary: str
    report_type: ReportType
    environment: BugEnvironment
    preconditions: list[str] = Field(default_factory=list)
    reproduction_steps: list[str] = Field(default_factory=list)
    reproduction_source: Literal["executed_steps", "root_cause_analysis", "test_case"]
    expected_result: str
    actual_result: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    probable_root_cause: ProbableCause
    severity: Severity
    severity_rationale: str
    priority: BugPriority
    priority_rationale: str
    affected_component: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the underlying root cause analysis.")
    related_test_case_ids: list[str] = Field(default_factory=list)
    related_bugs: list[str] = Field(
        default_factory=list, description="Knowledge base paths of similar past bugs (possible duplicates)."
    )
    uncertainties: list[str] = Field(default_factory=list)
    generated_by: Literal["llm", "fallback"] = Field(
        description="Who wrote title/summary/severity/priority; facts are always deterministic."
    )
    revision: int = Field(default=0, ge=0)
    reviewer_feedback: list[str] = Field(default_factory=list)
