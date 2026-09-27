"""Final QA report emitted at the end of every run."""

from typing import Literal

from pydantic import BaseModel, Field

from qa_agent.domain.bug_report import BugReport
from qa_agent.domain.confidence import ConfidenceScore
from qa_agent.domain.review import HumanDecision, HumanReview

ReportStatus = Literal["passed", "failed", "rejected", "error", "inconclusive"]


class FinalReport(BaseModel):
    """``total_tests`` counts executed tests; skipped (e.g. manual) tests are separate.

    ``failed`` counts tests whose assertions failed; ``errored`` counts tests that
    could not complete (tool, configuration or connection errors).
    """

    status: ReportStatus
    summary: str
    total_tests: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errored: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    bug_reports: list[BugReport] = Field(default_factory=list)
    confidence_score: ConfidenceScore | None = None
    human_decision: HumanDecision | None = None
    human_review: HumanReview | None = Field(
        default=None, description="The decided review that let the run finish, if any."
    )
    error_count: int = Field(
        default=0, ge=0, description="Workflow errors not recovered by a later attempt of the same step."
    )
    markdown: str = ""
