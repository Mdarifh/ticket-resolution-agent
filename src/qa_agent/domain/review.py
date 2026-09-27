"""Human-in-the-loop review: the request shown to a reviewer and their decision."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from qa_agent.domain.analysis import EvidenceItem

HumanDecision = Literal["APPROVE", "REJECT", "REQUEST_REANALYSIS"]
ProposedAction = Literal["report_bugs", "report_high_severity_bugs", "reclassify_failure_as_test_issue"]


class HumanReviewDecision(BaseModel):
    """Payload a reviewer sends to resume a paused run."""

    decision: HumanDecision
    comment: str | None = Field(
        default=None, description="Required for REQUEST_REANALYSIS: what the analysis should reconsider."
    )
    reviewer: str | None = None

    @field_validator("decision", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _reanalysis_needs_guidance(self) -> "HumanReviewDecision":
        if self.decision == "REQUEST_REANALYSIS" and not (self.comment and self.comment.strip()):
            raise ValueError("REQUEST_REANALYSIS requires a comment telling the analysis what to reconsider")
        return self


class HumanReview(BaseModel):
    review_id: str
    reason: str = Field(description="Why the run paused (all triggers, joined).")
    reasons: list[str] = Field(default_factory=list)
    ai_recommendation: str = Field(description="What the agent proposes, in plain words.")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_action: str = Field(description="The actions awaiting approval, described.")
    actions: list[ProposedAction] = Field(default_factory=list)
    sensitive_actions: list[ProposedAction] = Field(
        default_factory=list, description="Proposed actions that the configuration says need approval."
    )
    human_decision: HumanDecision | None = None
    reviewer_comment: str | None = None
    reviewer: str | None = None
    requested_at: datetime
    decided_at: datetime | None = None
