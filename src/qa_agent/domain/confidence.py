"""Confidence scoring used to decide whether a human must review the run."""

from pydantic import BaseModel, Field


class ConfidenceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
    reasons: list[str] = Field(default_factory=list)
