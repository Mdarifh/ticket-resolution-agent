"""Run-level bookkeeping: node errors and execution metadata."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class WorkflowError(BaseModel):
    node: str
    message: str
    error_type: str
    occurred_at: datetime = Field(default_factory=_utcnow)
    visit: int = Field(
        default=0, ge=0, description="0-based visit of the node that failed; a later visit supersedes it."
    )


class ExecutionMetadata(BaseModel):
    run_id: str | None = None
    status: Literal["running", "completed"] = "running"
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    nodes_visited: list[str] = Field(default_factory=list)
