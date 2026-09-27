"""Test execution results and the failures extracted from them."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ExecutionStatus = Literal["passed", "failed", "error", "skipped"]
FailureCategory = Literal[
    "assertion_mismatch",
    "timeout",
    "server_error",
    "element_not_found",
    "flaky",
    "environment",
    "unknown",
]


class ExecutionResult(BaseModel):
    test_case_id: str
    status: ExecutionStatus
    duration_ms: int = Field(default=0, ge=0)
    message: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ObservedFact(BaseModel):
    """Something the test execution actually recorded, never an inference."""

    id: str = Field(description="Stable reference such as 'TC-007:F2'.")
    source: str = Field(description="Where it came from, e.g. 'api.status_code', 'ui.console'.")
    detail: str


class FailureDetail(BaseModel):
    test_case_id: str
    status: Literal["failed", "error"]
    message: str | None = None
    category: FailureCategory | None = None
    signal: str | None = Field(default=None, description="The single most telling fact.")
    test_title: str | None = None
    automation_type: str | None = None
    expected_result: str | None = None
    actual_result: str | None = Field(
        default=None, description="What was observed, summarized from the facts."
    )
    observed_facts: list[ObservedFact] = Field(default_factory=list)
