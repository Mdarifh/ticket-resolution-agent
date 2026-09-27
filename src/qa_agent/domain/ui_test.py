"""UI test request/result contract used by the Playwright tool.

Failure analysis works from structured diagnostics (step, selector, expected
vs actual text, page URL, console errors, failed or blocked requests). A
screenshot, when enabled, is only a supporting artifact path.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

UiAction = Literal["navigate", "click", "fill", "select", "verify_text", "verify_visible"]
UiTestStatus = Literal["passed", "failed", "error"]

_NEEDS_VALUE = {"fill", "select", "verify_text"}


class UiStep(BaseModel):
    """One browser action.

    ``target`` is a path (or allowed-host URL) for ``navigate`` and a Playwright
    selector otherwise, e.g. ``data-testid=submit``, ``#email``,
    ``role=button[name="Send reset link"]`` or ``text=Forgot password?``.
    ``value`` is the text to type, the option to select, or the expected text.
    """

    action: UiAction
    target: str = Field(min_length=1)
    value: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _check_value(self) -> "UiStep":
        if self.action in _NEEDS_VALUE and self.value is None:
            raise ValueError(f"step '{self.action}' requires a value")
        return self


class UiTestRequest(BaseModel):
    test_case_id: str
    steps: list[UiStep] = Field(min_length=1)


class UiStepResult(BaseModel):
    index: int = Field(ge=1, description="1-based position in the test's steps.")
    action: UiAction
    target: str
    value: str | None = None
    status: Literal["passed", "failed"]
    duration_ms: float = Field(ge=0)
    error: str | None = None


class UiDiagnostics(BaseModel):
    """Page state captured when a step fails."""

    page_url: str | None = None
    page_title: str | None = None
    matching_elements: int | None = Field(
        default=None, description="How many elements the failed step's selector matched."
    )
    element_text: str | None = Field(
        default=None, description="Text of the first matching element, if any (truncated)."
    )
    console_errors: list[str] = Field(default_factory=list)
    failed_requests: list[str] = Field(default_factory=list)
    blocked_requests: list[str] = Field(
        default_factory=list, description="Requests to non-test hosts that were aborted."
    )
    screenshot_path: str | None = Field(
        default=None, description="Supporting artifact only; not used for analysis."
    )


class UiTestResult(BaseModel):
    test_case_id: str
    status: UiTestStatus = Field(
        description="passed: every step succeeded; failed: a step failed on the page; "
        "error: the test could not run (configuration, safety, browser)."
    )
    executed_steps: list[UiStepResult] = Field(default_factory=list)
    failed_step: UiStepResult | None = None
    error: str | None = None
    execution_time: float = Field(default=0.0, ge=0, description="Milliseconds.")
    diagnostics: UiDiagnostics | None = None
