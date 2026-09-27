"""API test request/result contract used by the ``execute_api_test`` tool."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
ApiTestStatus = Literal["passed", "failed", "error"]
JsonType = Literal["string", "number", "integer", "boolean", "array", "object", "null"]

ValidationRuleType = Literal[
    "json_path_equals",
    "json_path_exists",
    "json_path_not_exists",
    "json_path_type",
    "json_path_matches",
    "body_contains",
    "header_equals",
    "header_exists",
    "max_response_time_ms",
]

_NEEDS_TARGET = {
    "json_path_equals",
    "json_path_exists",
    "json_path_not_exists",
    "json_path_type",
    "json_path_matches",
    "header_equals",
    "header_exists",
}
_NEEDS_EXPECTED = {
    "json_path_equals",
    "json_path_type",
    "json_path_matches",
    "body_contains",
    "header_equals",
    "max_response_time_ms",
}


class ValidationRule(BaseModel):
    """One assertion on the response, beyond the expected status code.

    ``target`` is a JSON path (``data.items[0].id``, optional ``$.`` prefix)
    for ``json_path_*`` rules, or a header name for ``header_*`` rules.
    """

    type: ValidationRuleType
    target: str | None = None
    expected: Any = None

    @model_validator(mode="after")
    def _check_arguments(self) -> "ValidationRule":
        if self.type in _NEEDS_TARGET and not self.target:
            raise ValueError(f"rule '{self.type}' requires a target")
        if self.type in _NEEDS_EXPECTED and self.expected is None:
            raise ValueError(f"rule '{self.type}' requires an expected value")
        if self.type == "json_path_type" and self.expected not in JsonType.__args__:
            raise ValueError(f"json_path_type expected must be one of {list(JsonType.__args__)}")
        if self.type == "max_response_time_ms" and (
            isinstance(self.expected, bool) or not isinstance(self.expected, int | float)
        ):
            raise ValueError("max_response_time_ms expected must be a number")
        return self


class ApiTestRequest(BaseModel):
    test_case_id: str = Field(description="Id of the test case this request verifies.")
    method: HttpMethod
    url: str = Field(
        description="Path relative to the configured test base URL (e.g. '/api/v2/login'), "
        "or an absolute URL on an allowed test host."
    )
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str | int | float | bool | list[str | int | float | bool]] = Field(
        default_factory=dict
    )
    body: Any = Field(default=None, description="JSON-serializable request body, if any.")
    expected_status: int = Field(ge=100, le=599)
    validation_rules: list[ValidationRule] = Field(default_factory=list)
    timeout_s: float | None = Field(
        default=None, gt=0, le=120, description="Overrides the configured timeout."
    )

    @field_validator("method", mode="before")
    @classmethod
    def _upper_method(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class ApiTestResult(BaseModel):
    test_case_id: str
    status: ApiTestStatus = Field(
        description="passed: status and rules matched; failed: a response was received but "
        "did not match; error: no usable response (config, safety, network, timeout)."
    )
    status_code: int | None = None
    response_time: float | None = Field(default=None, description="Milliseconds.")
    response_body: Any = None
    validation_errors: list[str] = Field(default_factory=list)
    error_message: str | None = None
    method: HttpMethod | None = None
    url: str | None = Field(default=None, description="Resolved request URL, without query.")
