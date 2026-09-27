"""Realistic failure scenarios shared by analyzer and bug report tests.

Execution evidence is produced by the real API/UI tools (mocked HTTP, in-memory
browser), so analyzers see exactly what they would in a run.
"""

import httpx

from qa_agent.domain import ExecutionResult, PlannedTest, TestCase
from qa_agent.domain.api_test import ApiTestRequest
from qa_agent.domain.ui_test import UiStep, UiTestRequest
from qa_agent.graph.nodes.test_executor import from_api_result, from_ui_result
from qa_agent.tools.api_test_tool import ApiTestConfig, ApiTestExecutor
from tests.conftest import PASSING_UI_TEXTS, fake_ui_runner, load_fixture
from tests.fakes import FakeUiBrowser

def case_for(tc_id: str, **overrides) -> TestCase:
    raw = load_fixture("password_reset", "test_suite.json")["test_cases"]
    by_number = {f"TC-{i:03d}": tc for i, tc in enumerate(raw, start=1)}
    return TestCase.model_validate({**by_number[tc_id], "test_case_id": tc_id, **overrides})


def api_run(tc_id: str, respond, request: dict) -> ExecutionResult:
    client = httpx.Client(transport=httpx.MockTransport(respond))
    executor = ApiTestExecutor(ApiTestConfig(base_url="https://sut.test.local", timeout_s=2), client)
    result = executor.execute(ApiTestRequest(test_case_id=tc_id, **request))
    entry = PlannedTest(test_case_id=tc_id, automation_type="api", execution_order=1, rationale="r")
    return from_api_result(entry, result)


def ui_run(tc_id: str, steps: list[dict], **browser) -> ExecutionResult:
    runner = fake_ui_runner(FakeUiBrowser(texts=dict(PASSING_UI_TEXTS), **browser))
    result = runner.run(UiTestRequest(test_case_id=tc_id, steps=[UiStep.model_validate(s) for s in steps]))
    entry = PlannedTest(test_case_id=tc_id, automation_type="ui", execution_order=1, rationale="r")
    return from_ui_result(entry, result)


def api_500() -> tuple[ExecutionResult, TestCase]:
    result = api_run(
        "TC-001",
        lambda r: httpx.Response(500, json={"error": "internal_error", "trace_id": "abc123"}),
        {"method": "POST", "url": "/api/v2/password-reset", "body": {"email": "registered.user@example.com"}, "expected_status": 202},
    )
    return result, case_for("TC-001")


def validation_failure() -> tuple[ExecutionResult, TestCase]:
    result = api_run(
        "TC-007",
        lambda r: httpx.Response(400, json={"error": "token_invalid"}),
        {
            "method": "POST",
            "url": "/api/v2/password-reset/confirm",
            "body": {"token": "used-token-456", "new_password": "An0ther-Passw0rd!"},
            "expected_status": 400,
            "validation_rules": [{"type": "json_path_equals", "target": "error", "expected": "token_used"}],
        },
    )
    return result, case_for("TC-007")


def ui_element_missing() -> tuple[ExecutionResult, TestCase]:
    result = ui_run(
        "TC-005",
        [
            {"action": "navigate", "target": "/forgot-password.html"},
            {"action": "fill", "target": "data-testid=email-input", "value": "not-an-email"},
            {"action": "click", "target": "data-testid=send-reset-link"},
            {"action": "verify_visible", "target": "data-testid=email-error"},
        ],
        missing={"data-testid=send-reset-link"},
        console_errors=["Uncaught ReferenceError: submitForm is not defined"],
    )
    return result, case_for("TC-005")


def api_timeout() -> tuple[ExecutionResult, TestCase]:
    def hang(request):
        raise httpx.ReadTimeout("timed out", request=request)

    result = api_run(
        "TC-004",
        hang,
        {"method": "POST", "url": "/api/v2/password-reset", "body": {"email": "nobody@example.com"}, "expected_status": 202},
    )
    return result, case_for("TC-004")


def requirement_mismatch() -> tuple[ExecutionResult, TestCase]:
    """The test expects 404 for unknown emails; the requirement demands a generic 202."""
    test_case = case_for(
        "TC-004",
        title="Unregistered email is reported as not found",
        expected_result="Response is 404 Not Found for an unregistered email",
    )
    result = api_run(
        "TC-004",
        lambda r: httpx.Response(202, json={"message": "If the email is registered, a reset link has been sent."}),
        {"method": "POST", "url": "/api/v2/password-reset", "body": {"email": "nobody@example.com"}, "expected_status": 404},
    )
    return result, test_case


# --- root cause drafts (LLM output) ----------------------------------------


def make_finding(test_ids: list[str], **overrides) -> dict:
    finding = {
        "test_case_ids": test_ids,
        "summary": "summary",
        "observed_behavior": "observed",
        "expected_behavior": "expected",
        "suspected_origin": "product_defect",
        "probable_root_cause": {
            "description": "Possibly X",
            "reasoning": "because",
            "likelihood": "likely",
            "supporting_evidence": [f"{test_ids[0]}:F1"],
        },
        "alternative_causes": [],
        "evidence": [{"reference": f"{test_ids[0]}:F1", "interpretation": "what it shows"}],
        "unknowns": ["Server logs are not available"],
        "affected_component": "auth",
        "severity_suggestion": "high",
        "severity_rationale": "account security",
        "reproduction_steps": ["step"],
        "recommended_next_investigation": ["check logs"],
        "confidence": 0.8,
    }
    return {**finding, **overrides}


def make_draft(*findings: dict, summary: str = "overall") -> dict:
    return {"summary": summary, "findings": list(findings)}
