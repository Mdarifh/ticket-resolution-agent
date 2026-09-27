"""Failure Analyzer core: turn a failed execution into observed facts + a category.

Everything here is read from what the executors recorded (API result, UI
steps and diagnostics, logs). Nothing is inferred, so these facts are the
trustworthy "observed evidence" the Root Cause Analyzer must cite.
"""

import json
import re
from typing import Any

from qa_agent.domain import ExecutionResult, FailureDetail, ObservedFact, TestCase
from qa_agent.domain.execution import FailureCategory

MAX_DETAIL = 500
_TIMEOUT = re.compile(r"timed? ?out|timeout", re.IGNORECASE)
_NETWORK = re.compile(r"ConnectError|ConnectTimeout|connection refused|RemoteProtocolError|NetworkError|name resolution", re.IGNORECASE)
_ELEMENT_WAIT = re.compile(r"waiting for (locator|selector)|element\(s\) not found|not found", re.IGNORECASE)
_HTTP_STATUS = re.compile(r"HTTP (\d{3})")


def analyze_failure(result: ExecutionResult, test_case: TestCase | None) -> FailureDetail:
    evidence = result.evidence or {}
    facts = _Facts(result.test_case_id)

    if _is_api(evidence):
        category, actual = _api_facts(evidence, facts)
    elif _is_ui(evidence):
        category, actual = _ui_facts(evidence, facts)
    else:
        category, actual = _generic_facts(result, facts)

    for line in evidence.get("logs") or []:
        facts.add("logs", str(line))
    facts.add(
        "execution",
        f"Executed as {evidence.get('automation_type', 'unknown')} test; "
        f"status '{result.status}'; duration {result.duration_ms} ms",
    )

    return FailureDetail(
        test_case_id=result.test_case_id,
        status=result.status,
        message=result.message,
        category=category,
        signal=facts.items[0].detail if facts.items else result.message,
        test_title=test_case.title if test_case else None,
        automation_type=evidence.get("automation_type"),
        expected_result=test_case.expected_result if test_case else None,
        actual_result=actual,
        observed_facts=facts.items,
    )


class _Facts:
    def __init__(self, test_case_id: str) -> None:
        self.test_case_id = test_case_id
        self.items: list[ObservedFact] = []

    def add(self, source: str, detail: str) -> None:
        detail = " ".join(str(detail).split())
        if len(detail) > MAX_DETAIL:
            detail = detail[:MAX_DETAIL] + " [truncated]"
        self.items.append(
            ObservedFact(id=f"{self.test_case_id}:F{len(self.items) + 1}", source=source, detail=detail)
        )


def _is_api(evidence: dict[str, Any]) -> bool:
    return evidence.get("automation_type") == "api" and ("status_code" in evidence or "error_message" in evidence)


def _is_ui(evidence: dict[str, Any]) -> bool:
    return evidence.get("automation_type") == "ui" and "executed_steps" in evidence


def _api_facts(evidence: dict[str, Any], facts: _Facts) -> tuple[FailureCategory, str]:
    request = f"{evidence.get('method') or '?'} {evidence.get('url') or '?'}"
    status_code = evidence.get("status_code")
    error_message = evidence.get("error_message")
    validation_errors = evidence.get("validation_errors") or []

    if status_code is None:
        facts.add("api.error", f"{request} produced no response: {error_message}")
        if error_message and _TIMEOUT.search(error_message) and not _NETWORK.search(error_message):
            return "timeout", f"No response: {error_message}"
        if error_message and _NETWORK.search(error_message):
            return "environment", f"No response: {error_message}"
        return "unknown", f"No response: {error_message}"

    facts.add("api.status_code", f"{request} returned HTTP {status_code}")
    for error in validation_errors:
        facts.add("api.validation", error)
    body = evidence.get("response_body")
    if body is not None:
        rendered = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        facts.add("api.response_body", f"Response body: {rendered}")
    if evidence.get("response_time") is not None:
        facts.add("api.response_time", f"Response time {evidence['response_time']} ms")

    actual = f"HTTP {status_code}" + (f"; {'; '.join(validation_errors)}" if validation_errors else "")
    if status_code >= 500:
        return "server_error", actual
    return "assertion_mismatch", actual


def _ui_facts(evidence: dict[str, Any], facts: _Facts) -> tuple[FailureCategory, str]:
    failed = evidence.get("failed_step") or {}
    diagnostics = evidence.get("diagnostics") or {}
    error = evidence.get("error") or ""
    executed = evidence.get("executed_steps") or []

    if failed:
        step = f"Step {failed.get('index')} {failed.get('action')} {failed.get('target')!r}"
        if failed.get("value") is not None:
            step += f" with value {failed.get('value')!r}"
        facts.add("ui.failed_step", f"{step} failed: {failed.get('error')}")
        facts.add("ui.progress", f"{len(executed) - 1} of the executed steps passed before the failure")
    elif error:
        facts.add("ui.error", error)

    if diagnostics.get("page_url"):
        facts.add("ui.page", f"Page at failure: {diagnostics['page_url']} (title {diagnostics.get('page_title')!r})")
    matches = diagnostics.get("matching_elements")
    if matches is not None:
        facts.add("ui.selector", f"Selector matched {matches} element(s)")
    if diagnostics.get("element_text") is not None:
        facts.add("ui.element_text", f"Element text: {diagnostics['element_text']!r}")
    for line in diagnostics.get("console_errors") or []:
        facts.add("ui.console", f"Browser console error: {line}")
    for line in diagnostics.get("failed_requests") or []:
        facts.add("ui.network", f"Failed request: {line}")
    for line in diagnostics.get("blocked_requests") or []:
        facts.add("ui.blocked", f"Blocked non-test request: {line}")

    step_error = failed.get("error") or error
    actual = step_error or "UI step failed"
    if evidence.get("status") == "error" and not failed:
        return ("environment" if "browser" in error.lower() else "unknown"), actual
    status_match = _HTTP_STATUS.search(step_error)
    if failed.get("action") == "navigate" and status_match and int(status_match.group(1)) >= 500:
        return "server_error", actual
    if matches == 0 or (matches is None and _ELEMENT_WAIT.search(step_error)):
        return "element_not_found", actual
    if _TIMEOUT.search(step_error) and failed.get("action") not in ("verify_text", "verify_visible"):
        return "timeout", actual
    return "assertion_mismatch", actual


def _generic_facts(result: ExecutionResult, facts: _Facts) -> tuple[FailureCategory, str]:
    facts.add("result", f"Test reported status '{result.status}': {result.message or 'no message'}")
    message = result.message or ""
    if _TIMEOUT.search(message):
        return "timeout", message
    if result.status == "failed":
        return "assertion_mismatch", message
    return "unknown", message
