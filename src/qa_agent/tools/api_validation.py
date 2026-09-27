"""Response validation for API tests: status code plus ``ValidationRule``s."""

import json
import re
from typing import Any

import httpx

from qa_agent.domain.api_test import ValidationRule

_MISSING = object()
_SEGMENT = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_VALID_PATH = re.compile(r"(?:[^.\[\]]+|\[\d+\])(?:\.[^.\[\]]+|\[\d+\])*")


class JsonPathError(ValueError):
    pass


def resolve_json_path(document: Any, path: str) -> Any:
    """Resolve ``a.b[0].c`` (optional ``$``/``$.`` prefix). Returns ``_MISSING`` if absent."""
    expression = path.strip()
    if expression.startswith("$"):
        expression = expression[1:].lstrip(".")
    if not expression:
        return document

    if not _VALID_PATH.fullmatch(expression):
        raise JsonPathError(f"Malformed JSON path: {path!r}")

    current = document
    for key, index in _SEGMENT.findall(expression):
        if key:
            if not isinstance(current, dict) or key not in current:
                return _MISSING
            current = current[key]
        else:
            position = int(index)
            if not isinstance(current, list) or position >= len(current):
                return _MISSING
            current = current[position]
    return current


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _json_equal(actual: Any, expected: Any) -> bool:
    """Equality that keeps JSON types apart (``1 != True``, ``"1" != 1``) but lets
    integers and floats compare numerically (``1 == 1.0``)."""
    numeric = {"integer", "number"}
    actual_type, expected_type = json_type(actual), json_type(expected)
    if actual_type in numeric and expected_type in numeric:
        return actual == expected
    return actual_type == expected_type and actual == expected


def validate_response(
    *,
    expected_status: int,
    response: httpx.Response,
    body: Any,
    body_text: str,
    body_is_json: bool,
    truncated: bool,
    response_time_ms: float,
    rules: list[ValidationRule],
) -> list[str]:
    """Return human-readable validation errors; empty means the response matched."""
    errors: list[str] = []
    if response.status_code != expected_status:
        errors.append(f"Expected status {expected_status}, got {response.status_code}")

    for rule in rules:
        error = _check_rule(rule, response, body, body_text, body_is_json, truncated, response_time_ms)
        if error:
            errors.append(error)
    return errors


def _check_rule(
    rule: ValidationRule,
    response: httpx.Response,
    body: Any,
    body_text: str,
    body_is_json: bool,
    truncated: bool,
    response_time_ms: float,
) -> str | None:
    kind, target, expected = rule.type, rule.target, rule.expected

    if kind == "max_response_time_ms":
        if response_time_ms > expected:
            return f"Response time {response_time_ms:.1f} ms exceeds {expected} ms"
        return None

    if kind == "body_contains":
        if str(expected) not in body_text:
            suffix = " (body was truncated)" if truncated else ""
            return f"Body does not contain {str(expected)!r}{suffix}"
        return None

    if kind == "header_exists":
        return None if target in response.headers else f"Header {target!r} is missing"

    if kind == "header_equals":
        actual = response.headers.get(target)
        if actual is None:
            return f"Header {target!r} is missing"
        return None if actual == str(expected) else f"Header {target!r}: expected {expected!r}, got {actual!r}"

    # json_path_* rules
    if truncated:
        return f"{kind} {target!r}: response body was truncated; cannot evaluate"
    if not body_is_json:
        return f"{kind} {target!r}: response body is not JSON"
    try:
        actual = resolve_json_path(body, target)
    except JsonPathError as exc:
        return str(exc)

    if kind == "json_path_exists":
        return None if actual is not _MISSING else f"JSON path {target!r} not found"
    if kind == "json_path_not_exists":
        return None if actual is _MISSING else f"JSON path {target!r} should not exist"
    if actual is _MISSING:
        return f"JSON path {target!r} not found"
    if kind == "json_path_equals":
        return None if _json_equal(actual, expected) else (
            f"JSON path {target!r}: expected {json.dumps(expected)}, got {json.dumps(actual)}"
        )
    if kind == "json_path_type":
        actual_type = json_type(actual)
        matches = actual_type == expected or (expected == "number" and actual_type == "integer")
        return None if matches else f"JSON path {target!r}: expected type {expected}, got {actual_type}"
    if kind == "json_path_matches":
        if not isinstance(actual, str):
            return f"JSON path {target!r}: expected a string to match, got {json_type(actual)}"
        try:
            matched = re.search(str(expected), actual)
        except re.error as exc:
            return f"Invalid regular expression {expected!r}: {exc}"
        return None if matched else f"JSON path {target!r}: {actual!r} does not match {expected!r}"
    return f"Unsupported rule type {kind!r}"  # pragma: no cover - guarded by the model
