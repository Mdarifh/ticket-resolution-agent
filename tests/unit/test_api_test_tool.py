"""execute_api_test with mocked HTTP (httpx.MockTransport); no network access."""

import json

import httpx
import pytest
from pydantic import ValidationError

from qa_agent.config import Settings
from qa_agent.domain.api_test import ApiTestRequest, ValidationRule
from qa_agent.tools.api_test_tool import (
    ApiTestConfig,
    ApiTestExecutor,
    create_execute_api_test_tool,
    execute_api_test,
)
from qa_agent.tools.api_validation import JsonPathError, resolve_json_path

BASE_URL = "https://api.test.local/v2"
RESULT_FIELDS = {
    "test_case_id",
    "status",
    "status_code",
    "response_time",
    "response_body",
    "validation_errors",
    "error_message",
}


class Recorder:
    """Mock transport that records requests and replies via ``respond``."""

    def __init__(self, respond=None):
        self.requests: list[httpx.Request] = []
        self.respond = respond or (lambda request: httpx.Response(200, json={"ok": True}))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)


def _executor(respond=None, **config) -> tuple[ApiTestExecutor, Recorder]:
    recorder = Recorder(respond)
    settings = {"base_url": BASE_URL, **config}
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    return ApiTestExecutor(ApiTestConfig(**settings), client), recorder


def _request(**overrides) -> ApiTestRequest:
    return ApiTestRequest(
        **{"test_case_id": "TC-001", "method": "GET", "url": "/health", "expected_status": 200, **overrides}
    )


def _json(status: int, payload, headers=None):
    return lambda request: httpx.Response(status, json=payload, headers=headers)


# --- request construction ----------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
def test_supported_methods_are_sent(method):
    executor, recorder = _executor()

    result = executor.execute(_request(method=method, url="/items/7"))

    [sent] = recorder.requests
    assert sent.method == method
    assert str(sent.url) == f"{BASE_URL}/items/7"
    assert result.status == "passed"
    assert result.method == method


def test_method_is_case_insensitive_and_unknown_methods_are_rejected():
    assert _request(method="patch").method == "PATCH"
    with pytest.raises(ValidationError):
        _request(method="TRACE")


def test_headers_query_params_and_json_body_are_sent():
    executor, recorder = _executor()

    executor.execute(
        _request(
            method="POST",
            url="/password-reset",
            headers={"X-Trace": "abc", "Host": "evil.example.com"},
            query_params={"lang": "en", "ids": [1, 2]},
            body={"email": "registered.user@example.com"},
        )
    )

    [sent] = recorder.requests
    assert sent.headers["x-trace"] == "abc"
    assert sent.headers["host"] == "api.test.local"  # user-supplied Host is stripped
    assert sent.url.params.get_list("ids") == ["1", "2"]
    assert sent.url.params["lang"] == "en"
    assert json.loads(sent.content) == {"email": "registered.user@example.com"}


def test_request_without_body_sends_no_content():
    executor, recorder = _executor()

    executor.execute(_request(method="DELETE", url="/items/7"))

    assert recorder.requests[0].content == b""


def test_auth_token_is_sent_but_never_returned():
    executor, recorder = _executor(auth_token="s3cret-token")

    result = executor.execute(_request())

    assert recorder.requests[0].headers["authorization"] == "Bearer s3cret-token"
    assert "s3cret-token" not in result.model_dump_json()
    assert "s3cret-token" not in repr(executor.config)


def test_empty_auth_token_sends_no_authorization_header():
    # API_TEST_AUTH_TOKEN= in .env yields an empty secret; "Bearer " is an illegal header.
    executor, recorder = _executor(auth_token="")

    result = executor.execute(_request())

    assert "authorization" not in recorder.requests[0].headers
    assert result.status == "passed"


def test_explicit_authorization_header_wins_over_configured_token():
    executor, recorder = _executor(auth_token="configured")

    executor.execute(_request(headers={"Authorization": "Bearer per-test"}))

    assert recorder.requests[0].headers["authorization"] == "Bearer per-test"


# --- result structure --------------------------------------------------------


def test_passing_result_is_fully_structured():
    executor, _ = _executor(_json(202, {"message": "sent"}))

    result = executor.execute(_request(method="POST", url="/password-reset", expected_status=202))

    assert RESULT_FIELDS <= set(result.model_dump())
    assert result.status == "passed"
    assert result.status_code == 202
    assert result.response_time is not None and result.response_time >= 0
    assert result.response_body == {"message": "sent"}
    assert result.validation_errors == []
    assert result.error_message is None
    assert result.url == f"{BASE_URL}/password-reset"


def test_unexpected_status_fails_with_validation_error():
    executor, _ = _executor(_json(404, {"error": "not_found"}))

    result = executor.execute(_request(expected_status=200))

    assert result.status == "failed"
    assert result.status_code == 404
    assert result.validation_errors == ["Expected status 200, got 404"]
    assert result.response_body == {"error": "not_found"}


def test_non_json_body_is_returned_as_text():
    executor, _ = _executor(lambda r: httpx.Response(200, text="<h1>ok</h1>", headers={"content-type": "text/html"}))

    assert executor.execute(_request()).response_body == "<h1>ok</h1>"


def test_empty_body_is_none():
    executor, _ = _executor(lambda r: httpx.Response(204))

    result = executor.execute(_request(method="DELETE", expected_status=204))

    assert result.status == "passed"
    assert result.response_body is None


def test_malformed_json_body_is_kept_as_text():
    executor, _ = _executor(
        lambda r: httpx.Response(200, content=b"{not json", headers={"content-type": "application/json"})
    )

    result = executor.execute(
        _request(validation_rules=[ValidationRule(type="json_path_exists", target="id")])
    )

    assert result.response_body == "{not json"
    assert result.status == "failed"
    assert result.validation_errors == ["json_path_exists 'id': response body is not JSON"]


def test_oversized_response_is_truncated():
    executor, _ = _executor(lambda r: httpx.Response(200, text="x" * 5000), max_response_bytes=100)

    result = executor.execute(
        _request(validation_rules=[ValidationRule(type="json_path_exists", target="id")])
    )

    assert result.response_body == "x" * 100 + " [truncated]"
    assert "response body was truncated" in result.validation_errors[0]


# --- validation rules --------------------------------------------------------

BODY = {
    "message": "sent",
    "data": {"items": [{"id": 7, "price": 9.5, "active": True, "tags": []}], "next": None},
}
HEADERS = {"X-Request-Id": "req-1", "Content-Type": "application/json"}


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "json_path_equals", "target": "message", "expected": "sent"},
        {"type": "json_path_equals", "target": "$.data.items[0].id", "expected": 7},
        {"type": "json_path_equals", "target": "data.items[0].price", "expected": 9.5},
        {"type": "json_path_equals", "target": "data.items[0].tags", "expected": []},
        {"type": "json_path_exists", "target": "data.next"},
        {"type": "json_path_not_exists", "target": "data.missing"},
        {"type": "json_path_type", "target": "data.items", "expected": "array"},
        {"type": "json_path_type", "target": "data.items[0].id", "expected": "number"},
        {"type": "json_path_type", "target": "data.next", "expected": "null"},
        {"type": "json_path_matches", "target": "message", "expected": "^se"},
        {"type": "body_contains", "expected": '"sent"'},
        {"type": "header_exists", "target": "x-request-id"},
        {"type": "header_equals", "target": "X-Request-Id", "expected": "req-1"},
        {"type": "max_response_time_ms", "expected": 60_000},
    ],
)
def test_validation_rule_passes(rule):
    executor, _ = _executor(_json(200, BODY, HEADERS))

    result = executor.execute(_request(validation_rules=[rule]))

    assert result.validation_errors == []
    assert result.status == "passed"


@pytest.mark.parametrize(
    ("rule", "error"),
    [
        ({"type": "json_path_equals", "target": "message", "expected": "other"}, 'expected "other", got "sent"'),
        ({"type": "json_path_equals", "target": "data.items[0].active", "expected": 1}, "expected 1, got true"),
        ({"type": "json_path_equals", "target": "data.items[0].id", "expected": "7"}, 'expected "7", got 7'),
        ({"type": "json_path_equals", "target": "data.items[3].id", "expected": 7}, "not found"),
        ({"type": "json_path_exists", "target": "data.total"}, "not found"),
        ({"type": "json_path_not_exists", "target": "message"}, "should not exist"),
        ({"type": "json_path_type", "target": "message", "expected": "integer"}, "expected type integer, got string"),
        ({"type": "json_path_matches", "target": "message", "expected": "^x"}, "does not match"),
        ({"type": "json_path_matches", "target": "data.items", "expected": "a"}, "expected a string"),
        ({"type": "json_path_matches", "target": "message", "expected": "("}, "Invalid regular expression"),
        ({"type": "json_path_exists", "target": "data..items"}, "Malformed JSON path"),
        ({"type": "body_contains", "expected": "missing-text"}, "does not contain"),
        ({"type": "header_exists", "target": "X-Missing"}, "is missing"),
        ({"type": "header_equals", "target": "X-Request-Id", "expected": "req-2"}, "expected 'req-2', got 'req-1'"),
        ({"type": "max_response_time_ms", "expected": -1}, "exceeds -1 ms"),
    ],
)
def test_validation_rule_fails(rule, error):
    executor, _ = _executor(_json(200, BODY, HEADERS))

    result = executor.execute(_request(validation_rules=[rule]))

    assert result.status == "failed"
    [message] = result.validation_errors
    assert error in message


def test_all_validation_errors_are_reported_together():
    executor, _ = _executor(_json(500, BODY))

    result = executor.execute(
        _request(
            validation_rules=[
                {"type": "json_path_equals", "target": "message", "expected": "other"},
                {"type": "header_exists", "target": "X-Missing"},
            ]
        )
    )

    assert len(result.validation_errors) == 3  # status + two rules


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "json_path_equals", "expected": 1},  # no target
        {"type": "json_path_equals", "target": "a"},  # no expected
        {"type": "header_exists"},
        {"type": "json_path_type", "target": "a", "expected": "text"},
        {"type": "max_response_time_ms", "expected": "fast"},
        {"type": "not_a_rule", "target": "a"},
    ],
)
def test_invalid_rule_definitions_are_rejected(rule):
    with pytest.raises(ValidationError):
        ValidationRule.model_validate(rule)


def test_resolve_json_path():
    document = {"a": {"b": [{"c": 1}]}}

    assert resolve_json_path(document, "a.b[0].c") == 1
    assert resolve_json_path(document, "$") == document
    with pytest.raises(JsonPathError):
        resolve_json_path(document, "a[b]")


# --- timeouts and exceptions -------------------------------------------------


def test_timeout_becomes_error_result():
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    executor, _ = _executor(slow, timeout_s=2.5)

    result = executor.execute(_request())

    assert result.status == "error"
    assert result.status_code is None
    assert result.error_message == "Request timed out after 2.5s (ReadTimeout)"


def test_per_request_timeout_overrides_config():
    seen = {}

    def capture(request):
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200)

    executor, _ = _executor(capture, timeout_s=10)

    executor.execute(_request(timeout_s=1.5))

    assert seen["timeout"]["read"] == 1.5


def test_connection_error_becomes_error_result():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    executor, _ = _executor(refuse)

    result = executor.execute(_request())

    assert result.status == "error"
    assert result.error_message == "ConnectError: connection refused"


def test_protocol_error_becomes_error_result():
    def broken(request):
        raise httpx.RemoteProtocolError("peer closed connection", request=request)

    result = _executor(broken)[0].execute(_request())

    assert result.status == "error"
    assert "RemoteProtocolError" in result.error_message


# --- safe configuration ------------------------------------------------------


def test_missing_base_url_errors_without_sending():
    executor, recorder = _executor(base_url=None)

    result = executor.execute(_request())

    assert result.status == "error"
    assert "API_TEST_BASE_URL" in result.error_message
    assert recorder.requests == []


def test_non_http_base_url_is_rejected():
    executor, recorder = _executor(base_url="file:///etc")

    assert "http(s)" in executor.execute(_request()).error_message
    assert recorder.requests == []


def test_absolute_url_to_unlisted_host_is_blocked():
    executor, recorder = _executor()

    result = executor.execute(_request(url="https://production.example.com/api/users"))

    assert result.status == "error"
    assert "not an allowed test host" in result.error_message
    assert recorder.requests == []


def test_absolute_url_to_allowed_host_is_permitted():
    executor, recorder = _executor(allowed_hosts=["auth.test.local"])

    result = executor.execute(_request(url="https://auth.test.local/health"))

    assert result.status == "passed"
    assert recorder.requests[0].url.host == "auth.test.local"


def test_non_http_scheme_is_blocked():
    executor, recorder = _executor()

    result = executor.execute(_request(url="ftp://api.test.local/file"))

    assert "Only http(s)" in result.error_message
    assert recorder.requests == []


def test_production_environment_is_refused_unless_explicitly_allowed():
    blocked, recorder = _executor(app_env="production")
    allowed, _ = _executor(app_env="production", allow_production=True)

    assert "production" in blocked.execute(_request()).error_message
    assert recorder.requests == []
    assert allowed.execute(_request()).status == "passed"


def test_redirects_are_not_followed_by_default():
    executor, recorder = _executor(lambda r: httpx.Response(302, headers={"Location": f"{BASE_URL}/elsewhere"}))

    result = executor.execute(_request(expected_status=302))

    assert result.status == "passed"
    assert len(recorder.requests) == 1


def test_redirect_within_test_host_is_followed_when_enabled():
    def respond(request):
        if request.url.path.endswith("/old"):
            return httpx.Response(301, headers={"Location": f"{BASE_URL}/new"})
        return httpx.Response(200, json={"ok": True})

    executor, recorder = _executor(respond, follow_redirects=True)

    result = executor.execute(_request(url="/old"))

    assert result.status == "passed"
    assert [r.url.path for r in recorder.requests] == ["/v2/old", "/v2/new"]


def test_redirect_to_foreign_host_is_blocked():
    executor, recorder = _executor(
        lambda r: httpx.Response(302, headers={"Location": "https://attacker.example.net/steal"}),
        follow_redirects=True,
    )

    result = executor.execute(_request())

    assert result.status == "error"
    assert "attacker.example.net" in result.error_message
    assert len(recorder.requests) == 1


def test_redirect_loops_are_bounded():
    executor, recorder = _executor(
        lambda r: httpx.Response(302, headers={"Location": f"{BASE_URL}/loop"}), follow_redirects=True
    )

    result = executor.execute(_request())

    assert result.status == "error"
    assert "Too many redirects" in result.error_message
    assert len(recorder.requests) == 6


def test_config_from_settings_has_no_default_target():
    config = ApiTestConfig.from_settings(Settings(_env_file=None))

    assert config.base_url is None
    assert config.permitted_hosts() == set()


def test_config_from_settings_parses_hosts_and_hides_token():
    settings = Settings(
        _env_file=None,
        api_test_base_url="http://localhost:8080",
        api_test_allowed_hosts=" auth.test.local , ,Mail.Test.Local",
        api_test_auth_token="zq-secret-771",
        api_test_timeout_s=3,
    )

    config = ApiTestConfig.from_settings(settings)

    assert config.permitted_hosts() == {"localhost", "auth.test.local", "mail.test.local"}
    assert config.timeout_s == 3
    assert "zq-secret-771" not in repr(config)


# --- LangChain tool ----------------------------------------------------------


def test_langchain_tool_executes_and_returns_structured_dict():
    executor, recorder = _executor(_json(201, {"id": 42}))
    tool = create_execute_api_test_tool(executor)

    output = tool.invoke(
        {
            "test_case_id": "TC-009",
            "method": "put",
            "url": "/items/42",
            "body": {"name": "renamed"},
            "expected_status": 201,
            "validation_rules": [{"type": "json_path_equals", "target": "id", "expected": 42}],
        }
    )

    assert tool.name == "execute_api_test"
    assert RESULT_FIELDS <= set(output)
    assert output["status"] == "passed"
    assert output["test_case_id"] == "TC-009"
    assert recorder.requests[0].method == "PUT"


def test_langchain_tool_schema_exposes_inputs():
    tool = create_execute_api_test_tool(_executor()[0])

    properties = tool.args_schema.model_json_schema()["properties"]

    assert {
        "url",
        "method",
        "headers",
        "query_params",
        "body",
        "expected_status",
        "validation_rules",
    } <= set(properties)


def test_langchain_tool_rejects_invalid_input():
    tool = create_execute_api_test_tool(_executor()[0])

    with pytest.raises(ValidationError):
        tool.invoke({"test_case_id": "TC-1", "method": "GET", "url": "/x", "expected_status": 999})


def test_execute_api_test_function_uses_given_config():
    recorder = Recorder()
    client = httpx.Client(transport=httpx.MockTransport(recorder))

    result = execute_api_test(_request(), config=ApiTestConfig(base_url=BASE_URL), client=client)

    assert result.status == "passed"
    assert not client.is_closed  # injected clients are left to their owner
