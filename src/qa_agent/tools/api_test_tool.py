"""``execute_api_test``: run one HTTP request against the system under test.

Safety rules, enforced before any network I/O:

* There is no default target: ``API_TEST_BASE_URL`` must be configured.
* Relative paths resolve against the base URL. Absolute URLs, and every
  redirect hop, must stay on the base URL's host or ``API_TEST_ALLOWED_HOSTS``.
* Only http/https. Refuses to run when ``APP_ENV=production`` unless
  ``API_TEST_ALLOW_PRODUCTION`` is explicitly set.
* Every request has a timeout; response bodies are capped in size.
* The optional auth token is a ``SecretStr`` and never appears in results.

Transport and HTTP failures never raise: they become ``status="error"``
results so a single broken test cannot crash the workflow.
"""

import json
import logging
import time
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, SecretStr

from qa_agent.config import Settings, get_settings
from qa_agent.domain.api_test import ApiTestRequest, ApiTestResult
from qa_agent.observability import safe_url, traced
from qa_agent.tools import target_policy
from qa_agent.tools.api_validation import validate_response
from qa_agent.tools.target_policy import UnsafeRequestError

logger = logging.getLogger(__name__)

_STRIPPED_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}
_MAX_REDIRECTS = 5


class ApiTestConfig(BaseModel):
    base_url: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    timeout_s: float = Field(default=10.0, gt=0)
    max_response_bytes: int = Field(default=1_000_000, gt=0)
    verify_tls: bool = True
    follow_redirects: bool = False
    auth_token: SecretStr | None = None
    app_env: str = "local"
    allow_production: bool = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ApiTestConfig":
        settings = settings or get_settings()
        return cls(
            base_url=settings.api_test_base_url or None,
            allowed_hosts=target_policy.split_hosts(settings.api_test_allowed_hosts),
            timeout_s=settings.api_test_timeout_s,
            max_response_bytes=settings.api_test_max_response_bytes,
            verify_tls=settings.api_test_verify_tls,
            follow_redirects=settings.api_test_follow_redirects,
            auth_token=settings.api_test_auth_token,
            app_env=settings.app_env,
            allow_production=settings.api_test_allow_production,
        )

    def permitted_hosts(self) -> set[str]:
        return target_policy.permitted_hosts(self.base_url, self.allowed_hosts)

    def check_usable(self) -> None:
        """Raise ``UnsafeRequestError`` if API tests must not run with this config."""
        target_policy.check_base_url(
            self.base_url,
            setting="API_TEST_BASE_URL",
            app_env=self.app_env,
            allow_production=self.allow_production,
            override="API_TEST_ALLOW_PRODUCTION",
        )


class ApiTestExecutor:
    """Executes ``ApiTestRequest``s. Pass ``client`` to inject a transport (tests)."""

    def __init__(self, config: ApiTestConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client
        self._owns_client = client is None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ApiTestExecutor":
        return cls(ApiTestConfig.from_settings(settings))

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(verify=self.config.verify_tls)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "ApiTestExecutor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- public API ---------------------------------------------------------

    @traced("api_test", run_type="tool")
    def execute(self, request: ApiTestRequest) -> ApiTestResult:
        """Send one request and validate the response (traced in LangSmith, redacted)."""
        result = self._execute(request)
        log = logger.warning if result.status == "error" else logger.info
        log(
            "API test %s %s %s -> %s %s (%.0f ms)%s",
            request.test_case_id,
            request.method,
            safe_url(result.url or request.url),
            result.status_code or "-",
            result.status,
            result.response_time or 0,
            f": {result.error_message}" if result.error_message else "",
        )
        return result

    def _execute(self, request: ApiTestRequest) -> ApiTestResult:
        base = {"test_case_id": request.test_case_id, "method": request.method}
        try:
            self.config.check_usable()
            url = self.resolve_url(request.url)
        except UnsafeRequestError as exc:
            return ApiTestResult(**base, status="error", url=request.url, error_message=str(exc))

        timeout = request.timeout_s or self.config.timeout_s
        started = time.perf_counter()
        try:
            response, raw, truncated = self._send(request, url, timeout)
        except UnsafeRequestError as exc:
            return ApiTestResult(**base, status="error", url=url, error_message=str(exc))
        except httpx.TimeoutException as exc:
            return ApiTestResult(
                **base,
                status="error",
                url=url,
                response_time=_elapsed_ms(started),
                error_message=f"Request timed out after {timeout:g}s ({type(exc).__name__})",
            )
        except httpx.HTTPError as exc:  # connect errors, protocol errors, invalid URLs, ...
            return ApiTestResult(
                **base,
                status="error",
                url=url,
                response_time=_elapsed_ms(started),
                error_message=f"{type(exc).__name__}: {exc}",
            )
        response_time = _elapsed_ms(started)

        body, body_text, is_json = _decode_body(response, raw, truncated)
        errors = validate_response(
            expected_status=request.expected_status,
            response=response,
            body=body,
            body_text=body_text,
            body_is_json=is_json,
            truncated=truncated,
            response_time_ms=response_time,
            rules=request.validation_rules,
        )
        return ApiTestResult(
            **base,
            status="failed" if errors else "passed",
            url=url,
            status_code=response.status_code,
            response_time=response_time,
            response_body=body,
            validation_errors=errors,
        )

    def resolve_url(self, url: str) -> str:
        """Absolute URL for ``url``; raises ``UnsafeRequestError`` if not permitted."""
        return target_policy.resolve(self.config.base_url, url, self.config.permitted_hosts())

    # --- internals ----------------------------------------------------------

    def _headers(self, request: ApiTestRequest) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIPPED_HEADERS}
        has_auth = any(k.lower() == "authorization" for k in headers)
        if self.config.auth_token and self.config.auth_token.get_secret_value() and not has_auth:
            headers["Authorization"] = f"Bearer {self.config.auth_token.get_secret_value()}"
        return headers

    def _send(
        self, request: ApiTestRequest, url: str, timeout: float
    ) -> tuple[httpx.Response, bytes, bool]:
        outgoing = self.client.build_request(
            request.method,
            url,
            headers=self._headers(request),
            params=request.query_params or None,
            json=request.body,
            timeout=timeout,
        )
        for _ in range(_MAX_REDIRECTS + 1):
            response = self.client.send(outgoing, stream=True, follow_redirects=False)
            try:
                if self.config.follow_redirects and response.is_redirect and response.next_request:
                    outgoing = response.next_request
                    # Every hop must stay on a test host.
                    target_policy.check_target(str(outgoing.url), self.config.permitted_hosts())
                    continue
                raw, truncated = self._read_capped(response)
                return response, raw, truncated
            finally:
                response.close()
        raise UnsafeRequestError(f"Too many redirects (more than {_MAX_REDIRECTS})")

    def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        limit = self.config.max_response_bytes
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                return b"".join(chunks)[:limit], True
        return b"".join(chunks), False


def execute_api_test(
    request: ApiTestRequest,
    *,
    config: ApiTestConfig | None = None,
    client: httpx.Client | None = None,
) -> ApiTestResult:
    """Run one API test with the configured (or given) settings."""
    with ApiTestExecutor(config or ApiTestConfig.from_settings(), client) as executor:
        return executor.execute(request)


def create_execute_api_test_tool(executor: ApiTestExecutor) -> BaseTool:
    """``execute_api_test`` as a LangChain tool bound to ``executor``."""

    def _run(**kwargs: Any) -> dict:
        return executor.execute(ApiTestRequest.model_validate(kwargs)).model_dump(mode="json")

    return StructuredTool.from_function(
        func=_run,
        name="execute_api_test",
        description=(
            "Execute one HTTP API test (GET, POST, PUT, PATCH or DELETE) against the configured "
            "test environment and validate the response. `url` is a path relative to the test "
            "base URL. Returns test_case_id, status (passed/failed/error), status_code, "
            "response_time (ms), response_body, validation_errors and error_message."
        ),
        args_schema=ApiTestRequest,
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _decode_body(response: httpx.Response, raw: bytes, truncated: bool) -> tuple[Any, str, bool]:
    """(body, text, is_json). JSON bodies are parsed; others are returned as text."""
    if not raw:
        return None, "", False
    text = raw.decode(response.encoding or "utf-8", errors="replace")
    if truncated:
        return text + " [truncated]", text, False
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        try:
            return json.loads(text), text, True
        except json.JSONDecodeError:
            return text, text, False
    return text, text, False
