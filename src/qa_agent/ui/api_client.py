"""HTTP client for the QA Agent backend.

The UI talks to the workflow only through these calls. Every failure
(backend down, timeout, 4xx/5xx) becomes a ``BackendError`` with a message
fit to show to the user, so pages never see raw httpx exceptions.

Correlation: each call sends a fresh ``X-Request-ID``, the Streamlit session
id (``X-Client-Session``) and, for run endpoints, ``X-QA-Run-ID``. The backend
logs and LangSmith traces carry the same ids, so a UI error message's request
id can be looked up in the backend log.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)

_RUN_PATH = re.compile(r"^/runs/([A-Za-z0-9]{8,64})(?:/|$)")


class BackendError(Exception):
    def __init__(
        self, message: str, *, status_code: int | None = None, hint: str | None = None, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.hint = hint
        self.request_id = request_id

    @property
    def unreachable(self) -> bool:
        return self.status_code is None


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or response.reason_phrase
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):  # FastAPI validation errors
        parts = []
        for item in detail:
            loc = [str(p) for p in item.get("loc", []) if p not in ("body", "query", "path")]
            parts.append(f"{'.'.join(loc)}: {item.get('msg')}" if loc else str(item.get("msg")))
        return "; ".join(parts)
    return str(detail)


class QAApiClient:
    def __init__(
        self, base_url: str, *, timeout_s: float = 15.0, session_id: Callable[[], str | None] | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session_id = session_id or (lambda: None)
        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=3.0))

    def _headers(self, path: str, request_id: str) -> dict[str, str]:
        headers = {"X-Request-ID": request_id}
        if session := self._session_id():
            headers["X-Client-Session"] = session
        if match := _RUN_PATH.match(path):
            headers["X-QA-Run-ID"] = match.group(1)
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        request_id = uuid4().hex
        started = time.perf_counter()
        try:
            response = self._client.request(method, path, headers=self._headers(path, request_id), **kwargs)
        except httpx.ConnectError as exc:
            logger.warning("%s %s failed: backend unreachable (req=%s)", method, path, request_id[:8])
            raise BackendError(
                f"Cannot reach the QA Agent backend at {self.base_url}.",
                hint="Start it with: uvicorn qa_agent.api.main:app --app-dir src --port 8000",
                request_id=request_id,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.warning("%s %s timed out (req=%s)", method, path, request_id[:8])
            raise BackendError(
                "The backend did not respond in time. It may be busy; try again.", request_id=request_id
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("%s %s failed: %s (req=%s)", method, path, exc, request_id[:8])
            raise BackendError(f"Request to the backend failed: {exc}", request_id=request_id) from exc
        logger.debug(
            "%s %s -> %d (%.0f ms, req=%s)",
            method, path, response.status_code, (time.perf_counter() - started) * 1000, request_id[:8],
        )
        if response.status_code >= 400:
            hints = {
                404: "The backend keeps runs in memory; a backend restart clears them.",
                409: "The run moved on in the meantime. Refresh to see its current status.",
                422: "Check the values you entered.",
            }
            logger.warning("%s %s -> %d (req=%s)", method, path, response.status_code, request_id[:8])
            raise BackendError(
                _detail(response),
                status_code=response.status_code,
                hint=hints.get(response.status_code, "See the backend logs for details." if response.status_code >= 500 else None),
                request_id=request_id,
            )
        return response

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None}).json()

    def _post(self, path: str, body: Any = None) -> Any:
        return self._request("POST", path, json=body).json()

    # system
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def system_status(self) -> dict[str, Any]:
        return self._get("/system/status")

    # projects
    def list_projects(self) -> list[dict[str, Any]]:
        return self._get("/projects")

    def create_project(self, name: str, description: str | None = None) -> dict[str, Any]:
        return self._post("/projects", {"name": name, "description": description or None})

    # runs
    def start_run(
        self, requirement: str, *, project: str, metadata: dict[str, Any], pause_before_execution: bool
    ) -> dict[str, Any]:
        return self._post(
            "/runs",
            {
                "requirement": requirement,
                "project": project,
                "metadata": metadata,
                "pause_before_execution": pause_before_execution,
            },
        )

    def start_ticket_run(
        self, ticket: dict[str, Any], *, project: str, metadata: dict[str, Any], pause_before_execution: bool
    ) -> dict[str, Any]:
        """Verify a service desk bug ticket (the backend turns it into the requirement)."""
        return self._post(
            "/runs",
            {
                "bug_ticket": ticket,
                "project": project,
                "metadata": metadata,
                "pause_before_execution": pause_before_execution,
            },
        )

    def list_runs(self, project: str | None = None) -> list[dict[str, Any]]:
        return self._get("/runs", project=project)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{run_id}")

    def requirement_analysis(self, run_id: str) -> dict[str, Any] | None:
        return self._get(f"/runs/{run_id}/requirement-analysis")

    def test_cases(self, run_id: str) -> list[dict[str, Any]]:
        return self._get(f"/runs/{run_id}/test-cases")

    def test_plan(self, run_id: str) -> dict[str, Any] | None:
        return self._get(f"/runs/{run_id}/test-plan")

    def execute(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/runs/{run_id}/execute")

    def results(self, run_id: str) -> list[dict[str, Any]]:
        return self._get(f"/runs/{run_id}/results")

    def failures(self, run_id: str) -> list[dict[str, Any]]:
        return self._get(f"/runs/{run_id}/failures")

    def root_cause(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{run_id}/root-cause")

    def bug_reports(self, run_id: str) -> list[dict[str, Any]]:
        return self._get(f"/runs/{run_id}/bugs")

    def confidence(self, run_id: str) -> dict[str, Any] | None:
        return self._get(f"/runs/{run_id}/confidence")

    def review(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{run_id}/review")

    def submit_review(self, run_id: str, decision: str, comment: str | None, reviewer: str | None) -> dict[str, Any]:
        return self._post(
            f"/runs/{run_id}/review",
            {"decision": decision, "comment": comment or None, "reviewer": reviewer or None},
        )

    def report(self, run_id: str) -> dict[str, Any] | None:
        return self._get(f"/runs/{run_id}/report")

    def traces(self, run_id: str) -> list[dict[str, Any]]:
        return self._get(f"/runs/{run_id}/traces")
