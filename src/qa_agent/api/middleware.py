"""Request correlation and access logging.

Every request gets a request id: the caller's ``X-Request-ID`` (the Streamlit
client sends one per call) when well-formed, otherwise a new one. It is bound
to the log context together with the Streamlit session (``X-Client-Session``)
and the run id taken from ``/runs/{run_id}/...`` paths, and echoed back in the
``X-Request-ID`` / ``X-QA-Run-ID`` response headers. Background workflow
segments started by the request inherit these ids.

GET requests are logged at DEBUG (the dashboard polls every few seconds);
changes and errors at INFO/WARNING.
"""

import logging
import re
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from qa_agent.observability import context

logger = logging.getLogger("qa_agent.api.access")

_ID = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
_RUN_PATH = re.compile(r"^/runs/([A-Za-z0-9]{8,64})(?:/|$)")

REQUEST_ID_HEADER = "X-Request-ID"
SESSION_HEADER = "X-Client-Session"
RUN_ID_HEADER = "X-QA-Run-ID"


def _valid(value: str | None) -> str | None:
    return value if value and _ID.match(value) else None


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlate(request: Request, call_next) -> Response:
        request_id = _valid(request.headers.get(REQUEST_ID_HEADER)) or uuid4().hex
        session = _valid(request.headers.get(SESSION_HEADER))
        match = _RUN_PATH.match(request.url.path)
        run_id = match.group(1) if match else _valid(request.headers.get(RUN_ID_HEADER))

        started = time.perf_counter()
        with context.bind(request_id=request_id, client_session=session, run_id=run_id):
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("%s %s -> unhandled error", request.method, request.url.path)
                raise
            ms = (time.perf_counter() - started) * 1000
            if request.url.path != "/health":
                if response.status_code >= 500:
                    level = logging.ERROR
                elif response.status_code >= 400:
                    level = logging.WARNING
                elif request.method == "GET":
                    level = logging.DEBUG
                else:
                    level = logging.INFO
                logger.log(level, "%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code, ms)

        response.headers[REQUEST_ID_HEADER] = request_id
        if run_id and RUN_ID_HEADER not in response.headers:
            response.headers[RUN_ID_HEADER] = run_id
        return response
