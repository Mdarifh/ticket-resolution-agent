"""Application logging, shared by the FastAPI and Streamlit entry points.

Every record carries the correlation ids bound in ``qa_agent.observability.context``
(run, request, Streamlit session, workflow node), and every formatted line,
including tracebacks, passes through secret redaction before it is written.

    2026-09-26 23:40:12 | INFO | run=5455144e req=9f2c01ab node=test_executor | qa_agent.tools.api_test_tool | API test TC-001 POST https://sut/api/v2/password-reset -> 202 passed (84 ms)

``LOG_FORMAT=json`` emits one JSON object per line with the same fields.
"""

import json
import logging
import sys
from datetime import UTC, datetime

from qa_agent.config import get_settings
from qa_agent.observability import context
from qa_agent.observability.redaction import redact_text

_CONFIGURED = False

_SHORT = {"run_id": ("run", 8), "request_id": ("req", 8), "client_session": ("session", 8), "segment": ("segment", 0), "node": ("node", 0)}


class ContextFilter(logging.Filter):
    """Attach the bound correlation ids to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ids = context.current()
        record.qa_context = ids
        parts = []
        for name, (label, width) in _SHORT.items():
            if name in ids:
                parts.append(f"{label}={ids[name][:width] if width else ids[name]}")
        record.qa_ids = " ".join(parts) or "-"
        return True


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "qa_context", {}),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return redact_text(json.dumps(entry, default=str))


def build_handler(log_format: str = "text") -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            RedactingFormatter(
                fmt="%(asctime)s | %(levelname)-7s | %(qa_ids)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    return handler


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    handler = build_handler(settings.log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Route uvicorn through the same redacting handler. Its access log is replaced by
    # the API's request middleware (which knows the request/run ids), so keep it quiet.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Chatty libraries: their DEBUG output can include request details.
    for name in ("httpx", "httpx2", "httpcore", "openai", "urllib3", "langsmith"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root_logger.level))

    _CONFIGURED = True
