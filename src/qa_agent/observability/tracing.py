"""LangSmith tracing.

What is traced, and how:

* LangGraph execution, LangChain chains and LLM calls: automatically, once the
  ``LANGSMITH_*`` environment variables are set (``configure_tracing`` exports
  them from ``.env``, which LangChain would not read on its own).
* Test tools (API requests, browser runs) and knowledge base retrieval: plain
  Python, so they are wrapped with ``traced`` (LangSmith ``@traceable``) and
  nest under the LangGraph node that called them. Inputs/outputs are redacted.
* Failures: exceptions inside traced code mark the span as errored; node
  errors that the workflow absorbs are recorded as an errored ``node_failure``
  span so they stand out in the trace.

Each workflow segment is one LangSmith trace whose root run id is chosen here
(``graph_trace_config``), carries ``qa_run_id``/``thread_id`` metadata, and is
reported to the run manager so the API and UI can show it.

With tracing off, every helper here is a cheap no-op.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, TypeVar
from uuid import UUID, uuid4

from langsmith import traceable
from langsmith.utils import tracing_is_enabled

from qa_agent.config import Settings, get_settings
from qa_agent.observability import context
from qa_agent.observability.redaction import redact

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ENV_FLAGS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")


@dataclass(frozen=True)
class TracingStatus:
    enabled: bool
    project: str | None = None
    endpoint: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class TraceRecord:
    """One LangSmith trace = one workflow segment of a QA run."""

    segment: str
    trace_id: str
    started_at: datetime


_status = TracingStatus(enabled=False, reason="not configured yet")


def configure_tracing(settings: Settings | None = None) -> TracingStatus:
    """Export LangSmith settings to the environment LangChain/LangSmith read. Idempotent."""
    global _status
    settings = settings or get_settings()
    key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else ""

    if settings.langsmith_tracing and key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        _status = TracingStatus(True, settings.langsmith_project, settings.langsmith_endpoint)
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    else:
        # Make sure nothing tries to trace (and fail with 401s) half-configured.
        for flag in _ENV_FLAGS:
            os.environ[flag] = "false"
        reason = "LANGSMITH_API_KEY is not set" if settings.langsmith_tracing else "LANGSMITH_TRACING is off"
        if settings.langsmith_tracing:
            logger.warning("LangSmith tracing requested but disabled: %s", reason)
        else:
            logger.info("LangSmith tracing disabled (%s)", reason)
        _status = TracingStatus(False, settings.langsmith_project, reason=reason)
    _trace_url_base.cache_clear()
    return _status


def tracing_status() -> TracingStatus:
    return _status


def graph_trace_config(run_id: str, segment: str) -> dict[str, Any]:
    """Invocation config for one workflow segment: its root run id, name, tags, metadata
    and the logging callback. LangChain propagates metadata/tags/callbacks to every
    node, chain, LLM call and traceable tool inside the segment."""
    from qa_agent.observability.callbacks import LoggingCallbackHandler

    ids = context.current()
    trace_id = uuid4()
    metadata: dict[str, Any] = {
        "qa_run_id": run_id,
        "thread_id": run_id,  # groups a run's segments as one LangSmith thread
        "segment": segment,
        "app_env": get_settings().app_env,
    }
    metadata.update({k: ids[k] for k in ("project", "request_id", "client_session") if k in ids})
    tags = ["qa-agent", f"segment:{segment}"] + ([f"project:{ids['project']}"] if "project" in ids else [])

    listener = context.trace_listener()
    if listener is not None:
        listener(TraceRecord(segment=segment, trace_id=str(trace_id), started_at=datetime.now(UTC)))
    logger.info("Workflow segment %s started (trace_id=%s, tracing=%s)", segment, trace_id, _status.enabled)
    return {
        "run_id": trace_id,
        "run_name": f"qa_run.{segment}",
        "tags": tags,
        "metadata": metadata,
        "callbacks": [LoggingCallbackHandler()],
    }


def _clean_inputs(inputs: dict) -> dict:
    return redact({k: v for k, v in inputs.items() if k != "self"})


def _clean_outputs(outputs: Any) -> dict:
    cleaned = redact(outputs)
    return cleaned if isinstance(cleaned, dict) else {"output": cleaned}


def traced(name: str, *, run_type: str = "tool") -> Callable[[F], F]:
    """``@traceable`` with redacted inputs/outputs (``self`` dropped)."""

    def decorator(fn: F) -> F:
        return traceable(  # type: ignore[return-value]
            name=name, run_type=run_type, process_inputs=_clean_inputs, process_outputs=_clean_outputs
        )(fn)

    return decorator


class NodeFailure(Exception):
    """Raised only inside the ``node_failure`` span, to mark it errored in LangSmith."""


@traceable(name="node_failure", run_type="chain", process_inputs=_clean_inputs)
def _node_failure_span(node: str, error_type: str, message: str) -> None:
    raise NodeFailure(f"{node}: {error_type}: {message}")


def record_node_failure(node: str, exc: BaseException) -> None:
    """Make an error the workflow recovers from visible in the trace."""
    if not tracing_is_enabled():
        return
    try:
        _node_failure_span(node, type(exc).__name__, str(exc))
    except NodeFailure:
        pass


@lru_cache(maxsize=1)
def _trace_url_base() -> str | None:
    """``<host>/o/<tenant>/projects/p/<project>/r/`` via one LangSmith API call (cached)."""
    from langsmith import Client
    from langsmith.schemas import RunBase

    placeholder = RunBase(id=UUID(int=0), name="x", start_time=datetime.now(UTC), run_type="chain", inputs={})
    url = Client().get_run_url(run=placeholder, project_name=_status.project)
    return url.split(str(placeholder.id))[0]


def trace_url(trace_id: str) -> str | None:
    """Link to a trace in the LangSmith UI, or None when tracing is off or LangSmith is unreachable."""
    if not _status.enabled:
        return None
    try:
        base = _trace_url_base()
    except Exception as exc:  # network, auth, unknown project: links are optional
        logger.warning("Could not resolve LangSmith trace URL: %s", exc)
        return None
    return f"{base}{trace_id}?poll=true" if base else None


def flush_traces() -> None:
    """Send pending traces before the process exits."""
    if not _status.enabled:
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:  # never block shutdown on tracing
        logger.warning("Flushing LangSmith traces failed", exc_info=True)
