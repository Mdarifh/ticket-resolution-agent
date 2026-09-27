"""Correlation context: the ids that tie one QA run together across layers.

* ``run_id``: the QA run (also the LangGraph thread id, the LangSmith
  ``qa_run_id``/``thread_id`` metadata, and the ``agent_runs`` primary key).
* ``request_id``: one HTTP request (sent by Streamlit as ``X-Request-ID``).
* ``client_session``: one Streamlit browser session (``X-Client-Session``).
* ``segment``: the workflow segment being executed (start / execute / review).
* ``node``: the LangGraph node currently running.

Values live in context variables, so log records and traces pick them up
without passing ids through every function. Worker threads must be started
with ``contextvars.copy_context().run`` to inherit them.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

FIELDS = ("run_id", "request_id", "client_session", "project", "segment", "node")

_vars: dict[str, ContextVar[str | None]] = {name: ContextVar(f"qa_{name}", default=None) for name in FIELDS}
_trace_listener: ContextVar[Callable[[Any], None] | None] = ContextVar("qa_trace_listener", default=None)


def get(name: str) -> str | None:
    return _vars[name].get()


def current() -> dict[str, str]:
    """All bound ids (unset ones omitted)."""
    return {name: value for name in FIELDS if (value := _vars[name].get())}


@contextmanager
def bind(*, on_trace: Callable[[Any], None] | None = None, **values: str | None) -> Iterator[None]:
    """Bind ids for the duration of the block (``None`` values are ignored)."""
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise ValueError(f"Unknown context field(s): {sorted(unknown)}")
    tokens = [(_vars[name], _vars[name].set(value)) for name, value in values.items() if value is not None]
    listener_token = _trace_listener.set(on_trace) if on_trace is not None else None
    try:
        yield
    finally:
        if listener_token is not None:
            _trace_listener.reset(listener_token)
        for var, token in reversed(tokens):
            var.reset(token)


def trace_listener() -> Callable[[Any], None] | None:
    return _trace_listener.get()
