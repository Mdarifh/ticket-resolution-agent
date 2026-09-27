"""LangChain callback handler that writes LLM, tool and retriever activity to the app log.

It complements LangSmith: the log says *that* a call happened, how long it took
and how many tokens it used, tagged with the run id; LangSmith holds the full
prompts and outputs. Prompt and response text is never logged here.
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("qa_agent.langchain")


def _model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    params = kwargs.get("invocation_params") or {}
    for key in ("model", "model_name"):
        if params.get(key):
            return str(params[key])
    if serialized:
        return str((serialized.get("kwargs") or {}).get("model_name") or serialized.get("name") or "unknown")
    return "unknown"


def _token_usage(response: Any) -> dict[str, int]:
    usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if usage:
        return {k: int(v) for k, v in usage.items() if isinstance(v, int)}
    for generations in getattr(response, "generations", None) or []:
        for generation in generations:
            meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if meta:
                return {k: int(v) for k, v in dict(meta).items() if isinstance(v, int)}
    return {}


class LoggingCallbackHandler(BaseCallbackHandler):
    raise_error = False

    def __init__(self) -> None:
        self._started: dict[UUID, tuple[float, str]] = {}

    def _start(self, run_id: UUID, label: str) -> None:
        self._started[run_id] = (time.perf_counter(), label)

    def _finish(self, run_id: UUID) -> tuple[str, float]:
        started, label = self._started.pop(run_id, (time.perf_counter(), "unknown"))
        return label, (time.perf_counter() - started) * 1000

    # LLM calls
    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, _model_name(serialized, kwargs))

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, _model_name(serialized, kwargs))

    def on_llm_end(self, response, *, run_id: UUID, **kwargs: Any) -> None:
        model, ms = self._finish(run_id)
        usage = _token_usage(response)
        tokens = " ".join(f"{k}={v}" for k, v in usage.items() if k in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"))
        logger.info("LLM call finished model=%s duration_ms=%.0f %s", model, ms, tokens or "tokens=n/a")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        model, ms = self._finish(run_id)
        logger.warning("LLM call failed model=%s duration_ms=%.0f error=%s: %s", model, ms, type(error).__name__, error)

    # LangChain tools
    def on_tool_start(self, serialized, input_str, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, (serialized or {}).get("name", "tool"))

    def on_tool_end(self, output, *, run_id: UUID, **kwargs: Any) -> None:
        name, ms = self._finish(run_id)
        logger.info("Tool %s finished duration_ms=%.0f", name, ms)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name, ms = self._finish(run_id)
        logger.warning("Tool %s failed duration_ms=%.0f error=%s: %s", name, ms, type(error).__name__, error)

    # Retrievers
    def on_retriever_start(self, serialized, query, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, (serialized or {}).get("name", "retriever"))

    def on_retriever_end(self, documents, *, run_id: UUID, **kwargs: Any) -> None:
        name, ms = self._finish(run_id)
        logger.info("Retriever %s returned %d documents duration_ms=%.0f", name, len(documents), ms)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name, ms = self._finish(run_id)
        logger.warning("Retriever %s failed duration_ms=%.0f error=%s: %s", name, ms, type(error).__name__, error)

    # Chains: only a failure of the whole segment is worth a log line; inner chain
    # errors are either retried, absorbed by the node wrapper, or bubble up to here.
    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        if parent_run_id is None and type(error).__name__ != "GraphInterrupt":
            logger.error("Workflow segment failed: %s: %s", type(error).__name__, error)
