"""Wraps every node so the graph records visits and captures node errors.

A node that raises does not crash the run: the exception is recorded in
``state["errors"]`` and the graph continues along its normal edges, so the
final report can surface the failure. LangGraph control-flow exceptions
(e.g. the ``interrupt()`` used by human review) are re-raised untouched.

The wrapper is also the node's observability hook: the node name is bound to
the log context, each node logs its duration, and an absorbed error is both
logged and recorded as an errored span in the LangSmith trace.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphBubbleUp

from qa_agent.domain import WorkflowError
from qa_agent.graph.state import QAAgentState
from qa_agent.observability import context
from qa_agent.observability.tracing import record_node_failure

logger = logging.getLogger(__name__)

NodeFn = Callable[[QAAgentState], dict[str, Any]]


def instrument(name: str, fn: NodeFn) -> NodeFn:
    def wrapped(state: QAAgentState) -> dict[str, Any]:
        with context.bind(node=name):
            started = time.perf_counter()
            logger.debug("Node %s started", name)
            try:
                update = dict(fn(state) or {})
            except GraphBubbleUp:
                logger.info("Node %s paused the run (%.0f ms)", name, (time.perf_counter() - started) * 1000)
                raise
            except Exception as exc:
                logger.error(
                    "Node %s failed after %.0f ms: %s: %s",
                    name,
                    (time.perf_counter() - started) * 1000,
                    type(exc).__name__,
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                record_node_failure(name, exc)
                metadata = state.get("execution_metadata")
                visit = metadata.nodes_visited.count(name) if metadata else 0
                update = {
                    "errors": [
                        WorkflowError(node=name, message=str(exc), error_type=type(exc).__name__, visit=visit)
                    ]
                }
            else:
                logger.info("Node %s finished in %.0f ms", name, (time.perf_counter() - started) * 1000)

        metadata_patch = dict(update.pop("execution_metadata", None) or {})
        metadata_patch["nodes_visited"] = [name]
        update["execution_metadata"] = metadata_patch
        return update

    wrapped.__name__ = name
    return wrapped
