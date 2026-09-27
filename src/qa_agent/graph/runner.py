"""Drives the compiled graph one segment at a time and reports node progress.

A run executes in segments. Each segment ends when the graph finishes or
pauses: at the optional execution gate (before ``test_executor``, so a person
can look at the generated tests before they hit the system under test) or
inside ``human_review`` (the ``interrupt()`` used for low-confidence results).
Both pauses are ordinary LangGraph checkpoints keyed by ``thread_id = run_id``.
"""

from collections.abc import Callable
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from qa_agent.graph.build_graph import NODE_NAMES, TEST_EXECUTOR, run_config
from qa_agent.observability import context
from qa_agent.observability.tracing import graph_trace_config

NodeCallback = Callable[[str], None]

EXECUTION_GATE = TEST_EXECUTOR


def advance(
    graph: CompiledStateGraph,
    graph_input: Any,
    run_id: str,
    *,
    on_node: NodeCallback | None = None,
    pause_before_execution: bool = False,
    segment: str = "run",
) -> dict[str, Any]:
    """Run the graph until it finishes or pauses; returns the resulting state.

    ``graph_input`` is the initial state, ``None`` to continue past the execution
    gate, or a ``Command(resume=...)`` for a human review. ``on_node`` is called
    with each node name as the node finishes. ``segment`` names the LangSmith
    trace of this stretch of the run (``qa_run.<segment>``).
    """
    kwargs = {"interrupt_before": [EXECUTION_GATE]} if pause_before_execution else {}
    with context.bind(run_id=run_id, segment=segment):
        config = {**run_config(run_id), **graph_trace_config(run_id, segment)}
        for chunk in graph.stream(graph_input, config, stream_mode="updates", **kwargs):
            for node in chunk:
                if on_node is not None and node in NODE_NAMES:
                    on_node(node)
    return dict(graph.get_state(run_config(run_id)).values)


def is_awaiting_execution(graph: CompiledStateGraph, run_id: str) -> bool:
    """True when the run is paused at the execution gate (not at a human review)."""
    snapshot = graph.get_state(run_config(run_id))
    return EXECUTION_GATE in snapshot.next and not any(task.interrupts for task in snapshot.tasks)


def next_nodes(graph: CompiledStateGraph, run_id: str) -> tuple[str, ...]:
    """Nodes the run will execute next (the ones running now, while it runs)."""
    return tuple(graph.get_state(run_config(run_id)).next)
