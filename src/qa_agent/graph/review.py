"""Pause/resume helpers for human-in-the-loop review.

A run pauses inside the ``human_review`` node via LangGraph's ``interrupt()``;
its state is kept by the checkpointer under ``thread_id = run_id``. Any graph
compiled with the same checkpointer can report the pending review and resume
the run, including after the original process is gone (with a persistent
checkpointer).
"""

from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from qa_agent.domain.review import HumanReview, HumanReviewDecision
from qa_agent.graph.build_graph import run_config
from qa_agent.graph.runner import NodeCallback, advance


def get_pending_review(graph: CompiledStateGraph, run_id: str) -> HumanReview | None:
    """The review a paused run is waiting on, or None if it is not waiting."""
    snapshot = graph.get_state(run_config(run_id))
    for task in snapshot.tasks:
        for pending in task.interrupts:
            return HumanReview.model_validate(pending.value["review"])
    return None


def submit_review_decision(
    graph: CompiledStateGraph,
    run_id: str,
    decision: HumanReviewDecision | dict[str, Any],
    *,
    on_node: NodeCallback | None = None,
) -> dict[str, Any]:
    """Validate the decision and resume the paused run; returns the new state."""
    if get_pending_review(graph, run_id) is None:
        raise ValueError(f"Run {run_id!r} is not waiting for a human review")
    validated = HumanReviewDecision.model_validate(decision)
    return advance(graph, Command(resume=validated.model_dump()), run_id, on_node=on_node, segment="review")
