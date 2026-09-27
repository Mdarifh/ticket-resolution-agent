"""RunService: runs the QA workflow and records it in the database.

This is the seam between LangGraph and persistence: the graph stays free of
database code, and after every segment (paused before test execution, paused
for review, finished, or crashed) the service syncs the resulting state
through ``PersistenceService``.
"""

from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from qa_agent.domain import FinalReport, Requirement
from qa_agent.domain.review import HumanReview, HumanReviewDecision
from qa_agent.graph import initial_state
from qa_agent.graph.review import get_pending_review, submit_review_decision
from qa_agent.graph.runner import NodeCallback, advance, is_awaiting_execution
from qa_agent.persistence.service import PersistenceService


class RunOutcome(BaseModel):
    run_id: str
    status: str  # awaiting_execution | awaiting_review | completed
    pending_review: HumanReview | None = None
    final_report: FinalReport | None = None


class RunService:
    def __init__(self, graph: CompiledStateGraph, persistence: PersistenceService | None = None) -> None:
        self.graph = graph
        self.persistence = persistence

    def start(
        self,
        requirement_text: str,
        *,
        project: str = "default",
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        pause_before_execution: bool = False,
        on_node: NodeCallback | None = None,
    ) -> RunOutcome:
        """Start a run. With ``pause_before_execution`` it stops after test planning
        (status ``awaiting_execution``) until ``execute`` is called."""
        requirement = Requirement(text=requirement_text, metadata=metadata or {}, **({"id": run_id} if run_id else {}))
        if self.persistence:
            self.persistence.start_run(requirement, project_name=project)
        return self._invoke(
            requirement.id,
            lambda: advance(
                self.graph,
                initial_state(requirement),
                requirement.id,
                on_node=on_node,
                pause_before_execution=pause_before_execution,
                segment="start",
            ),
        )

    def execute(self, run_id: str, *, on_node: NodeCallback | None = None) -> RunOutcome:
        """Continue a run paused before test execution."""
        if not is_awaiting_execution(self.graph, run_id):
            raise ValueError(f"Run {run_id!r} is not waiting to execute tests")
        return self._invoke(run_id, lambda: advance(self.graph, None, run_id, on_node=on_node, segment="execute"))

    def decide(
        self, run_id: str, decision: HumanReviewDecision | dict[str, Any], *, on_node: NodeCallback | None = None
    ) -> RunOutcome:
        return self._invoke(run_id, lambda: submit_review_decision(self.graph, run_id, decision, on_node=on_node))

    def _invoke(self, run_id: str, call) -> RunOutcome:
        try:
            state = call()
        except Exception as exc:
            # Validation errors on a decision leave the run paused; anything else is a crash.
            if self.persistence and not isinstance(exc, ValueError):
                self.persistence.mark_failed(run_id, f"{type(exc).__name__}: {exc}")
            raise
        pending = get_pending_review(self.graph, run_id)
        awaiting_execution = pending is None and is_awaiting_execution(self.graph, run_id)
        if self.persistence:
            self.persistence.sync_state(
                run_id, state, awaiting_review=pending is not None, awaiting_execution=awaiting_execution
            )
        if pending:
            status = "awaiting_review"
        elif awaiting_execution:
            status = "awaiting_execution"
        else:
            status = "completed"
        return RunOutcome(
            run_id=run_id,
            status=status,
            pending_review=pending,
            final_report=state.get("final_report"),
        )
