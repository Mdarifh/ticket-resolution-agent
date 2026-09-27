"""RunManager: runs workflows in the background for the HTTP API.

A workflow segment takes from seconds to minutes (LLM calls, real API and
browser tests), so the API answers immediately and each segment runs on a
worker thread. The manager records live progress per run (which nodes
finished and when, which one runs now, whether the run is paused or crashed)
so the UI can poll it. Business logic stays in the graph and ``RunService``;
this class only schedules segments and keeps the bookkeeping.

Progress lives in process memory, like the default in-memory checkpointer:
runs started by an earlier backend process cannot be resumed from here.
"""

import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from qa_agent.domain.review import HumanReview, HumanReviewDecision
from qa_agent.graph.build_graph import run_config
from qa_agent.graph.review import get_pending_review
from qa_agent.graph.runner import NodeCallback, next_nodes
from qa_agent.observability import context
from qa_agent.observability.tracing import TraceRecord
from qa_agent.services.run_service import RunOutcome, RunService

logger = logging.getLogger(__name__)

RunPhase = Literal["running", "awaiting_execution", "awaiting_review", "completed", "error"]

DEFAULT_PROJECT = "default"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunNotFoundError(LookupError):
    pass


class RunConflictError(RuntimeError):
    """The run is not in a state that allows the requested action."""


@dataclass
class NodeEvent:
    node: str
    started_at: datetime
    finished_at: datetime


@dataclass
class LiveRun:
    run_id: str
    project: str
    requirement: str
    metadata: dict[str, Any]
    pause_before_execution: bool
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    phase: RunPhase = "running"
    error: str | None = None
    events: list[NodeEvent] = field(default_factory=list)
    traces: list[TraceRecord] = field(default_factory=list)
    segment_started_at: datetime = field(default_factory=_utcnow)

    @property
    def current_node_started_at(self) -> datetime:
        """When the node running now started: the end of the last node of this segment."""
        if self.events and self.events[-1].finished_at >= self.segment_started_at:
            return self.events[-1].finished_at
        return self.segment_started_at


class RunManager:
    def __init__(self, service: RunService, *, max_workers: int = 2) -> None:
        self.service = service
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qa-run")
        self._runs: dict[str, LiveRun] = {}
        self._projects: dict[str, str | None] = {DEFAULT_PROJECT: "Default project"}
        self._lock = threading.Lock()

    # --- projects -------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        projects = dict(self._projects)
        if self.service.persistence:
            try:
                for stored in self.service.persistence.list_projects():
                    projects.setdefault(stored["name"], stored["description"])
            except Exception:  # the database being down must not hide the live runs
                logger.exception("Could not list projects from the database")
        with self._lock:
            runs = list(self._runs.values())
        for run in runs:
            projects.setdefault(run.project, None)
        return [
            {"name": name, "description": description, "run_count": sum(r.project == name for r in runs)}
            for name, description in sorted(projects.items(), key=lambda item: (item[0] != DEFAULT_PROJECT, item[0]))
        ]

    def create_project(self, name: str, description: str | None = None) -> dict[str, Any]:
        if self.service.persistence:
            self.service.persistence.create_project(name, description)
        self._projects.setdefault(name, description)
        return {"name": name, "description": self._projects[name], "run_count": self._count_runs(name)}

    def _count_runs(self, project: str) -> int:
        with self._lock:
            return sum(r.project == project for r in self._runs.values())

    # --- runs -----------------------------------------------------------------

    def start(
        self,
        requirement: str,
        *,
        project: str = DEFAULT_PROJECT,
        metadata: dict[str, Any] | None = None,
        pause_before_execution: bool = True,
    ) -> LiveRun:
        run = LiveRun(
            run_id=uuid4().hex,
            project=project,
            requirement=requirement,
            metadata=dict(metadata or {}),
            pause_before_execution=pause_before_execution,
        )
        with self._lock:
            self._runs[run.run_id] = run
        self._projects.setdefault(project, None)
        logger.info("Run %s created in project %s (execution gate %s)", run.run_id, project,
                    "on" if pause_before_execution else "off")
        self._submit(
            run,
            "start",
            lambda on_node: self.service.start(
                requirement,
                project=project,
                metadata=run.metadata,
                run_id=run.run_id,
                pause_before_execution=pause_before_execution,
                on_node=on_node,
            ),
        )
        return run

    def execute(self, run_id: str) -> LiveRun:
        run = self._claim(run_id, "awaiting_execution", "Test execution can start only while the run waits for it")
        self._submit(run, "execute", lambda on_node: self.service.execute(run_id, on_node=on_node))
        return run

    def decide(self, run_id: str, decision: HumanReviewDecision) -> LiveRun:
        run = self._claim(run_id, "awaiting_review", "The run is not waiting for a human review")
        logger.info("Run %s review decision %s by %s", run_id, decision.decision, decision.reviewer or "anonymous")
        self._submit(run, "review", lambda on_node: self.service.decide(run_id, decision, on_node=on_node))
        return run

    def get(self, run_id: str) -> LiveRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(f"Run {run_id!r} not found (runs are kept in memory by this backend process)")
        return run

    def list(self, project: str | None = None) -> list[LiveRun]:
        with self._lock:
            runs = list(self._runs.values())
        if project:
            runs = [r for r in runs if r.project == project]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def state(self, run_id: str) -> dict[str, Any]:
        """Latest checkpointed workflow state of the run."""
        self.get(run_id)
        return dict(self.service.graph.get_state(run_config(run_id)).values)

    def pending_review(self, run_id: str) -> HumanReview | None:
        self.get(run_id)
        return get_pending_review(self.service.graph, run_id)

    def next_nodes(self, run_id: str) -> tuple[str, ...]:
        self.get(run_id)
        return next_nodes(self.service.graph, run_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --- internals ------------------------------------------------------------

    def _claim(self, run_id: str, required: RunPhase, message: str) -> LiveRun:
        """Atomically move a paused run back to running, so an action cannot run twice."""
        run = self.get(run_id)
        with self._lock:
            if run.phase != required:
                raise RunConflictError(f"{message} (current status: {run.phase})")
            run.phase = "running"
            run.error = None
            run.segment_started_at = run.updated_at = _utcnow()
        return run

    def _submit(self, run: LiveRun, segment: str, call: Callable[[NodeCallback], RunOutcome]) -> None:
        run.segment_started_at = run.updated_at = _utcnow()
        # The worker inherits the caller's context (request id, Streamlit session).
        ctx = contextvars.copy_context()
        self._executor.submit(ctx.run, self._run_segment, run, segment, call)

    def _run_segment(self, run: LiveRun, segment: str, call: Callable[[NodeCallback], RunOutcome]) -> None:
        started = time.perf_counter()
        with context.bind(run_id=run.run_id, project=run.project, segment=segment, on_trace=run.traces.append):
            logger.info("Segment %s started", segment)
            try:
                outcome = call(lambda node: self._record(run, node))
            except Exception as exc:
                logger.exception("Segment %s crashed after %.0f ms", segment, (time.perf_counter() - started) * 1000)
                with self._lock:
                    run.phase = "error"
                    run.error = f"{type(exc).__name__}: {exc}"
                    run.updated_at = _utcnow()
                return
            with self._lock:
                run.phase = outcome.status  # type: ignore[assignment]
                run.updated_at = _utcnow()
            logger.info(
                "Segment %s finished in %.0f ms: status=%s%s",
                segment,
                (time.perf_counter() - started) * 1000,
                outcome.status,
                f" outcome={outcome.final_report.status}" if outcome.final_report else "",
            )

    def _record(self, run: LiveRun, node: str) -> None:
        now = _utcnow()
        with self._lock:
            run.events.append(NodeEvent(node=node, started_at=run.current_node_started_at, finished_at=now))
            run.updated_at = now
