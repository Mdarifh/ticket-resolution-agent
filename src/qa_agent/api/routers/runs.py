"""Run endpoints: start a run, follow its progress, read its artifacts, act on its pauses.

Actions (start, execute, review) return 202 immediately; the workflow segment
runs in the background and clients poll ``GET /runs/{run_id}``.
"""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from langchain_core.language_models import BaseChatModel

from qa_agent.api.dependencies import get_chat_model_factory, get_run_manager
from qa_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    NodeProgress,
    ReviewOut,
    RootCauseOut,
    RunCreate,
    RunDetail,
    RunSummary,
    TestCounts,
    TraceInfo,
)
from qa_agent.domain import (
    BugReport,
    ConfidenceScore,
    ExecutionResult,
    FailureDetail,
    FinalReport,
    HumanReviewDecision,
    RequirementAnalysis,
    TestCase,
    TestExecutionPlan,
)
from qa_agent.chains.run_chat_chain import answer_question, build_run_context
from qa_agent.domain.bug_ticket import BUG_VERIFICATION_MODE, ticket_to_requirement, ticket_verdict
from qa_agent.errors import LLMConfigurationError
from qa_agent.graph import NODE_NAMES
from qa_agent.observability import trace_url, tracing_status
from qa_agent.services.run_manager import LiveRun, RunConflictError, RunManager, RunNotFoundError

router = APIRouter(prefix="/runs", tags=["runs"])

_PAUSED = {"awaiting_execution", "awaiting_review"}


def _state(manager: RunManager, run_id: str) -> dict[str, Any]:
    try:
        return manager.state(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def _summary(manager: RunManager, run: LiveRun, state: dict[str, Any] | None = None) -> RunSummary:
    state = manager.state(run.run_id) if state is None else state
    results = state.get("execution_results") or []
    statuses = [r.status for r in results]
    report = state.get("final_report")
    confidence = state.get("confidence_score")
    cases = state.get("test_cases") or []
    return RunSummary(
        run_id=run.run_id,
        project=run.project,
        requirement=run.requirement,
        status=run.phase,
        outcome=report.status if report else None,
        current_nodes=list(manager.next_nodes(run.run_id)) if run.phase != "completed" else [],
        error=run.error,
        created_at=run.created_at,
        updated_at=run.updated_at,
        pause_before_execution=run.pause_before_execution,
        metadata=run.metadata,
        test_case_count=len(cases),
        test_counts=TestCounts(
            total=len(statuses),
            passed=statuses.count("passed"),
            failed=statuses.count("failed"),
            error=statuses.count("error"),
            skipped=statuses.count("skipped"),
        ),
        confidence=confidence.score if confidence else None,
        bug_count=len(state.get("bug_reports") or []),
        node_error_count=len(state.get("errors") or []),
        verdict=(
            ticket_verdict(
                run.phase,
                results,
                report.human_decision if report else None,
                reproduction_test=cases[0].test_case_id if cases else None,
            )
            if run.metadata.get("mode") == BUG_VERIFICATION_MODE
            else None
        ),
    )


def _node_progress(run: LiveRun, state: dict[str, Any], current: list[str]) -> list[NodeProgress]:
    failed_nodes = {error.node for error in state.get("errors") or []}
    progress = []
    for name in NODE_NAMES:
        events = [e for e in run.events if e.node == name]
        last = events[-1] if events else None
        node = NodeProgress(
            name=name,
            status="pending",
            visits=len(events),
            started_at=last.started_at if last else None,
            finished_at=last.finished_at if last else None,
            duration_ms=(
                sum(int((e.finished_at - e.started_at).total_seconds() * 1000) for e in events) if events else None
            ),
        )
        if name in current and run.phase == "running":
            node.status = "running"
            node.started_at, node.finished_at = run.current_node_started_at, None
        elif name in current and run.phase in _PAUSED:
            node.status = "waiting"
        elif name in current and run.phase == "error":
            node.status = "error"
        elif events:
            node.status = "error" if name in failed_nodes else "completed"
        elif run.phase == "completed":
            node.status = "skipped"
        progress.append(node)
    return progress


def _run(manager: RunManager, run_id: str) -> LiveRun:
    try:
        return manager.get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=RunSummary)
def start_run(body: RunCreate, response: Response, manager: RunManager = Depends(get_run_manager)) -> RunSummary:
    requirement, metadata = body.requirement, dict(body.metadata)
    if body.bug_ticket is not None:
        requirement = ticket_to_requirement(body.bug_ticket)
        metadata |= {"mode": BUG_VERIFICATION_MODE, "bug_ticket": body.bug_ticket.model_dump()}
        if body.bug_ticket.ticket_id:
            metadata.setdefault("ticket", body.bug_ticket.ticket_id)
    run = manager.start(
        requirement,
        project=body.project.strip() or "default",
        metadata=metadata,
        pause_before_execution=body.pause_before_execution,
    )
    response.headers["X-QA-Run-ID"] = run.run_id
    return _summary(manager, run)


@router.get("", response_model=list[RunSummary])
def list_runs(project: str | None = None, manager: RunManager = Depends(get_run_manager)) -> list[RunSummary]:
    return [_summary(manager, run) for run in manager.list(project)]


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, manager: RunManager = Depends(get_run_manager)) -> RunDetail:
    run = _run(manager, run_id)
    state = _state(manager, run_id)
    summary = _summary(manager, run, state)
    return RunDetail(
        **summary.model_dump(),
        nodes=_node_progress(run, state, summary.current_nodes),
        errors=state.get("errors") or [],
        traces=[TraceInfo(segment=t.segment, trace_id=t.trace_id, started_at=t.started_at) for t in run.traces],
        langsmith_tracing=tracing_status().enabled,
        langsmith_project=tracing_status().project if tracing_status().enabled else None,
    )


@router.get("/{run_id}/traces", response_model=list[TraceInfo])
def get_traces(run_id: str, manager: RunManager = Depends(get_run_manager)) -> list[TraceInfo]:
    """The run's LangSmith traces with links (resolving links may call the LangSmith API)."""
    run = _run(manager, run_id)
    return [
        TraceInfo(segment=t.segment, trace_id=t.trace_id, started_at=t.started_at, url=trace_url(t.trace_id))
        for t in run.traces
    ]


@router.get("/{run_id}/requirement-analysis", response_model=RequirementAnalysis | None)
def get_requirement_analysis(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("requirement_analysis")


@router.get("/{run_id}/test-cases", response_model=list[TestCase])
def get_test_cases(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("test_cases") or []


@router.get("/{run_id}/test-plan", response_model=TestExecutionPlan | None)
def get_test_plan(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("test_plan")


@router.post("/{run_id}/execute", status_code=status.HTTP_202_ACCEPTED, response_model=RunSummary)
def execute_tests(run_id: str, manager: RunManager = Depends(get_run_manager)) -> RunSummary:
    _run(manager, run_id)
    try:
        run = manager.execute(run_id)
    except RunConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _summary(manager, run)


@router.get("/{run_id}/results", response_model=list[ExecutionResult])
def get_results(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("execution_results") or []


@router.get("/{run_id}/failures", response_model=list[FailureDetail])
def get_failures(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("failures") or []


@router.get("/{run_id}/root-cause", response_model=RootCauseOut)
def get_root_cause(run_id: str, manager: RunManager = Depends(get_run_manager)) -> RootCauseOut:
    state = _state(manager, run_id)
    return RootCauseOut(
        analysis=state.get("root_cause_analysis"),
        knowledge_usage=state.get("retrieved_knowledge") or [],
    )


@router.get("/{run_id}/bugs", response_model=list[BugReport])
def get_bug_reports(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("bug_reports") or []


@router.get("/{run_id}/confidence", response_model=ConfidenceScore | None)
def get_confidence(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("confidence_score")


@router.get("/{run_id}/review", response_model=ReviewOut)
def get_review(run_id: str, manager: RunManager = Depends(get_run_manager)) -> ReviewOut:
    state = _state(manager, run_id)
    return ReviewOut(
        pending=manager.pending_review(run_id),
        latest=state.get("human_review"),
        history=state.get("review_history") or [],
    )


@router.post("/{run_id}/review", status_code=status.HTTP_202_ACCEPTED, response_model=RunSummary)
def submit_review(
    run_id: str, decision: HumanReviewDecision, manager: RunManager = Depends(get_run_manager)
) -> RunSummary:
    _run(manager, run_id)
    try:
        run = manager.decide(run_id, decision)
    except RunConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _summary(manager, run)


@router.get("/{run_id}/report", response_model=FinalReport | None)
def get_report(run_id: str, manager: RunManager = Depends(get_run_manager)):
    return _state(manager, run_id).get("final_report")


@router.get("/{run_id}/report.md", response_class=Response)
def download_report(run_id: str, manager: RunManager = Depends(get_run_manager)) -> Response:
    report = _state(manager, run_id).get("final_report")
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The final report is not ready yet")
    return Response(
        report.markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="qa-report-{run_id[:8]}.md"'},
    )


@router.post("/{run_id}/chat", response_model=ChatResponse)
def chat_about_run(
    run_id: str,
    body: ChatRequest,
    manager: RunManager = Depends(get_run_manager),
    llm_factory: Callable[[], BaseChatModel] = Depends(get_chat_model_factory),
) -> ChatResponse:
    """Answer a question about the run from its workflow state (LLM-backed)."""
    run = _run(manager, run_id)
    state = _state(manager, run_id)
    summary = _summary(manager, run, state).model_dump(mode="json")
    try:
        llm = llm_factory()
    except LLMConfigurationError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    try:
        answer = answer_question(
            llm,
            build_run_context(summary, state),
            body.message,
            [turn.model_dump() for turn in body.history],
        )
    except Exception as exc:  # provider outages (429/5xx) should not surface as 500s
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"The LLM call failed: {exc}") from exc
    return ChatResponse(answer=answer or "(empty answer)")
