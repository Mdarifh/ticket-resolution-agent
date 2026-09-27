"""Final Report Generator node: aggregates the run into a ``FinalReport``.

Deterministic for now; an LLM-written executive summary is added later.
"""

from datetime import UTC, datetime
from typing import Any

from qa_agent.chains.bug_report_chain import render_bug_report_markdown
from qa_agent.chains.root_cause_chain import render_root_cause_markdown
from qa_agent.domain import BugReport, FinalReport, HumanReview, RootCauseAnalysis, WorkflowError
from qa_agent.domain.qa_report import ReportStatus
from qa_agent.graph.state import QAAgentState


def final_report_generator(state: QAAgentState) -> dict[str, Any]:
    results = state.get("execution_results") or []
    failures = state.get("failures") or []
    errors = state.get("errors") or []
    review = state.get("human_review")
    decision = review.human_decision if review else None

    executed = [r for r in results if r.status != "skipped"]
    total = len(executed)
    skipped = len(results) - total
    passed = sum(1 for r in executed if r.status == "passed")
    failed = sum(1 for r in executed if r.status == "failed")
    errored = sum(1 for r in executed if r.status == "error")
    current_errors = unrecovered_errors(errors, state)

    # Most decisive first: a human rejection, then real test failures, then anything
    # that prevents a verdict (unrecovered workflow errors, tests that could not run).
    status: ReportStatus
    if decision == "REJECT":
        status = "rejected"
    elif failed:
        status = "failed"
    elif current_errors or errored:
        status = "error"
    elif total == 0:
        status = "inconclusive"
    else:
        status = "passed"

    summary = f"{status.upper()}: {passed}/{total} tests passed"
    if errored:
        summary += f", {errored} could not complete"
    if skipped:
        summary += f", {skipped} skipped"
    if current_errors:
        summary += f", {len(current_errors)} workflow error(s)"
    if len(errors) > len(current_errors):
        summary += f", {len(errors) - len(current_errors)} recovered"
    if decision:
        summary += f", human decision: {decision}"

    report = FinalReport(
        status=status,
        summary=summary,
        total_tests=total,
        passed=passed,
        failed=failed,
        errored=errored,
        skipped=skipped,
        pass_rate=passed / total if total else 0.0,
        bug_reports=(state.get("bug_reports") or []) if failures else [],
        confidence_score=state.get("confidence_score") if failures else None,
        human_decision=decision,
        human_review=review if decision else None,
        error_count=len(current_errors),
        markdown=_render_markdown(
            state["requirement"].text,
            summary,
            failures,
            errors,
            current_errors,
            state.get("root_cause_analysis"),
            (state.get("bug_reports") or []) if failures else [],
            state.get("review_history") or [],
        ),
    )
    return {
        "final_report": report,
        "execution_metadata": {"status": "completed", "completed_at": datetime.now(UTC)},
    }


def unrecovered_errors(errors: list[WorkflowError], state: QAAgentState) -> list[WorkflowError]:
    """Errors whose node did not run again afterwards (a later visit supersedes the failed one)."""
    metadata = state.get("execution_metadata")
    visited = metadata.nodes_visited if metadata else []
    return [e for e in errors if visited.count(e.node) <= e.visit + 1]


def _render_markdown(
    requirement: str,
    summary: str,
    failures: list,
    errors: list,
    current_errors: list,
    root_cause: RootCauseAnalysis | None,
    bug_reports: list[BugReport],
    reviews: list[HumanReview],
) -> str:
    lines = ["# QA Report", "", f"**Requirement:** {requirement}", "", f"**Result:** {summary}"]
    if failures:
        lines += ["", "## Failures", *(f"- {f.test_case_id}: {f.message}" for f in failures)]
    if failures and root_cause:
        lines += ["", render_root_cause_markdown(root_cause).rstrip()]
    if reviews:
        lines += ["", "## Human review", ""]
        for review in reviews:
            lines.append(
                f"- {review.review_id}: **{review.human_decision}** by {review.reviewer or 'unknown reviewer'}"
                f" ({review.reason})" + (f". Comment: {review.reviewer_comment}" if review.reviewer_comment else "")
            )
    if bug_reports:
        lines += ["", "## Bug reports", ""]
        lines += [render_bug_report_markdown(report) + "\n" for report in bug_reports]
    if errors:
        lines += [
            "",
            "## Workflow errors",
            *(f"- {e.node}: {e.message}" + ("" if e in current_errors else " (recovered on a later attempt)") for e in errors),
        ]
    return "\n".join(lines) + "\n"
