"""Human Review node: pauses the graph with LangGraph's ``interrupt()``.

The run is checkpointed (thread_id = run_id) and control returns to the
caller with the review request. It resumes with
``graph.invoke(Command(resume={"decision": "APPROVE" | "REJECT" |
"REQUEST_REANALYSIS", "comment": ..., "reviewer": ...}), config)`` (or
``qa_agent.graph.review.submit_review_decision``); the resume value becomes
the return value of ``interrupt()`` below. The decided review is appended to
``review_history`` for the audit trail.
"""

from datetime import UTC, datetime
from typing import Any

from langgraph.types import interrupt

from qa_agent.domain.review import HumanReview, HumanReviewDecision
from qa_agent.graph.state import QAAgentState


def human_review(state: QAAgentState) -> dict[str, Any]:
    review = state.get("human_review") or _fallback_request(state)
    bug_reports = state.get("bug_reports") or []

    raw_decision = interrupt(
        {
            "review": review.model_dump(mode="json"),
            "bug_reports": [report.model_dump(mode="json") for report in bug_reports],
        }
    )
    decision = HumanReviewDecision.model_validate(raw_decision)

    decided = review.model_copy(
        update={
            "human_decision": decision.decision,
            "reviewer_comment": decision.comment,
            "reviewer": decision.reviewer,
            "decided_at": datetime.now(UTC),
        }
    )
    return {"human_review": decided, "review_history": [decided]}


def _fallback_request(state: QAAgentState) -> HumanReview:
    history = state.get("review_history") or []
    return HumanReview(
        review_id=f"{state['requirement'].id}-review-{len(history) + 1}",
        reason="Review requested without a confidence check",
        ai_recommendation="No recommendation available",
        confidence=0.0,
        proposed_action="Finalize the QA report",
        requested_at=datetime.now(UTC),
    )
