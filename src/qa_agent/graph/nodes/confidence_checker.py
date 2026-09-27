"""Confidence Checker node: decides whether a human must approve before the run finalizes.

A review is required when
1. the root cause analysis confidence is below ``CONFIDENCE_THRESHOLD``, or
2. the agent proposes an action listed in ``SENSITIVE_ACTIONS``.

The node only records the decision (``confidence_score.requires_human_review``)
and the review request; the graph's conditional edge does the routing.
"""

from typing import Any

from qa_agent.analysis.review_request import build_review_request, parse_sensitive_actions, proposed_actions
from qa_agent.config import get_settings
from qa_agent.domain import ConfidenceScore
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.state import QAAgentState


def make_confidence_checker(
    threshold: float | None = None, sensitive_actions: set[str] | None = None
) -> NodeFn:
    """Defaults come from settings (``CONFIDENCE_THRESHOLD``, ``SENSITIVE_ACTIONS``) at call time."""

    def confidence_checker(state: QAAgentState) -> dict[str, Any]:
        settings = get_settings()
        limit = threshold if threshold is not None else settings.confidence_threshold
        sensitive_config = (
            sensitive_actions if sensitive_actions is not None else parse_sensitive_actions(settings.sensitive_actions)
        )
        rca = state.get("root_cause_analysis")
        bug_reports = state.get("bug_reports") or []

        score = rca.confidence if rca else 0.0
        reasons: list[str] = []
        if rca is None:
            reasons.append("No root cause analysis available")
        if score < limit:
            reasons.append(f"Confidence {score:.2f} is below threshold {limit:.2f}")

        actions = proposed_actions(bug_reports)
        sensitive = [action for action in actions if action in sensitive_config]
        for action in sensitive:
            reasons.append(f"Sensitive action requires approval: {action} ({', '.join(actions[action])})")

        confidence = ConfidenceScore(
            score=score, threshold=limit, requires_human_review=bool(reasons), reasons=reasons
        )
        if not reasons:
            return {"confidence_score": confidence, "human_review": None}

        history = state.get("review_history") or []
        review = build_review_request(
            review_id=f"{state['requirement'].id}-review-{len(history) + 1}",
            score=confidence,
            actions=actions,
            sensitive=sensitive,
            bug_reports=bug_reports,
            root_cause=rca,
        )
        return {"confidence_score": confidence, "human_review": review}

    return confidence_checker


confidence_checker = make_confidence_checker()
