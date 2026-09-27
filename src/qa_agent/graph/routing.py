"""Conditional-edge routers.

These only read state and name the next node; LangGraph evaluates them via
``add_conditional_edges`` and performs the actual transition.
"""

from typing import Literal

from qa_agent.graph.state import QAAgentState

BUG_REPORT_GENERATOR = "bug_report_generator"
ROOT_CAUSE_ANALYZER = "root_cause_analyzer"
FAILURE_ANALYZER = "failure_analyzer"
FINAL_REPORT_GENERATOR = "final_report_generator"
HUMAN_REVIEW = "human_review"


def route_after_result_analysis(
    state: QAAgentState,
) -> Literal["failure_analyzer", "final_report_generator"]:
    """Any failed/errored test goes down the failure path; otherwise straight to the report."""
    return FAILURE_ANALYZER if state.get("failures") else FINAL_REPORT_GENERATOR


def route_after_confidence_check(
    state: QAAgentState,
) -> Literal["human_review", "final_report_generator"]:
    """Low confidence (or another review trigger) goes to human review."""
    confidence = state.get("confidence_score")
    if confidence is None or confidence.requires_human_review:
        return HUMAN_REVIEW
    return FINAL_REPORT_GENERATOR


def route_after_human_review(
    state: QAAgentState,
) -> Literal["root_cause_analyzer", "final_report_generator"]:
    """REQUEST_REANALYSIS re-runs root cause analysis (then bug reports and the confidence
    check); APPROVE and REJECT go to the final report."""
    review = state.get("human_review")
    if review is not None and review.human_decision == "REQUEST_REANALYSIS":
        return ROOT_CAUSE_ANALYZER
    return FINAL_REPORT_GENERATOR
