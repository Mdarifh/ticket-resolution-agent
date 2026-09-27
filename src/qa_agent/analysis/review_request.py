"""Human review requests: what the agent proposes to do and why a human must approve.

Pure functions (no LangGraph): used by the Confidence Checker node.
"""

from datetime import UTC, datetime
from typing import get_args

from qa_agent.domain import BugReport, ConfidenceScore, EvidenceItem, RootCauseAnalysis
from qa_agent.domain.review import HumanReview, ProposedAction

KNOWN_ACTIONS: frozenset[str] = frozenset(get_args(ProposedAction))
MAX_REVIEW_EVIDENCE = 10
_HIGH = {"high", "critical"}
_RECLASSIFIED = {"test_issue", "requirement_issue"}


def parse_sensitive_actions(value: str) -> set[str]:
    actions = {a.strip() for a in value.split(",") if a.strip()}
    unknown = actions - KNOWN_ACTIONS
    if unknown:
        raise ValueError(f"Unknown sensitive action(s) {sorted(unknown)}; known: {sorted(KNOWN_ACTIONS)}")
    return actions


def proposed_actions(bug_reports: list[BugReport]) -> dict[str, list[str]]:
    """Actions the agent intends to take, each with the bug report ids that cause it."""
    actions: dict[str, list[str]] = {}
    if bug_reports:
        actions["report_bugs"] = [r.id for r in bug_reports]
    high = [r.id for r in bug_reports if r.severity in _HIGH]
    if high:
        actions["report_high_severity_bugs"] = high
    reclassified = [r.id for r in bug_reports if r.report_type in _RECLASSIFIED]
    if reclassified:
        actions["reclassify_failure_as_test_issue"] = reclassified
    return actions


def build_review_request(
    *,
    review_id: str,
    score: ConfidenceScore,
    actions: dict[str, list[str]],
    sensitive: list[str],
    bug_reports: list[BugReport],
    root_cause: RootCauseAnalysis | None,
) -> HumanReview:
    return HumanReview(
        review_id=review_id,
        reason="; ".join(score.reasons),
        reasons=list(score.reasons),
        ai_recommendation=_recommendation(bug_reports, root_cause, score),
        evidence=_key_evidence(bug_reports),
        confidence=score.score,
        proposed_action=_describe_actions(actions, bug_reports),
        actions=list(actions),
        sensitive_actions=sensitive,
        requested_at=datetime.now(UTC),
    )


def _recommendation(
    bug_reports: list[BugReport], root_cause: RootCauseAnalysis | None, score: ConfidenceScore
) -> str:
    if not bug_reports:
        base = "No bug reports were produced"
    else:
        items = "; ".join(
            f"{r.id} [{r.report_type.replace('_', ' ')}, {r.severity}/{r.priority}] {r.title}" for r in bug_reports
        )
        base = f"Report {len(bug_reports)} issue(s): {items}"
    if root_cause is None:
        return f"{base}. No root cause analysis is available."
    return f"{base}. Root cause analysis (confidence {score.score:.2f}, unconfirmed): {root_cause.summary}"


def _describe_actions(actions: dict[str, list[str]], bug_reports: list[BugReport]) -> str:
    if not actions:
        return "Finalize the QA report without bug reports"
    parts = []
    if "report_bugs" in actions:
        parts.append(f"include {len(bug_reports)} bug report(s) in the final QA report")
    if "report_high_severity_bugs" in actions:
        parts.append(f"flag {', '.join(actions['report_high_severity_bugs'])} as high/critical severity")
    if "reclassify_failure_as_test_issue" in actions:
        parts.append(
            f"treat {', '.join(actions['reclassify_failure_as_test_issue'])} as a test or requirement "
            "issue rather than a product bug"
        )
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]


def _key_evidence(bug_reports: list[BugReport]) -> list[EvidenceItem]:
    seen, observed, knowledge = set(), [], []
    for report in bug_reports:
        for item in report.evidence:
            if item.reference in seen:
                continue
            seen.add(item.reference)
            (observed if item.kind == "observed" else knowledge).append(item)
    return (observed + knowledge)[:MAX_REVIEW_EVIDENCE]
