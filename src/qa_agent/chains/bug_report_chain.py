"""Failed tests + failure analysis + RCA + requirement + RAG -> ``BugReport``s.

One report per root cause finding. The split is deliberate:

* Deterministic (never written by the LLM): environment, preconditions,
  reproduction steps, expected and actual results, evidence (recorded facts
  verbatim plus knowledge base references the RCA validated), probable root
  cause with its likelihood, affected component, confidence, uncertainties,
  and the report type implied by the suspected origin.
* LLM narrative: title, summary, severity, priority (with rationales) and
  related past bugs; the latter must be previous-bug documents that were
  actually retrieved.

If the LLM fails, or skips a finding, a plain deterministic narrative is used
instead and flagged (``generated_by="fallback"``), so a report is never lost.
"""

import json
import logging
from datetime import UTC, datetime
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import (
    BugEnvironment,
    BugReport,
    EvidenceItem,
    ExecutionResult,
    FailureDetail,
    KnowledgeSearchResult,
    RequirementAnalysis,
    RootCauseAnalysis,
    RootCauseFinding,
    TestCase,
)
from qa_agent.domain.bug_report import BugPriority, ReportType, Severity
from qa_agent.rag.context import REFERENCE_RULES, format_reference_material

logger = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.5
REPORT_TYPES: dict[str, ReportType] = {
    "product_defect": "product_bug",
    "test_defect": "test_issue",
    "requirement_mismatch": "requirement_issue",
    "environment": "environment_issue",
    "unknown": "needs_investigation",
}
DEFAULT_PRIORITY: dict[str, BugPriority] = {"critical": "p0", "high": "p1", "medium": "p2", "low": "p3"}
ENVIRONMENT_UNKNOWNS = ["Build or version of the system under test was not recorded"]


# --- LLM-facing schema --------------------------------------------------------


class BugNarrative(BaseModel):
    finding_id: str = Field(description="The finding this report describes, e.g. 'RC-1'.")
    title: str = Field(description="Short, factual description of the symptom, not the suspected cause.")
    summary: str = Field(
        description="What was observed and its impact. Mention the suspected cause only as "
        "unconfirmed ('likely', 'possibly')."
    )
    severity: Severity
    severity_rationale: str
    priority: BugPriority
    priority_rationale: str
    related_bugs: list[str] = Field(
        description="Source paths of past bugs in the reference material that look like the same "
        "problem (possible duplicate or regression); empty if none."
    )


class BugNarrativeDraft(BaseModel):
    reports: list[BugNarrative]


BUG_REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write the narrative parts of bug reports: title, summary, severity, priority and "
            "related past bugs. Facts, evidence, steps and environment are filled in by the "
            "system from recorded data; do not restate or add to them.\n"
            "Rules:\n"
            "- Write one narrative per finding id.\n"
            "- Describe only the observed symptom as fact. The probable root cause is unconfirmed: "
            "if you mention it, hedge it ('likely', 'possibly').\n"
            "- Never introduce facts that are not in the finding or its observed facts.\n"
            "- Severity reflects user and business impact (security, data loss, blocked core "
            "flow = high or critical). Start from the analysis's severity suggestion and explain "
            "any change.\n"
            "- Priority: p0 fix immediately, p1 next release, p2 planned, p3 backlog. If the "
            "report type is not product_bug (a test, requirement or environment issue), rate "
            "priority for fixing that issue.\n"
            "- related_bugs: only past bugs from the reference material that plausibly describe "
            "the same problem, copied exactly by source path.\n"
            "- If reviewer feedback is given, address it.\n\n" + REFERENCE_RULES,
        ),
        (
            "human",
            "Requirement:\n{requirement}\n\n"
            "Findings with their failed tests (JSON):\n{findings}\n\n"
            "Reviewer feedback on the previous version:\n{feedback}\n\n"
            "{reference_material}",
        ),
    ]
)


def build_bug_report_chain(llm: BaseChatModel) -> Runnable[dict, BugNarrativeDraft]:
    return structured_chain(BUG_REPORT_PROMPT, llm, BugNarrativeDraft)


# --- assembly -----------------------------------------------------------------


class BugReportContext(BaseModel):
    """Everything the reports are built from."""

    requirement_text: str
    requirement_analysis: RequirementAnalysis | None = None
    root_cause: RootCauseAnalysis
    failures: list[FailureDetail]
    test_cases: list[TestCase]
    execution_results: list[ExecutionResult] = Field(default_factory=list)
    app_env: str = "local"
    run_id: str | None = None
    reviewer_feedback: list[str] = Field(default_factory=list)
    revision: int = 0


def finding_id(index: int) -> str:
    return f"RC-{index + 1}"


def generate_bug_reports(
    llm: BaseChatModel | None, context: BugReportContext, knowledge: KnowledgeSearchResult
) -> tuple[list[BugReport], list[str], str | None]:
    """Return (reports, discarded related-bug citations, fallback reason or None)."""
    findings = context.root_cause.findings
    if not findings:
        return [], [], None

    narratives: dict[str, BugNarrative] = {}
    fallback_reason = None
    if llm is None:
        fallback_reason = "No LLM available"
    else:
        try:
            draft = build_bug_report_chain(llm).invoke(_prompt_inputs(context, knowledge))
            for narrative in draft.reports:
                narratives.setdefault(narrative.finding_id.strip(), narrative)
        except Exception as exc:  # the facts must still be reported
            logger.warning("Bug report narrative generation failed: %s", exc)
            fallback_reason = f"Narrative generation failed ({type(exc).__name__}: {exc})"

    retrieved_bugs = {r.source for r in knowledge.results if r.metadata.get("doc_type") == "previous_bug"}
    reports, discarded = [], []
    for index, finding in enumerate(findings):
        facts = build_facts(finding, context)
        narrative = narratives.get(finding_id(index))
        if narrative is None:
            reports.append(_assemble(index, finding, facts, _fallback_narrative(index, finding), context, "fallback"))
            continue
        related = [s for s in dict.fromkeys(narrative.related_bugs) if s in retrieved_bugs]
        discarded += [s for s in narrative.related_bugs if s not in retrieved_bugs]
        narrative = narrative.model_copy(update={"related_bugs": related})
        reports.append(_assemble(index, finding, facts, narrative, context, "llm"))
    return reports, list(dict.fromkeys(discarded)), fallback_reason


class ReportFacts(BaseModel):
    environment: BugEnvironment
    preconditions: list[str]
    reproduction_steps: list[str]
    reproduction_source: str
    expected_result: str
    actual_result: str
    evidence: list[EvidenceItem]
    uncertainties: list[str]


def build_facts(finding: RootCauseFinding, context: BugReportContext) -> ReportFacts:
    """Deterministic sections of a report, from recorded data only."""
    test_ids = finding.test_case_ids
    failures = {f.test_case_id: f for f in context.failures}
    cases = {tc.test_case_id: tc for tc in context.test_cases}
    results = {r.test_case_id: r for r in context.execution_results}
    related_failures = [failures[t] for t in test_ids if t in failures]
    related_cases = [cases[t] for t in test_ids if t in cases]
    related_results = [results[t] for t in test_ids if t in results]

    preconditions = list(dict.fromkeys(p for tc in related_cases for p in tc.preconditions))

    steps, source = _executed_ui_steps(related_results), "executed_steps"
    if not steps and finding.reproduction_steps:
        steps, source = list(finding.reproduction_steps), "root_cause_analysis"
    if not steps and related_cases:
        steps, source = list(related_cases[0].steps), "test_case"
    for tc in related_cases[:1]:
        if tc.test_data:
            steps.append("Test data: " + ", ".join(f"{d.name}={d.value}" for d in tc.test_data))

    expected = _per_test(test_ids, {tc.test_case_id: tc.expected_result for tc in related_cases}) or finding.expected_behavior
    actual = _per_test(
        test_ids, {f.test_case_id: f.actual_result or f.message or "" for f in related_failures}
    ) or finding.observed_behavior

    interpretations = {e.reference: e.interpretation for e in finding.evidence}
    evidence = [
        EvidenceItem(kind="observed", reference=fact.id, detail=fact.detail, interpretation=interpretations.get(fact.id))
        for failure in related_failures
        for fact in failure.observed_facts
    ]
    evidence += [e for e in finding.evidence if e.kind == "knowledge_base"]

    cause = finding.probable_root_cause
    uncertainties = [f"Probable root cause is not confirmed (assessed as {cause.likelihood})"]
    uncertainties += finding.unknowns
    if finding.confidence < LOW_CONFIDENCE:
        uncertainties.append(
            f"Root cause analysis confidence is low ({finding.confidence:.2f}); verify before acting"
        )
    if finding.affected_component == "unknown":
        uncertainties.append("Affected component could not be determined")

    return ReportFacts(
        environment=_environment(related_results, context),
        preconditions=preconditions,
        reproduction_steps=steps,
        reproduction_source=source,
        expected_result=expected,
        actual_result=actual,
        evidence=evidence,
        uncertainties=list(dict.fromkeys(uncertainties)),
    )


def _assemble(
    index: int,
    finding: RootCauseFinding,
    facts: ReportFacts,
    narrative: BugNarrative,
    context: BugReportContext,
    generated_by: str,
) -> BugReport:
    uncertainties = list(facts.uncertainties)
    if generated_by == "fallback":
        uncertainties.append("Title, summary, severity and priority were generated without the LLM")
    return BugReport(
        id=f"BR-{index + 1}",
        title=narrative.title,
        summary=narrative.summary,
        report_type=REPORT_TYPES[finding.suspected_origin],
        environment=facts.environment,
        preconditions=facts.preconditions,
        reproduction_steps=facts.reproduction_steps,
        reproduction_source=facts.reproduction_source,
        expected_result=facts.expected_result,
        actual_result=facts.actual_result,
        evidence=facts.evidence,
        probable_root_cause=finding.probable_root_cause,
        severity=narrative.severity,
        severity_rationale=narrative.severity_rationale,
        priority=narrative.priority,
        priority_rationale=narrative.priority_rationale,
        affected_component=finding.affected_component,
        confidence=finding.confidence,
        related_test_case_ids=finding.test_case_ids,
        related_bugs=narrative.related_bugs,
        uncertainties=uncertainties,
        generated_by=generated_by,
        revision=context.revision,
        reviewer_feedback=context.reviewer_feedback,
    )


def _fallback_narrative(index: int, finding: RootCauseFinding) -> BugNarrative:
    tests = ", ".join(finding.test_case_ids)
    return BugNarrative(
        finding_id=finding_id(index),
        title=f"{tests}: {finding.summary}",
        summary=f"{finding.observed_behavior} Probable cause ({finding.probable_root_cause.likelihood}, "
        f"unconfirmed): {finding.probable_root_cause.description}",
        severity=finding.severity_suggestion,
        severity_rationale=f"Taken from the root cause analysis: {finding.severity_rationale}",
        priority=DEFAULT_PRIORITY[finding.severity_suggestion],
        priority_rationale="Derived from severity",
        related_bugs=[],
    )


def _prompt_inputs(context: BugReportContext, knowledge: KnowledgeSearchResult) -> dict:
    failures = {f.test_case_id: f for f in context.failures}
    findings = []
    for index, finding in enumerate(context.root_cause.findings):
        findings.append(
            {
                "finding_id": finding_id(index),
                "report_type": REPORT_TYPES[finding.suspected_origin],
                "finding": finding.model_dump(mode="json", exclude={"adjustments"}),
                "observed_facts": [
                    fact.model_dump() for t in finding.test_case_ids if t in failures for fact in failures[t].observed_facts
                ],
            }
        )
    requirement = context.requirement_text
    if context.requirement_analysis:
        requirement += "\n\nAcceptance criteria:\n" + "\n".join(
            f"- {c}" for c in context.requirement_analysis.acceptance_criteria
        )
    return {
        "requirement": requirement,
        "findings": json.dumps(findings, indent=2, ensure_ascii=False),
        "feedback": "\n".join(f"- {f}" for f in context.reviewer_feedback) or "none",
        "reference_material": format_reference_material(knowledge),
    }


def _executed_ui_steps(results: list[ExecutionResult]) -> list[str]:
    """For UI tests, the steps the browser actually ran are the most exact reproduction."""
    for result in results:
        executed = result.evidence.get("executed_steps") if result.evidence else None
        if not executed:
            continue
        steps = []
        for step in executed:
            text = f"{step['action']} {step['target']}"
            if step.get("value") is not None:
                text += f" with value {step['value']!r}"
            if step.get("status") == "failed":
                text += "  <- fails here"
            steps.append(text)
        return steps
    return []


def _environment(results: list[ExecutionResult], context: BugReportContext) -> BugEnvironment:
    targets, kinds = [], []
    for result in results:
        evidence = result.evidence or {}
        kinds.append(evidence.get("automation_type"))
        for url in (evidence.get("url"), (evidence.get("diagnostics") or {}).get("page_url")):
            if url:
                parts = urlsplit(url)
                if parts.scheme in ("http", "https") and parts.netloc:
                    targets.append(f"{parts.scheme}://{parts.netloc}")
    unknowns = list(ENVIRONMENT_UNKNOWNS)
    if not targets:
        unknowns.append("Target URL of the system under test was not recorded")
    return BugEnvironment(
        app_env=context.app_env,
        run_id=context.run_id,
        targets=list(dict.fromkeys(targets)),
        automation_types=[k for k in dict.fromkeys(kinds) if k],
        recorded_at=datetime.now(UTC),
        unknowns=unknowns,
    )


def _per_test(test_ids: list[str], values: dict[str, str]) -> str:
    present = [(t, values[t]) for t in test_ids if values.get(t)]
    if len(present) == 1:
        return present[0][1]
    return "; ".join(f"{t}: {v}" for t, v in present)


def render_bug_report_markdown(report: BugReport) -> str:
    cause = report.probable_root_cause
    lines = [
        f"### {report.id}: {report.title}",
        f"*{report.report_type.replace('_', ' ')} | severity {report.severity} | priority {report.priority} "
        f"| confidence {report.confidence:.2f} | tests {', '.join(report.related_test_case_ids)}*",
        "",
        report.summary,
        "",
        f"- **Expected:** {report.expected_result}",
        f"- **Actual (observed):** {report.actual_result}",
        f"- **Probable root cause (UNCONFIRMED, {cause.likelihood}):** {cause.description}",
        f"- **Affected component:** {report.affected_component}",
        "- **Reproduction steps:**",
        *(f"  {i}. {s}" for i, s in enumerate(report.reproduction_steps, start=1)),
        "- **Evidence:**",
        *(f"  - [{e.kind}] {e.reference}: {e.detail}" for e in report.evidence),
        "- **Uncertain / unknown:**",
        *(f"  - {u}" for u in report.uncertainties),
    ]
    if report.related_bugs:
        lines += ["- **Possibly related past bugs:**", *(f"  - {b}" for b in report.related_bugs)]
    return "\n".join(lines)
