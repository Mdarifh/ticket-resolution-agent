"""Failures (+ requirement, test cases, retrieved QA knowledge) -> ``RootCauseAnalysis``.

The LLM drafts one finding per failed test. ``finalize_root_cause`` then
enforces, in code, that speculation cannot pass as fact:

* Evidence must reference an observed fact id produced by the Failure
  Analyzer or a knowledge base source that was actually retrieved; the
  stored evidence text is the recorded fact, not the LLM's paraphrase.
  Anything else is dropped and noted.
* A probable cause with no supporting observed fact becomes "speculative"
  and its confidence is capped at 0.3; one supported only by knowledge base
  documents is at most "possible" with confidence capped at 0.5.
* Findings may only name tests that failed; every failed test gets a
  finding (a placeholder saying "not analyzed" when the LLM skipped it).
* Overall confidence is the lowest finding confidence.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import (
    EvidenceItem,
    FailureDetail,
    KnowledgeSearchResult,
    ProbableCause,
    Requirement,
    RequirementAnalysis,
    RootCauseAnalysis,
    RootCauseFinding,
    TestCase,
)
from qa_agent.domain.analysis import FailureOrigin, Likelihood
from qa_agent.domain.bug_report import Severity
from qa_agent.rag.context import REFERENCE_RULES, format_reference_material

SPECULATIVE_CONFIDENCE_CAP = 0.3
KNOWLEDGE_ONLY_CONFIDENCE_CAP = 0.5


# --- LLM-facing schema --------------------------------------------------------


class CauseDraft(BaseModel):
    description: str = Field(description="The hypothesis, phrased as a hypothesis, not a fact.")
    reasoning: str = Field(description="Why the cited evidence points to this cause.")
    likelihood: Likelihood
    supporting_evidence: list[str] = Field(
        description="Observed fact ids (e.g. 'TC-007:F1') and/or knowledge base source paths."
    )


class EvidenceDraft(BaseModel):
    reference: str = Field(description="An observed fact id or a knowledge base source path.")
    interpretation: str = Field(description="What this evidence shows about the failure.")


class FindingDraft(BaseModel):
    test_case_ids: list[str]
    summary: str
    observed_behavior: str = Field(description="Only what the observed facts show.")
    expected_behavior: str = Field(description="From the test case and the requirement.")
    suspected_origin: FailureOrigin
    probable_root_cause: CauseDraft
    alternative_causes: list[CauseDraft]
    evidence: list[EvidenceDraft]
    unknowns: list[str] = Field(description="What the available data cannot tell.")
    affected_component: str = Field(description="Component or 'unknown'.")
    severity_suggestion: Severity
    severity_rationale: str
    reproduction_steps: list[str]
    recommended_next_investigation: list[str]
    confidence: float = Field(description="0 to 1: how well the evidence supports the probable cause.")


class RootCauseDraft(BaseModel):
    summary: str
    findings: list[FindingDraft]


ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a QA engineer performing root cause analysis of failed tests. Write one "
            "finding per failed test (you may group tests only if they clearly share a cause).\n"
            "Keep three kinds of information strictly apart:\n"
            "1. Observed behavior: only what the observed facts state. Do not add details.\n"
            "2. Probable root cause: a hypothesis. Phrase it as one ('likely', 'possibly'), "
            "never as established fact, and cite the fact ids or knowledge base sources that "
            "support it. Offer alternative causes when more than one explanation fits.\n"
            "3. Unknowns: list what the data cannot tell (e.g. server logs not available, "
            "whether the issue is intermittent).\n"
            "Rules:\n"
            "- Evidence references must be observed fact ids exactly as given (e.g. 'TC-007:F2') "
            "or knowledge base source paths exactly as given. Nothing else counts as evidence.\n"
            "- Decide suspected_origin: product_defect (the system violates the requirement), "
            "test_defect (the test itself is wrong), requirement_mismatch (the test's expected "
            "result conflicts with the requirement or the documented contract, or the "
            "requirement is ambiguous about it), environment (infrastructure, network, test "
            "data, timeouts caused by the environment) or unknown.\n"
            "- Compare the test's expected result with the requirement and any documented "
            "contract before blaming the product.\n"
            "- A resembling past bug is a lead, not proof: it cannot raise confidence on its own.\n"
            "- Confidence reflects how directly observed facts support the probable cause; keep "
            "it low when evidence is thin or several causes fit.\n"
            "- Reproduction steps come from the test case steps and data; next investigation "
            "steps must be concrete checks that would confirm or rule out the cause.\n\n"
            + REFERENCE_RULES,
        ),
        (
            "human",
            "Requirement:\n{requirement}\n\n"
            "Requirement analysis (JSON):\n{analysis}\n\n"
            "Failed tests with their test cases and observed facts (JSON):\n{failures}\n\n"
            "Human reviewer feedback on the previous analysis (address it; it does not "
            "count as evidence):\n{reviewer_feedback}\n\n"
            "{reference_material}",
        ),
    ]
)


def build_root_cause_chain(llm: BaseChatModel) -> Runnable[dict, RootCauseDraft]:
    return structured_chain(ROOT_CAUSE_PROMPT, llm, RootCauseDraft)


def analyze_root_cause(
    llm: BaseChatModel,
    requirement: Requirement,
    analysis: RequirementAnalysis | None,
    failures: list[FailureDetail],
    test_cases: list[TestCase],
    knowledge: KnowledgeSearchResult,
    reviewer_feedback: list[str] | None = None,
) -> tuple[RootCauseAnalysis, list[str]]:
    """Return the guarded analysis and the discarded (unretrieved) knowledge citations."""
    by_id = {tc.test_case_id: tc for tc in test_cases}
    payload = [
        {
            "test_case_id": f.test_case_id,
            "status": f.status,
            "category": f.category,
            "test_case": by_id[f.test_case_id].model_dump(mode="json") if f.test_case_id in by_id else None,
            "expected_result": f.expected_result,
            "actual_result": f.actual_result,
            "observed_facts": [fact.model_dump() for fact in f.observed_facts],
        }
        for f in failures
    ]
    draft = build_root_cause_chain(llm).invoke(
        {
            "requirement": requirement.text,
            "analysis": analysis.model_dump_json(indent=2) if analysis else "not available",
            "failures": json.dumps(payload, indent=2, ensure_ascii=False),
            "reviewer_feedback": "\n".join(f"- {f}" for f in reviewer_feedback or []) or "none",
            "reference_material": format_reference_material(knowledge),
        }
    )
    return finalize_root_cause(draft, failures, knowledge)


# --- guards -------------------------------------------------------------------


def finalize_root_cause(
    draft: RootCauseDraft, failures: list[FailureDetail], knowledge: KnowledgeSearchResult
) -> tuple[RootCauseAnalysis, list[str]]:
    facts = {fact.id: fact for f in failures for fact in f.observed_facts}
    sources = {r.source: r for r in knowledge.results}
    failed = {f.test_case_id: f for f in failures}
    discarded_citations: list[str] = []

    def classify(reference: str) -> str | None:
        reference = reference.strip()
        if reference in facts:
            return "observed"
        if reference in sources:
            return "knowledge_base"
        if not _looks_like_fact_id(reference):
            discarded_citations.append(reference)
        return None

    findings: list[RootCauseFinding] = []
    covered: set[str] = set()
    for item in draft.findings:
        adjustments: list[str] = []
        test_ids = [t for t in dict.fromkeys(item.test_case_ids) if t in failed]
        for dropped in set(item.test_case_ids) - set(test_ids):
            adjustments.append(f"Removed test {dropped}: it did not fail in this run")
        if not test_ids:
            continue
        covered.update(test_ids)

        evidence: list[EvidenceItem] = []
        for ref in item.evidence:
            kind = classify(ref.reference)
            if kind is None:
                adjustments.append(f"Dropped evidence {ref.reference!r}: not an observed fact or retrieved source")
            elif kind == "observed":
                fact = facts[ref.reference.strip()]
                evidence.append(EvidenceItem(kind="observed", reference=fact.id, detail=fact.detail, interpretation=ref.interpretation))
            else:
                doc = sources[ref.reference.strip()]
                evidence.append(
                    EvidenceItem(kind="knowledge_base", reference=doc.source, detail=str(doc.metadata.get("title", doc.source)), interpretation=ref.interpretation)
                )

        cause, confidence, cause_notes = _guard_cause(item.probable_root_cause, item.confidence, classify)
        adjustments.extend(cause_notes)
        alternatives = [_guard_cause(alt, 1.0, classify)[0] for alt in item.alternative_causes]

        unknowns = list(item.unknowns)
        if not any(ref in facts for ref in cause.supporting_evidence):
            unknowns.append("No observed evidence directly supports the probable cause")

        categories = {failed[t].category for t in test_ids}
        findings.append(
            RootCauseFinding(
                test_case_ids=test_ids,
                failure_category=categories.pop() if len(categories) == 1 else None,
                summary=item.summary,
                observed_behavior=item.observed_behavior,
                expected_behavior=item.expected_behavior,
                suspected_origin=item.suspected_origin,
                probable_root_cause=cause,
                alternative_causes=alternatives,
                evidence=evidence,
                unknowns=unknowns,
                affected_component=item.affected_component.strip() or "unknown",
                severity_suggestion=item.severity_suggestion,
                severity_rationale=item.severity_rationale,
                reproduction_steps=item.reproduction_steps,
                recommended_next_investigation=item.recommended_next_investigation,
                confidence=confidence,
                adjustments=adjustments,
            )
        )

    for test_id, failure in failed.items():
        if test_id not in covered:
            findings.append(_unanalyzed_finding(failure))

    overall = min((f.confidence for f in findings), default=0.0)
    analysis = RootCauseAnalysis(summary=draft.summary, findings=findings, confidence=overall)
    return analysis, list(dict.fromkeys(discarded_citations))


def _guard_cause(draft: CauseDraft, confidence: float, classify) -> tuple[ProbableCause, float, list[str]]:
    notes: list[str] = []
    kinds: dict[str, str] = {}
    for ref in dict.fromkeys(r.strip() for r in draft.supporting_evidence):
        kind = classify(ref)
        if kind is None:
            notes.append(f"Dropped support {ref!r} for the probable cause: not an observed fact or retrieved source")
        else:
            kinds[ref] = kind

    likelihood: Likelihood = draft.likelihood
    confidence = min(max(confidence, 0.0), 1.0)
    has_observed = "observed" in kinds.values()
    if not has_observed and not kinds:
        if likelihood != "speculative" or confidence > SPECULATIVE_CONFIDENCE_CAP:
            notes.append("No evidence supports the probable cause: marked speculative, confidence capped at 0.3")
        likelihood, confidence = "speculative", min(confidence, SPECULATIVE_CONFIDENCE_CAP)
    elif not has_observed:
        if likelihood == "likely" or confidence > KNOWLEDGE_ONLY_CONFIDENCE_CAP:
            notes.append("Probable cause rests only on knowledge base documents: at most 'possible', confidence capped at 0.5")
        likelihood = "speculative" if likelihood == "speculative" else "possible"
        confidence = min(confidence, KNOWLEDGE_ONLY_CONFIDENCE_CAP)

    cause = ProbableCause(
        description=draft.description,
        reasoning=draft.reasoning,
        likelihood=likelihood,
        supporting_evidence=list(kinds),
    )
    return cause, round(confidence, 3), notes


def _unanalyzed_finding(failure: FailureDetail) -> RootCauseFinding:
    return RootCauseFinding(
        test_case_ids=[failure.test_case_id],
        failure_category=failure.category,
        summary=f"{failure.test_case_id} failed; no root cause analysis was produced for it",
        observed_behavior=failure.actual_result or failure.message or "See observed facts",
        expected_behavior=failure.expected_result or "unknown",
        suspected_origin="unknown",
        probable_root_cause=ProbableCause(
            description="Not analyzed", reasoning="The analysis did not cover this test", likelihood="speculative"
        ),
        evidence=[
            EvidenceItem(kind="observed", reference=f.id, detail=f.detail) for f in failure.observed_facts
        ],
        unknowns=["Root cause not analyzed"],
        affected_component="unknown",
        severity_suggestion="medium",
        severity_rationale="Default: severity not assessed",
        confidence=0.0,
        adjustments=["Added placeholder: the LLM produced no finding for this failed test"],
    )


def _looks_like_fact_id(reference: str) -> bool:
    """Unknown fact-style ids are fabricated facts, not knowledge citations."""
    head, _, tail = reference.rpartition(":F")
    return bool(head) and tail.isdigit()


def render_root_cause_markdown(analysis: RootCauseAnalysis) -> str:
    """Report section that labels observed evidence, probable causes and unknowns."""
    lines = ["## Root cause analysis", "", analysis.summary, ""]
    for finding in analysis.findings:
        lines += [
            f"### {', '.join(finding.test_case_ids)}: {finding.summary}",
            f"- **Observed (recorded by test execution):** {finding.observed_behavior}",
            f"- **Expected:** {finding.expected_behavior}",
            f"- **Probable cause (not confirmed, {finding.probable_root_cause.likelihood}):** "
            f"{finding.probable_root_cause.description}",
            f"- **Suspected origin:** {finding.suspected_origin}; **component:** {finding.affected_component}; "
            f"**severity suggestion:** {finding.severity_suggestion}; **confidence:** {finding.confidence:.2f}",
            "- **Evidence:**",
            *(f"  - [{e.kind}] {e.reference}: {e.detail}" for e in finding.evidence),
            "- **Unknown:**",
            *(f"  - {u}" for u in finding.unknowns or ["nothing listed"]),
            "- **Next investigation:**",
            *(f"  - {s}" for s in finding.recommended_next_investigation),
            "",
        ]
    return "\n".join(lines)
