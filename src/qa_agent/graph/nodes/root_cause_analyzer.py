"""Root Cause Analyzer node: failures + requirement + QA knowledge -> guarded findings via LLM.

``make_mock_root_cause_analyzer`` remains for tests that need a fixed
confidence without an LLM.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from qa_agent.chains import get_chat_model
from qa_agent.chains.root_cause_chain import analyze_root_cause
from qa_agent.domain import (
    EvidenceItem,
    FailureDetail,
    ProbableCause,
    RootCauseAnalysis,
    RootCauseFinding,
    TestCase,
)
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.state import QAAgentState
from qa_agent.rag.context import KnowledgeBaseProvider, retrieve, usage_record
from qa_agent.rag.knowledge_base import get_default_knowledge_base

# Requirements are included so the analysis can spot tests that contradict them.
RCA_DOC_TYPES = ["previous_bug", "troubleshooting", "api_doc", "requirement"]


def knowledge_query(failures: list[FailureDetail], test_cases: list[TestCase]) -> str:
    titles = {tc.test_case_id: tc.title for tc in test_cases}
    lines = []
    for failure in failures:
        parts = [titles.get(failure.test_case_id) or failure.test_title, failure.category, failure.actual_result]
        parts += [fact.detail for fact in failure.observed_facts[:3]]
        lines.append(" ".join(p for p in parts if p))
    return "\n".join(dict.fromkeys(line for line in lines if line))


def reanalysis_feedback(state: QAAgentState) -> list[str]:
    """Comments from every review that asked for re-analysis, oldest first."""
    return [
        review.reviewer_comment
        for review in state.get("review_history") or []
        if review.human_decision == "REQUEST_REANALYSIS" and review.reviewer_comment
    ]


def make_root_cause_analyzer(
    llm: BaseChatModel | None = None,
    knowledge_base: KnowledgeBaseProvider | None = None,
) -> NodeFn:
    """``llm`` / ``knowledge_base`` default to the configured ones, resolved when the node runs."""
    provider = knowledge_base or get_default_knowledge_base

    def root_cause_analyzer(state: QAAgentState) -> dict[str, Any]:
        failures = state.get("failures") or []
        if not failures:
            return {"root_cause_analysis": None}
        test_cases = state.get("test_cases") or []

        knowledge = retrieve(provider, knowledge_query(failures, test_cases), doc_types=RCA_DOC_TYPES)
        analysis, discarded = analyze_root_cause(
            llm or get_chat_model(),
            state["requirement"],
            state.get("requirement_analysis"),
            failures,
            test_cases,
            knowledge,
            reviewer_feedback=reanalysis_feedback(state),
        )
        return {
            "root_cause_analysis": analysis,
            "retrieved_knowledge": [
                usage_record("root_cause_analyzer", knowledge, analysis.knowledge_sources, discarded)
            ],
        }

    return root_cause_analyzer


def make_mock_root_cause_analyzer(confidence: float = 0.85) -> NodeFn:
    """One finding per failed test at a fixed confidence; no LLM or retrieval."""

    def root_cause_analyzer(state: QAAgentState) -> dict[str, Any]:
        failures = state.get("failures") or []
        if not failures:
            return {"root_cause_analysis": None}

        findings = [
            RootCauseFinding(
                test_case_ids=[f.test_case_id],
                failure_category=f.category,
                summary=f"Mock finding: {f.category or 'unknown'} in {f.test_case_id}",
                observed_behavior=f.actual_result or f.message or "unknown",
                expected_behavior=f.expected_result or "unknown",
                suspected_origin="unknown",
                probable_root_cause=ProbableCause(
                    description=f"Mock hypothesis: {f.category or 'unknown'}",
                    reasoning="Mock analyzer",
                    likelihood="possible",
                    supporting_evidence=[fact.id for fact in f.observed_facts[:1]],
                ),
                evidence=[
                    EvidenceItem(kind="observed", reference=fact.id, detail=fact.detail)
                    for fact in f.observed_facts
                ],
                affected_component="unknown",
                severity_suggestion="medium",
                severity_rationale="Mock analyzer",
                confidence=confidence,
            )
            for f in failures
        ]
        analysis = RootCauseAnalysis(
            summary="; ".join(f.summary for f in findings), findings=findings, confidence=confidence
        )
        return {"root_cause_analysis": analysis}

    return root_cause_analyzer
