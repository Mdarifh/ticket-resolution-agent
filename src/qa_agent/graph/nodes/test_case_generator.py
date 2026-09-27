"""Test Case Generator node: ``RequirementAnalysis`` + QA knowledge -> test cases via LLM."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from qa_agent.chains import generate_test_suite, get_chat_model
from qa_agent.domain import RequirementAnalysis
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.state import QAAgentState
from qa_agent.rag.context import (
    KnowledgeBaseProvider,
    retrieve,
    usage_record,
    validate_citations,
)
from qa_agent.rag.knowledge_base import get_default_knowledge_base


def knowledge_query(analysis: RequirementAnalysis) -> str:
    parts = [analysis.feature, analysis.summary, *analysis.risk_areas, *analysis.edge_cases]
    return "\n".join(p for p in parts if p)


def make_test_case_generator(
    llm: BaseChatModel | None = None,
    knowledge_base: KnowledgeBaseProvider | None = None,
) -> NodeFn:
    """``llm`` / ``knowledge_base`` default to the configured ones, resolved when the node runs."""
    provider = knowledge_base or get_default_knowledge_base

    def test_case_generator(state: QAAgentState) -> dict[str, Any]:
        analysis = state.get("requirement_analysis")
        if analysis is None:
            raise ValueError("requirement_analysis is required to generate test cases")

        knowledge = retrieve(provider, knowledge_query(analysis))
        suite = generate_test_suite(
            llm or get_chat_model(), state["requirement"].text, analysis, knowledge
        )
        cited, discarded = validate_citations(suite.knowledge_sources, knowledge)
        return {
            "test_cases": suite.test_cases,
            "retrieved_knowledge": [
                usage_record("test_case_generator", knowledge, cited, discarded)
            ],
        }

    return test_case_generator
