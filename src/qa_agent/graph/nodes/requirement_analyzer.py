"""Requirement Analyzer node: requirement text -> ``RequirementAnalysis`` via LLM."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from qa_agent.chains import analyze_requirement, get_chat_model
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.state import QAAgentState


def make_requirement_analyzer(llm: BaseChatModel | None = None) -> NodeFn:
    """``llm`` defaults to the configured chat model, resolved when the node runs."""

    def requirement_analyzer(state: QAAgentState) -> dict[str, Any]:
        analysis = analyze_requirement(llm or get_chat_model(), state["requirement"])
        return {"requirement_analysis": analysis}

    return requirement_analyzer
