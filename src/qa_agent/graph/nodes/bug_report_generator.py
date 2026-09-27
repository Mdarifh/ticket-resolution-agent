"""Bug Report Generator node: one structured report per root cause finding.

Facts come from recorded test data; the LLM writes only the narrative (see
``qa_agent.chains.bug_report_chain``). Past bugs are retrieved from the
knowledge base to spot possible duplicates and regressions. When a reviewer
requests changes, reports are regenerated with a higher revision and the
feedback in the prompt.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from qa_agent.chains import get_chat_model
from qa_agent.chains.bug_report_chain import BugReportContext, generate_bug_reports
from qa_agent.config import get_settings
from qa_agent.domain import KnowledgeSearchResult
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.nodes.root_cause_analyzer import reanalysis_feedback
from qa_agent.graph.state import QAAgentState
from qa_agent.rag.context import KnowledgeBaseProvider, retrieve, usage_record
from qa_agent.rag.knowledge_base import get_default_knowledge_base

BUG_REPORT_DOC_TYPES = ["previous_bug"]


def make_bug_report_generator(
    llm: BaseChatModel | None = None,
    knowledge_base: KnowledgeBaseProvider | None = None,
    *,
    use_llm: bool = True,
) -> NodeFn:
    """``use_llm=False`` builds reports without any LLM or retrieval (deterministic narrative)."""
    provider = knowledge_base or get_default_knowledge_base

    def bug_report_generator(state: QAAgentState) -> dict[str, Any]:
        root_cause = state.get("root_cause_analysis")
        failures = state.get("failures") or []
        if root_cause is None or not failures:
            return {"bug_reports": []}

        # Each REQUEST_REANALYSIS round produces a new revision carrying the feedback.
        feedback = reanalysis_feedback(state)
        revision = sum(
            1 for r in state.get("review_history") or [] if r.human_decision == "REQUEST_REANALYSIS"
        )

        context = BugReportContext(
            requirement_text=state["requirement"].text,
            requirement_analysis=state.get("requirement_analysis"),
            root_cause=root_cause,
            failures=failures,
            test_cases=state.get("test_cases") or [],
            execution_results=state.get("execution_results") or [],
            app_env=get_settings().app_env,
            run_id=state["requirement"].id,
            reviewer_feedback=feedback,
            revision=revision,
        )

        if not use_llm:
            no_retrieval = KnowledgeSearchResult(query="", message="retrieval not performed")
            reports, _, _ = generate_bug_reports(None, context, no_retrieval)
            return {"bug_reports": reports}

        query = "\n".join(f"{f.affected_component} {f.summary} {f.observed_behavior}" for f in root_cause.findings)
        knowledge = retrieve(provider, query, doc_types=BUG_REPORT_DOC_TYPES)
        try:
            chat_model = llm or get_chat_model()
        except Exception:
            chat_model = None  # no LLM configured: fall back to the deterministic narrative
        reports, discarded, _ = generate_bug_reports(chat_model, context, knowledge)
        cited = list(dict.fromkeys(b for r in reports for b in r.related_bugs))
        return {
            "bug_reports": reports,
            "retrieved_knowledge": [usage_record("bug_report_generator", knowledge, cited, discarded)],
        }

    return bug_report_generator

