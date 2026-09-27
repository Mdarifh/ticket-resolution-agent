"""LangGraph wiring for the QA agent workflow.

    requirement_analyzer -> test_case_generator -> test_planner -> test_executor
    -> result_analyzer
         |-- no failures ---------------------------------> final_report_generator
         `-- failures --> failure_analyzer -> root_cause_analyzer
                          -> bug_report_generator -> confidence_checker
                               |-- confident -------------> final_report_generator
                               `-- low confidence --> human_review
                                    |-- APPROVE / REJECT --> final_report_generator
                                    `-- REQUEST_REANALYSIS -> root_cause_analyzer (loop)
    final_report_generator -> END
"""

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from qa_agent.graph import routing
from qa_agent.graph.checkpointer import create_default_checkpointer
from qa_agent.graph.instrumentation import NodeFn, instrument
from qa_agent.graph.nodes import default_nodes
from qa_agent.graph.state import QAAgentState
from qa_agent.rag.knowledge_base import QAKnowledgeBase
from qa_agent.tools.api_test_tool import ApiTestExecutor
from qa_agent.tools.ui_test_tool import UiTestRunner

REQUIREMENT_ANALYZER = "requirement_analyzer"
TEST_CASE_GENERATOR = "test_case_generator"
TEST_PLANNER = "test_planner"
TEST_EXECUTOR = "test_executor"
RESULT_ANALYZER = "result_analyzer"
FAILURE_ANALYZER = routing.FAILURE_ANALYZER
ROOT_CAUSE_ANALYZER = routing.ROOT_CAUSE_ANALYZER
BUG_REPORT_GENERATOR = routing.BUG_REPORT_GENERATOR
CONFIDENCE_CHECKER = "confidence_checker"
HUMAN_REVIEW = routing.HUMAN_REVIEW
FINAL_REPORT_GENERATOR = routing.FINAL_REPORT_GENERATOR

NODE_NAMES: tuple[str, ...] = (
    REQUIREMENT_ANALYZER,
    TEST_CASE_GENERATOR,
    TEST_PLANNER,
    TEST_EXECUTOR,
    RESULT_ANALYZER,
    FAILURE_ANALYZER,
    ROOT_CAUSE_ANALYZER,
    BUG_REPORT_GENERATOR,
    CONFIDENCE_CHECKER,
    HUMAN_REVIEW,
    FINAL_REPORT_GENERATOR,
)


def build_state_graph(
    nodes: Mapping[str, NodeFn] | None = None,
    *,
    llm: BaseChatModel | None = None,
    knowledge_base: QAKnowledgeBase | None = None,
    api_executor: ApiTestExecutor | None = None,
    ui_runner: UiTestRunner | None = None,
) -> StateGraph:
    """Build the uncompiled StateGraph.

    ``nodes`` overrides individual node implementations by name: used by tests
    now, and by later phases to swap mocks for real LLM/tool-backed nodes.
    ``llm`` is the chat model for LLM-backed nodes (default: configured model);
    ``knowledge_base`` backs RAG in the generator and root cause analyzer
    (default: the persistent knowledge base from settings); ``api_executor`` runs
    API tests (default: built from API_TEST_* settings per run); ``ui_runner``
    runs UI tests in a browser (default: built from UI_TEST_* settings per run).
    """
    overrides = dict(nodes or {})
    unknown = set(overrides) - set(NODE_NAMES)
    if unknown:
        raise ValueError(f"Unknown node name(s): {sorted(unknown)}")

    implementations = {
        **default_nodes(llm, knowledge_base, api_executor, ui_runner),
        **overrides,
    }

    graph = StateGraph(QAAgentState)
    for name in NODE_NAMES:
        graph.add_node(name, instrument(name, implementations[name]))

    graph.add_edge(START, REQUIREMENT_ANALYZER)
    graph.add_edge(REQUIREMENT_ANALYZER, TEST_CASE_GENERATOR)
    graph.add_edge(TEST_CASE_GENERATOR, TEST_PLANNER)
    graph.add_edge(TEST_PLANNER, TEST_EXECUTOR)
    graph.add_edge(TEST_EXECUTOR, RESULT_ANALYZER)

    graph.add_conditional_edges(
        RESULT_ANALYZER,
        routing.route_after_result_analysis,
        [FAILURE_ANALYZER, FINAL_REPORT_GENERATOR],
    )

    graph.add_edge(FAILURE_ANALYZER, ROOT_CAUSE_ANALYZER)
    graph.add_edge(ROOT_CAUSE_ANALYZER, BUG_REPORT_GENERATOR)
    graph.add_edge(BUG_REPORT_GENERATOR, CONFIDENCE_CHECKER)

    graph.add_conditional_edges(
        CONFIDENCE_CHECKER,
        routing.route_after_confidence_check,
        [HUMAN_REVIEW, FINAL_REPORT_GENERATOR],
    )
    graph.add_conditional_edges(
        HUMAN_REVIEW,
        routing.route_after_human_review,
        [ROOT_CAUSE_ANALYZER, FINAL_REPORT_GENERATOR],
    )

    graph.add_edge(FINAL_REPORT_GENERATOR, END)
    return graph


def build_qa_graph(
    nodes: Mapping[str, NodeFn] | None = None,
    *,
    llm: BaseChatModel | None = None,
    knowledge_base: QAKnowledgeBase | None = None,
    api_executor: ApiTestExecutor | None = None,
    ui_runner: UiTestRunner | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the workflow.

    A checkpointer is always attached because human review pauses the graph
    with ``interrupt()`` and resumes it later. In-memory is the default until
    the Postgres checkpointer lands.
    """
    return build_state_graph(
        nodes,
        llm=llm,
        knowledge_base=knowledge_base,
        api_executor=api_executor,
        ui_runner=ui_runner,
    ).compile(
        checkpointer=checkpointer or create_default_checkpointer()
    )


def run_config(run_id: str, **extra: Any) -> dict[str, Any]:
    """Invocation config: LangGraph checkpoints are keyed by thread_id = run_id."""
    return {"configurable": {"thread_id": run_id, **extra}}
