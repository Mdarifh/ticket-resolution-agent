"""Graph node implementations.

Nodes marked MOCK in their module docstring return deterministic fake data
until the chains/tools that back them are built in later phases.
"""

from langchain_core.language_models import BaseChatModel

from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.nodes.bug_report_generator import make_bug_report_generator
from qa_agent.graph.nodes.confidence_checker import confidence_checker
from qa_agent.graph.nodes.failure_analyzer import failure_analyzer
from qa_agent.graph.nodes.final_report_generator import final_report_generator
from qa_agent.graph.nodes.human_review import human_review
from qa_agent.graph.nodes.requirement_analyzer import make_requirement_analyzer
from qa_agent.graph.nodes.result_analyzer import result_analyzer
from qa_agent.graph.nodes.root_cause_analyzer import make_root_cause_analyzer
from qa_agent.graph.nodes.test_case_generator import make_test_case_generator
from qa_agent.graph.nodes.test_executor import make_test_executor
from qa_agent.graph.nodes.test_planner import make_test_planner
from qa_agent.rag.knowledge_base import QAKnowledgeBase
from qa_agent.tools.api_test_tool import ApiTestExecutor
from qa_agent.tools.ui_test_tool import UiTestRunner


def default_nodes(
    llm: BaseChatModel | None = None,
    knowledge_base: QAKnowledgeBase | None = None,
    api_executor: ApiTestExecutor | None = None,
    ui_runner: UiTestRunner | None = None,
) -> dict[str, NodeFn]:
    """Shared dependencies for the nodes that need them; None means the configured
    model, the default persistent knowledge base, and API/UI settings from the env."""
    kb_provider = (lambda: knowledge_base) if knowledge_base is not None else None
    api_provider = (lambda: api_executor) if api_executor is not None else None
    ui_provider = (lambda: ui_runner) if ui_runner is not None else None
    return {
        "requirement_analyzer": make_requirement_analyzer(llm),
        "test_case_generator": make_test_case_generator(llm, kb_provider),
        "test_planner": make_test_planner(llm),
        "test_executor": make_test_executor(llm, kb_provider, api_provider, ui_provider),
        "result_analyzer": result_analyzer,
        "failure_analyzer": failure_analyzer,
        "root_cause_analyzer": make_root_cause_analyzer(llm, kb_provider),
        "bug_report_generator": make_bug_report_generator(llm, kb_provider),
        "confidence_checker": confidence_checker,
        "human_review": human_review,
        "final_report_generator": final_report_generator,
    }
