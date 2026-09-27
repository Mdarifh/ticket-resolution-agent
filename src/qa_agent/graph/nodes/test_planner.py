"""Test Planner node: decides api / ui / manual per test and the execution order."""

from collections.abc import Collection
from typing import Any

from langchain_core.language_models import BaseChatModel

from qa_agent.chains import apply_plan, get_chat_model, plan_tests
from qa_agent.config import Settings, get_settings
from qa_agent.domain import TestExecutionPlan
from qa_agent.domain.test_case import AutomationType
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.state import QAAgentState


def configured_automation_types(settings: Settings) -> set[AutomationType] | None:
    """Automation types with a configured target, or None when neither api nor ui has one
    (then nothing is rerouted and the executor reports the missing configuration)."""
    available: set[AutomationType] = {"manual"}
    if settings.api_test_base_url:
        available.add("api")
    if settings.ui_test_base_url:
        available.add("ui")
    return available if len(available) > 1 else None


def make_test_planner(
    llm: BaseChatModel | None = None,
    available: Collection[AutomationType] | None = None,
) -> NodeFn:
    """``llm`` and ``available`` default to the configured chat model and test targets,
    resolved when the node runs."""

    def test_planner(state: QAAgentState) -> dict[str, Any]:
        test_cases = state.get("test_cases") or []
        if not test_cases:
            return {"test_plan": TestExecutionPlan(strategy="No test cases to plan")}

        types = available if available is not None else configured_automation_types(get_settings())
        plan = plan_tests(llm or get_chat_model(), test_cases, types)
        return {"test_plan": plan, "test_cases": apply_plan(test_cases, plan)}

    return test_planner
