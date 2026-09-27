"""Test cases -> ``TestExecutionPlan``.

The LLM decides each test's automation type (api / ui / manual) with a
rationale. Deterministic post-processing then guarantees every test case is
planned exactly once and assigns the execution order.
"""

from collections.abc import Collection
from typing import ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import PlannedTest, TestCase, TestExecutionPlan
from qa_agent.domain.test_case import AutomationType


class AutomationDecision(BaseModel):
    test_case_id: str
    automation_type: AutomationType
    rationale: str = Field(description="One sentence on why this automation type fits.")


class TestPlanDraft(BaseModel):
    """LLM-facing schema; turned into a ``TestExecutionPlan`` by ``finalize_plan``."""

    __test__: ClassVar[bool] = False  # not a pytest test class

    strategy: str = Field(description="Short description of the overall execution approach.")
    decisions: list[AutomationDecision] = Field(description="Exactly one decision per test case.")


TEST_PLANNING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a QA test planner. For every test case decide how it will be executed:\n"
            "- api: the behaviour is fully verifiable through HTTP requests and responses.\n"
            "- ui: verification needs a browser: rendering, navigation, client-side "
            "validation or an end-to-end user journey.\n"
            "- manual: needs human judgement (usability, visual quality) or depends on "
            "something a test script cannot reliably control, such as a real email inbox, "
            "SMS, third-party systems or physical devices.\n"
            "Prefer api over ui when both could verify the behaviour. "
            "Return exactly one decision per test case id.",
        ),
        ("human", "Test cases (JSON):\n{test_cases}"),
    ]
)

_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
_AUTOMATION_RANK = {"api": 0, "ui": 1, "manual": 2}


def build_test_plan_chain(llm: BaseChatModel) -> Runnable[dict, TestPlanDraft]:
    return structured_chain(TEST_PLANNING_PROMPT, llm, TestPlanDraft)


def plan_tests(
    llm: BaseChatModel,
    test_cases: list[TestCase],
    available: Collection[AutomationType] | None = None,
) -> TestExecutionPlan:
    payload = "[" + ",".join(tc.model_dump_json() for tc in test_cases) + "]"
    draft = build_test_plan_chain(llm).invoke({"test_cases": payload})
    return finalize_plan(draft, test_cases, available)


def _reroute(
    automation_type: AutomationType, available: Collection[AutomationType] | None
) -> AutomationType:
    """Swap api <-> ui when only the other has a configured target; otherwise keep it."""
    if available is None or automation_type in available:
        return automation_type
    if automation_type == "api" and "ui" in available:
        return "ui"
    if automation_type == "ui" and "api" in available:
        return "api"
    return automation_type


def finalize_plan(
    draft: TestPlanDraft,
    test_cases: list[TestCase],
    available: Collection[AutomationType] | None = None,
) -> TestExecutionPlan:
    """Reconcile LLM decisions with the real test cases and order execution.

    Unknown ids and duplicate decisions are dropped; test cases without a
    decision keep the generator's suggested automation type. When ``available``
    is given, an api/ui test whose target is not configured moves to the other
    type if that one is. Each adjustment is recorded in ``warnings``. Order:
    priority, then api -> ui -> manual, then original position.
    """
    known = {tc.test_case_id for tc in test_cases}
    decisions: dict[str, AutomationDecision] = {}
    warnings: list[str] = []

    for decision in draft.decisions:
        if decision.test_case_id not in known:
            warnings.append(f"Ignored decision for unknown test case {decision.test_case_id}")
        elif decision.test_case_id in decisions:
            warnings.append(f"Ignored duplicate decision for {decision.test_case_id}")
        else:
            decisions[decision.test_case_id] = decision

    unordered: list[tuple[TestCase, AutomationType, str]] = []
    for tc in test_cases:
        decision = decisions.get(tc.test_case_id)
        if decision is None:
            warnings.append(
                f"No planner decision for {tc.test_case_id}; "
                f"kept suggested automation type '{tc.automation_type}'"
            )
            unordered.append((tc, tc.automation_type, "Generator suggestion (planner gave none)"))
        else:
            unordered.append((tc, decision.automation_type, decision.rationale))

    for i, (tc, automation_type, rationale) in enumerate(unordered):
        rerouted = _reroute(automation_type, available)
        if rerouted != automation_type:
            warnings.append(
                f"{tc.test_case_id}: planned as '{automation_type}' but no {automation_type} "
                f"test target is configured; running it as '{rerouted}'"
            )
            note = f"(Moved to {rerouted}: no {automation_type} target configured.)"
            unordered[i] = (tc, rerouted, f"{rationale} {note}")

    ordered = sorted(
        enumerate(unordered),
        key=lambda item: (
            _PRIORITY_RANK[item[1][0].priority],
            _AUTOMATION_RANK[item[1][1]],
            item[0],
        ),
    )
    entries = [
        PlannedTest(
            test_case_id=tc.test_case_id,
            automation_type=automation_type,
            execution_order=order,
            rationale=rationale,
        )
        for order, (_, (tc, automation_type, rationale)) in enumerate(ordered, start=1)
    ]
    return TestExecutionPlan(strategy=draft.strategy, entries=entries, warnings=warnings)


def apply_plan(test_cases: list[TestCase], plan: TestExecutionPlan) -> list[TestCase]:
    """Copy the planner's automation type onto each test case so state stays consistent."""
    planned = {e.test_case_id: e.automation_type for e in plan.entries}
    return [
        tc.model_copy(update={"automation_type": planned.get(tc.test_case_id, tc.automation_type)})
        for tc in test_cases
    ]
