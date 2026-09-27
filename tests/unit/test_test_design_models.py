import pytest
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import ValidationError

from qa_agent.chains.api_request_chain import ApiRequestPlan
from qa_agent.chains.bug_report_chain import BugNarrativeDraft
from qa_agent.chains.root_cause_chain import RootCauseDraft
from qa_agent.chains.test_plan_chain import TestPlanDraft
from qa_agent.chains.ui_script_chain import UiScriptPlan
from qa_agent.domain import (
    PlannedTest,
    RequirementAnalysis,
    TestCase,
    TestExecutionPlan,
    TestSuite,
)
from tests.conftest import load_fixture

TEST_CASE_FIELDS = {
    "test_case_id",
    "title",
    "description",
    "preconditions",
    "test_data",
    "steps",
    "expected_result",
    "priority",
    "test_type",
    "automation_type",
}


def _test_case(**overrides) -> dict:
    payload = load_fixture("password_reset", "test_suite.json")["test_cases"][0]
    return {**payload, **overrides}


def test_test_case_has_exactly_the_required_fields():
    assert set(TestCase.model_fields) == TEST_CASE_FIELDS


@pytest.mark.parametrize("field", sorted(TEST_CASE_FIELDS))
def test_test_case_rejects_missing_field(field):
    payload = _test_case()
    del payload[field]

    with pytest.raises(ValidationError):
        TestCase.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("automation_type", "robot"), ("test_type", "smoke"), ("priority", "urgent")],
)
def test_test_case_rejects_values_outside_enums(field, value):
    with pytest.raises(ValidationError):
        TestCase.model_validate(_test_case(**{field: value}))


def test_fixture_payloads_validate():
    RequirementAnalysis.model_validate(load_fixture("password_reset", "requirement_analysis.json"))
    suite = TestSuite.model_validate(load_fixture("password_reset", "test_suite.json"))
    TestPlanDraft.model_validate(load_fixture("password_reset", "test_plan_draft.json"))

    assert {tc.test_type for tc in suite.test_cases} == {"positive", "negative", "edge_case"}


def test_execution_plan_ids_for_filters_and_orders():
    plan = TestExecutionPlan(
        strategy="s",
        entries=[
            PlannedTest(test_case_id="TC-2", automation_type="api", execution_order=2, rationale="r"),
            PlannedTest(test_case_id="TC-3", automation_type="manual", execution_order=3, rationale="r"),
            PlannedTest(test_case_id="TC-1", automation_type="api", execution_order=1, rationale="r"),
        ],
    )

    assert plan.ids_for("api") == ["TC-1", "TC-2"]
    assert plan.ids_for("manual") == ["TC-3"]
    assert plan.ids_for("ui") == []


def _objects(schema: dict):
    """Yield every object-typed node in a JSON schema."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _objects(item)


@pytest.mark.parametrize(
    "model",
    [
        RequirementAnalysis,
        TestSuite,
        TestPlanDraft,
        RootCauseDraft,
        ApiRequestPlan,
        UiScriptPlan,
        BugNarrativeDraft,
    ],
)
def test_llm_schemas_are_strict_structured_output_compatible(model):
    """OpenAI strict mode needs every property required and no open-ended objects."""
    schema = type_to_response_format_param(model)["json_schema"]["schema"]

    for node in _objects(schema):
        assert node.get("additionalProperties") is False
        assert set(node.get("required", [])) == set(node.get("properties", {}))
