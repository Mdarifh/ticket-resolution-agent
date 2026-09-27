"""test_executor node running API tests through execute_api_test with mocked HTTP."""

import json
from uuid import uuid4

import httpx
import pytest
from langgraph.types import Command

from qa_agent.chains.api_request_chain import ApiRequestSpec, InvalidApiRequestSpec, spec_to_request
from qa_agent.domain import PlannedTest, Requirement, TestCase, TestExecutionPlan
from qa_agent.graph import build_qa_graph, initial_state, run_config
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.test_executor import make_test_executor
from qa_agent.tools.api_test_tool import ApiTestConfig, ApiTestExecutor
from tests.conftest import (
    PASSWORD_RESET_REQUIREMENT,
    fake_ui_runner,
    load_fixture,
    make_knowledge_base,
    password_reset_responses,
    repo_knowledge_base,
)
from tests.fakes import FakeStructuredChatModel

BASE_URL = "https://sut.test.local"
GENERIC = {"message": "If the email is registered, a reset link has been sent."}


class PasswordResetServer:
    """Mock of the documented password reset API. ``token_reuse_bug`` re-creates BUG-1042."""

    def __init__(self, token_reuse_bug: bool = False):
        self.token_reuse_bug = token_reuse_bug
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content or b"{}")
        if request.url.path == "/api/v2/password-reset":
            if "@" not in body.get("email", ""):
                return httpx.Response(400, json={"error": "invalid_email"})
            return httpx.Response(202, json=GENERIC)
        if request.url.path == "/api/v2/password-reset/confirm":
            if body.get("token") == "used-token-456" and not self.token_reuse_bug:
                return httpx.Response(400, json={"error": "token_used"})
            return httpx.Response(200, json={"message": "Password changed"})
        return httpx.Response(404, json={"error": "not_found"})


def _api_executor(server, **config) -> ApiTestExecutor:
    client = httpx.Client(transport=httpx.MockTransport(server))
    return ApiTestExecutor(ApiTestConfig(base_url=BASE_URL, **config), client)


def _test_cases() -> list[TestCase]:
    raw = load_fixture("password_reset", "test_suite.json")["test_cases"]
    return [
        TestCase.model_validate({**tc, "test_case_id": f"TC-{i:03d}"})
        for i, tc in enumerate(raw, start=1)
    ]


def _plan() -> TestExecutionPlan:
    automation = {"TC-002": "ui", "TC-003": "manual", "TC-005": "ui"}
    order = ["TC-001", "TC-004", "TC-002", "TC-006", "TC-007", "TC-005", "TC-003", "TC-008"]
    return TestExecutionPlan(
        strategy="s",
        entries=[
            PlannedTest(
                test_case_id=tc_id,
                automation_type=automation.get(tc_id, "api"),
                execution_order=position,
                rationale="r",
            )
            for position, tc_id in enumerate(order, start=1)
        ],
    )


def _state() -> dict:
    return {
        "requirement": Requirement(text=PASSWORD_RESET_REQUIREMENT),
        "test_cases": _test_cases(),
        "test_plan": _plan(),
    }


def _run(server=None, llm=None, knowledge_base=repo_knowledge_base, **config) -> dict:
    server = server or PasswordResetServer()
    node = make_test_executor(
        llm or FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base,
        lambda: _api_executor(server, **config),
        fake_ui_runner,
    )
    return node(_state())


def _by_id(update: dict) -> dict:
    return {r.test_case_id: r for r in update["execution_results"]}


# --- execution ---------------------------------------------------------------


def test_api_tests_run_against_mocked_service():
    server = PasswordResetServer()

    update = _run(server)

    results = _by_id(update)
    assert {tc: results[tc].status for tc in ["TC-001", "TC-004", "TC-007", "TC-008"]} == {
        "TC-001": "passed",
        "TC-004": "passed",
        "TC-007": "passed",
        "TC-008": "passed",
    }
    assert [str(r.url) for r in server.requests] == [
        f"{BASE_URL}/api/v2/password-reset",
        f"{BASE_URL}/api/v2/password-reset",
        f"{BASE_URL}/api/v2/password-reset/confirm",
        f"{BASE_URL}/api/v2/password-reset",
    ]


def test_results_keep_plan_order_and_cover_every_test():
    update = _run()

    assert [r.test_case_id for r in update["execution_results"]] == [e.test_case_id for e in _plan().entries]


def test_manual_and_non_executable_tests_are_skipped_ui_tests_run():
    results = _by_id(_run())

    assert results["TC-003"].status == "skipped"
    assert results["TC-002"].status == results["TC-005"].status == "passed"
    assert results["TC-006"].status == "skipped"
    assert results["TC-006"].message.startswith("Not executable: Needs a reset token past its expiry")


def test_api_result_is_kept_as_evidence():
    result = _by_id(_run())["TC-007"]

    evidence = result.evidence
    assert evidence["status_code"] == 400
    assert evidence["response_body"] == {"error": "token_used"}
    assert evidence["method"] == "POST"
    assert evidence["url"] == f"{BASE_URL}/api/v2/password-reset/confirm"
    assert evidence["automation_type"] == "api"
    assert evidence["response_time"] is not None


def test_regression_is_reported_as_failure_with_reasons():
    results = _by_id(_run(PasswordResetServer(token_reuse_bug=True)))

    failed = results["TC-007"]
    assert failed.status == "failed"
    assert failed.message == (
        'Expected status 400, got 200; JSON path \'error\' not found'
    )


def test_network_failure_is_reported_as_error():
    def unreachable(request):
        raise httpx.ConnectError("connection refused", request=request)

    results = _by_id(_run(unreachable))

    assert results["TC-001"].status == "error"
    assert results["TC-001"].message == "ConnectError: connection refused"


def test_request_prompt_contains_api_tests_and_api_docs():
    llm = FakeStructuredChatModel(responses=password_reset_responses())

    update = _run(llm=llm)

    [messages] = llm.calls_for("ApiRequestPlan")
    prompt = messages[-1].content
    for tc_id in ["TC-001", "TC-004", "TC-006", "TC-007", "TC-008"]:
        assert tc_id in prompt
    assert "TC-003" not in prompt  # manual tests are not sent
    assert "[source: api_docs/auth-password-reset-api.md" in prompt
    assert "Never output a scheme, hostname or full URL" in messages[0].content
    [usage] = update["retrieved_knowledge"]
    assert usage.node == "test_executor"
    assert all(source.startswith("api_docs/") for source in usage.retrieved_sources)


def test_works_without_reference_material():
    empty = make_knowledge_base()

    results = _by_id(_run(knowledge_base=lambda: empty))

    assert results["TC-001"].status == "passed"


# --- LLM spec safety ---------------------------------------------------------


def _responses_with(**changes) -> dict:
    responses = password_reset_responses()
    plan = responses["ApiRequestPlan"][0]
    for request in plan["requests"]:
        request.update(changes.get(request["test_case_id"], {}))
    return responses


def test_absolute_url_from_llm_is_rejected_without_traffic():
    server = PasswordResetServer()
    llm = FakeStructuredChatModel(
        responses=_responses_with(**{"TC-001": {"path": "https://prod.example.com/api/v2/password-reset"}})
    )

    results = _by_id(_run(server, llm=llm))

    assert results["TC-001"].status == "error"
    assert "must be relative" in results["TC-001"].message
    assert all(r.url.host == "sut.test.local" for r in server.requests)


def test_invalid_body_json_from_llm_is_an_error_for_that_test_only():
    llm = FakeStructuredChatModel(responses=_responses_with(**{"TC-004": {"body_json": "{email:"}}))

    results = _by_id(_run(llm=llm))

    assert results["TC-004"].status == "error"
    assert "body_json is not valid JSON" in results["TC-004"].message
    assert results["TC-001"].status == "passed"


def test_missing_spec_is_an_error():
    responses = password_reset_responses()
    responses["ApiRequestPlan"][0]["requests"] = [
        r for r in responses["ApiRequestPlan"][0]["requests"] if r["test_case_id"] != "TC-008"
    ]

    results = _by_id(_run(llm=FakeStructuredChatModel(responses=responses)))

    assert results["TC-008"].status == "error"
    assert results["TC-008"].message == "No API request was generated for this test"


def test_unconfigured_base_url_errors_without_llm_call_or_traffic():
    server = PasswordResetServer()
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    node = make_test_executor(
        llm,
        repo_knowledge_base,
        lambda: ApiTestExecutor(ApiTestConfig(base_url=None), httpx.Client(transport=httpx.MockTransport(server))),
        lambda: fake_ui_runner(base_url=None),
    )

    results = _by_id(node(_state()))

    assert results["TC-001"].status == "error"
    assert "API_TEST_BASE_URL" in results["TC-001"].message
    assert results["TC-003"].status == "skipped"
    assert llm.calls == []
    assert server.requests == []


def test_no_api_tests_means_no_llm_call():
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    state = _state()
    state["test_plan"] = TestExecutionPlan(
        strategy="s",
        entries=[PlannedTest(test_case_id="TC-003", automation_type="manual", execution_order=1, rationale="r")],
    )

    update = make_test_executor(llm, repo_knowledge_base, lambda: _api_executor(PasswordResetServer()))(state)

    assert [r.status for r in update["execution_results"]] == ["skipped"]
    assert llm.calls == []


@pytest.mark.parametrize("path", ["api/v2/x", "//evil.example.com/x", "/redirect?to=https://x"])
def test_spec_paths_must_be_relative(path):
    spec = ApiRequestSpec.model_validate(
        {**load_fixture("password_reset", "api_request_plan.json")["requests"][0], "path": path}
    )

    with pytest.raises(InvalidApiRequestSpec):
        spec_to_request(spec)


def test_spec_converts_to_api_test_request():
    spec = ApiRequestSpec.model_validate(load_fixture("password_reset", "api_request_plan.json")["requests"][3])

    request = spec_to_request(spec)

    assert request.method == "POST"
    assert request.url == "/api/v2/password-reset/confirm"
    assert request.body == {"token": "used-token-456", "new_password": "An0ther-Passw0rd!"}
    assert request.validation_rules[0].expected == "token_used"


def test_spec_with_invalid_rule_is_rejected():
    raw = load_fixture("password_reset", "api_request_plan.json")["requests"][0]
    raw["validation_rules"] = [{"type": "json_path_equals", "target": None, "expected_json": "1"}]

    with pytest.raises(InvalidApiRequestSpec, match="requires a target"):
        spec_to_request(ApiRequestSpec.model_validate(raw))


# --- graph -------------------------------------------------------------------


def _graph(server, llm):
    return build_qa_graph(
        {"confidence_checker": make_confidence_checker(threshold=0.7)},
        llm=llm,
        knowledge_base=repo_knowledge_base(),
        api_executor=_api_executor(server),
        ui_runner=fake_ui_runner(),
    )


def test_graph_success_path_with_real_api_executor():
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)

    state = _graph(PasswordResetServer(), llm).invoke(initial_state(requirement), run_config(requirement.id))

    report = state["final_report"]
    assert report.status == "passed"
    assert (report.total_tests, report.passed, report.skipped) == (6, 6, 2)
    assert state["execution_metadata"].nodes_visited[-2:] == ["result_analyzer", "final_report_generator"]


def test_graph_failure_path_with_real_api_executor():
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    graph = _graph(PasswordResetServer(token_reuse_bug=True), llm)
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)
    config = run_config(requirement.id)

    paused = graph.invoke(initial_state(requirement), config)

    assert [f.test_case_id for f in paused["failures"]] == ["TC-007"]
    assert "Expected status 400, got 200" in paused["failures"][0].message
    assert graph.get_state(config).next == ("human_review",)  # RCA confidence 0.62 < 0.7

    state = graph.invoke(Command(resume={"decision": "approve"}), config)
    assert state["final_report"].status == "failed"
    [bug] = state["final_report"].bug_reports
    assert bug.related_test_case_ids == ["TC-007"]
    assert bug.related_bugs == ["previous_bugs/BUG-1042-reset-token-reusable.md"]
