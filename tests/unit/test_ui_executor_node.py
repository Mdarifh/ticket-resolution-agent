"""test_executor node orchestrating UI tests (in-memory browser, fake LLM)."""

import json
from uuid import uuid4

import httpx
from langgraph.types import Command

from qa_agent.domain import PlannedTest, Requirement, TestCase, TestExecutionPlan
from qa_agent.graph import build_qa_graph, initial_state, run_config
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.test_executor import make_test_executor
from qa_agent.tools.api_test_tool import ApiTestConfig, ApiTestExecutor
from tests.conftest import (
    PASSING_UI_TEXTS,
    PASSWORD_RESET_REQUIREMENT,
    UI_BASE_URL,
    demo_inventory,
    fake_ui_runner,
    load_fixture,
    password_reset_responses,
    repo_knowledge_base,
)
from tests.fakes import FakeStructuredChatModel, FakeUiBrowser


def _test_cases() -> list[TestCase]:
    raw = load_fixture("password_reset", "test_suite.json")["test_cases"]
    return [
        TestCase.model_validate({**tc, "test_case_id": f"TC-{i:03d}"})
        for i, tc in enumerate(raw, start=1)
    ]


def _plan(automation: dict[str, str]) -> TestExecutionPlan:
    return TestExecutionPlan(
        strategy="s",
        entries=[
            PlannedTest(test_case_id=tc_id, automation_type=kind, execution_order=i, rationale="r")
            for i, (tc_id, kind) in enumerate(automation.items(), start=1)
        ],
    )


UI_ONLY = {"TC-002": "ui", "TC-005": "ui", "TC-003": "manual"}


def _state(automation: dict[str, str] = UI_ONLY) -> dict:
    return {
        "requirement": Requirement(text=PASSWORD_RESET_REQUIREMENT),
        "test_cases": _test_cases(),
        "test_plan": _plan(automation),
    }


def _browser(**kwargs) -> FakeUiBrowser:
    return FakeUiBrowser(texts=dict(PASSING_UI_TEXTS), pages=demo_inventory(), **kwargs)


def _run(browser=None, llm=None, state=None, api_executor=None, **config) -> dict:
    browser = browser or _browser()
    node = make_test_executor(
        llm or FakeStructuredChatModel(responses=password_reset_responses()),
        repo_knowledge_base,
        api_executor,
        lambda: fake_ui_runner(browser, **config),
    )
    return node(state or _state())


def _by_id(update: dict) -> dict:
    return {r.test_case_id: r for r in update["execution_results"]}


# --- orchestration -----------------------------------------------------------


def test_ui_tests_are_scripted_and_run():
    browser = _browser()

    results = _by_id(_run(browser))

    assert results["TC-002"].status == "passed"
    assert results["TC-005"].status == "passed"
    assert results["TC-003"].status == "skipped"
    navigations = [a[1] for a in browser.actions if a[0] == "navigate"]
    assert navigations == [
        f"{UI_BASE_URL}/reset-password.html?token=valid-token-123",
        f"{UI_BASE_URL}/forgot-password.html",
    ]
    assert browser.inventory_calls == 1


def test_ui_evidence_keeps_steps_and_timing():
    evidence = _by_id(_run())["TC-005"].evidence

    assert evidence["automation_type"] == "ui"
    assert evidence["status"] == "passed"
    assert [s["action"] for s in evidence["executed_steps"]] == [
        "navigate", "fill", "click", "verify_visible", "verify_text",
    ]
    assert evidence["execution_time"] >= 0


def test_ui_failure_reports_failed_step_and_structured_diagnostics():
    browser = _browser(console_errors=["Uncaught TypeError: validate is not a function"])
    browser.texts["data-testid=email-error"] = ""

    result = _by_id(_run(browser))["TC-005"]

    assert result.status == "failed"
    assert result.message.startswith("Step 5 (verify_text 'data-testid=email-error') failed:")
    evidence = result.evidence
    assert evidence["failed_step"]["index"] == 5
    assert evidence["diagnostics"]["console_errors"] == ["Uncaught TypeError: validate is not a function"]
    assert evidence["diagnostics"]["screenshot_path"] is None  # analysis does not rely on screenshots


def test_prompt_contains_ui_tests_and_page_inventory_only():
    llm = FakeStructuredChatModel(responses=password_reset_responses())

    _run(llm=llm)

    [messages] = llm.calls_for("UiScriptPlan")
    prompt = messages[-1].content
    assert "TC-002" in prompt and "TC-005" in prompt
    assert "TC-003" not in prompt  # manual tests are not scripted
    assert '"test_id": "send-reset-link"' in prompt
    assert '"path": "/forgot-password.html"' in prompt
    assert "never output a scheme, hostname or full URL" in messages[0].content


def test_skip_reason_from_llm_skips_the_test():
    responses = password_reset_responses()
    responses["UiScriptPlan"][0]["scripts"][0].update(steps=[], skip_reason="Needs an email inbox")

    results = _by_id(_run(llm=FakeStructuredChatModel(responses=responses)))

    assert results["TC-002"].status == "skipped"
    assert results["TC-002"].message == "Not executable: Needs an email inbox"


def test_absolute_navigation_from_llm_is_rejected_without_browser_traffic():
    responses = password_reset_responses()
    responses["UiScriptPlan"][0]["scripts"][1]["steps"][0]["target"] = "https://www.example.com/forgot"
    browser = _browser()

    results = _by_id(_run(browser, llm=FakeStructuredChatModel(responses=responses)))

    assert results["TC-005"].status == "error"
    assert "relative path" in results["TC-005"].message
    assert all("example.com" not in a[1] for a in browser.actions)


def test_missing_script_is_an_error():
    responses = password_reset_responses()
    responses["UiScriptPlan"][0]["scripts"] = responses["UiScriptPlan"][0]["scripts"][:1]

    results = _by_id(_run(llm=FakeStructuredChatModel(responses=responses)))

    assert results["TC-005"].status == "error"
    assert results["TC-005"].message == "No UI script was generated for this test"


def test_unconfigured_ui_errors_without_browser_or_llm():
    browser = _browser()
    llm = FakeStructuredChatModel(responses=password_reset_responses())

    results = _by_id(_run(browser, llm=llm, base_url=None))

    assert results["TC-002"].status == results["TC-005"].status == "error"
    assert "UI_TEST_BASE_URL" in results["TC-002"].message
    assert llm.calls == []
    assert browser.inventory_calls == 0 and browser.sessions == 0


def test_empty_inventory_is_an_error_without_llm_call():
    llm = FakeStructuredChatModel(responses=password_reset_responses())

    results = _by_id(_run(FakeUiBrowser(pages=[]), llm=llm))

    assert results["TC-002"].status == "error"
    assert "No pages of the app could be loaded" in results["TC-002"].message
    assert llm.calls == []


def test_browser_launch_failure_marks_ui_tests_as_errors():
    browser = _browser(fail_launch=RuntimeError("Executable doesn't exist"))

    results = _by_id(_run(browser))

    assert results["TC-002"].status == "error"
    assert "Could not start browser" in results["TC-002"].message


def test_ui_preparation_failure_does_not_lose_api_results():
    def server(request):
        body = json.loads(request.content or b"{}")
        return httpx.Response(202 if "@" in body.get("email", "") else 400, json={"message": "If the email is registered, a reset link has been sent."})

    api_executor = ApiTestExecutor(
        ApiTestConfig(base_url="https://sut.test.local"), httpx.Client(transport=httpx.MockTransport(server))
    )

    class BrokenInventoryBrowser(FakeUiBrowser):
        def inventory(self, start_url, max_pages):
            raise RuntimeError("inventory crashed")

    update = _run(
        BrokenInventoryBrowser(),
        api_executor=lambda: api_executor,
        state=_state({"TC-001": "api", "TC-002": "ui"}),
    )

    results = _by_id(update)
    assert results["TC-001"].status == "passed"
    assert results["TC-002"].status == "error"
    assert results["TC-002"].message == "UI tests could not run: RuntimeError: inventory crashed"


def test_no_ui_tests_means_no_browser_or_script_generation():
    browser = _browser()
    llm = FakeStructuredChatModel(responses=password_reset_responses())

    _run(browser, llm=llm, state=_state({"TC-003": "manual"}))

    assert browser.inventory_calls == 0
    assert llm.calls == []


# --- graph -------------------------------------------------------------------


def _mocked_api(token_reuse_bug: bool = False) -> ApiTestExecutor:
    def server(request):
        body = json.loads(request.content or b"{}")
        if request.url.path.endswith("/confirm"):
            if body.get("token") == "used-token-456" and not token_reuse_bug:
                return httpx.Response(400, json={"error": "token_used"})
            return httpx.Response(200, json={"message": "Password changed"})
        return httpx.Response(202, json={"message": "If the email is registered, a reset link has been sent."})

    return ApiTestExecutor(ApiTestConfig(base_url="https://sut.test.local"), httpx.Client(transport=httpx.MockTransport(server)))


def test_graph_runs_api_and_ui_tests_to_a_passing_report():
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    graph = build_qa_graph(
        llm=llm, knowledge_base=repo_knowledge_base(), api_executor=_mocked_api(), ui_runner=fake_ui_runner()
    )
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)

    state = graph.invoke(initial_state(requirement), run_config(requirement.id))

    statuses = {r.test_case_id: r.status for r in state["execution_results"]}
    assert statuses["TC-002"] == statuses["TC-005"] == "passed"
    assert state["final_report"].status == "passed"


def test_graph_ui_failure_goes_down_the_failure_path():
    browser = _browser()
    browser.texts["data-testid=login-message"] = "Invalid email or password."
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    graph = build_qa_graph(
        {"confidence_checker": make_confidence_checker(threshold=0.7)},
        llm=llm,
        knowledge_base=repo_knowledge_base(),
        api_executor=_mocked_api(),
        ui_runner=fake_ui_runner(browser),
    )
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)
    config = run_config(requirement.id)

    paused = graph.invoke(initial_state(requirement), config)

    [failure] = paused["failures"]
    assert failure.test_case_id == "TC-002"
    assert "Step 10 (verify_text 'data-testid=login-message') failed" in failure.message
    assert "Actual value: Invalid email or password." in failure.message
    assert graph.get_state(config).next == ("human_review",)

    state = graph.invoke(Command(resume={"decision": "approve"}), config)
    assert state["final_report"].status == "failed"
