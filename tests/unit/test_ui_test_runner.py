"""UiTestRunner orchestration over an in-memory browser (no Playwright needed)."""

import pytest
from pydantic import ValidationError

from qa_agent.chains.ui_script_chain import InvalidUiScript, UiScriptSpec, script_to_request
from qa_agent.config import Settings
from qa_agent.domain.ui_test import UiStep, UiTestRequest
from qa_agent.tools.ui_test_tool import (
    UiTestConfig,
    UiTestRunner,
    clean_playwright_message,
    create_execute_ui_test_tool,
)
from tests.conftest import PASSING_UI_TEXTS, UI_BASE_URL, fake_ui_runner, load_fixture
from tests.fakes import FakeUiBrowser

RESULT_FIELDS = {"test_case_id", "status", "executed_steps", "failed_step", "error", "execution_time"}


def _request(*steps: dict, test_case_id: str = "TC-005") -> UiTestRequest:
    return UiTestRequest(test_case_id=test_case_id, steps=[UiStep.model_validate(s) for s in steps])


FORGOT_PASSWORD_INVALID_EMAIL = (
    {"action": "navigate", "target": "/forgot-password.html"},
    {"action": "select", "target": "data-testid=language-select", "value": "en"},
    {"action": "fill", "target": "data-testid=email-input", "value": "not-an-email"},
    {"action": "click", "target": "data-testid=send-reset-link"},
    {"action": "verify_visible", "target": "data-testid=email-error"},
    {"action": "verify_text", "target": "data-testid=email-error", "value": "valid email address"},
)


def _runner(**browser_kwargs) -> tuple[UiTestRunner, FakeUiBrowser]:
    browser = FakeUiBrowser(texts=dict(PASSING_UI_TEXTS), **browser_kwargs)
    return fake_ui_runner(browser), browser


# --- step execution ----------------------------------------------------------


def test_all_supported_actions_run_in_order():
    runner, browser = _runner()

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.status == "passed"
    assert [a[0] for a in browser.actions] == [
        "navigate", "select", "fill", "click", "verify_visible", "verify_text",
    ]
    assert browser.actions[0] == ("navigate", f"{UI_BASE_URL}/forgot-password.html", None)
    assert browser.actions[2] == ("fill", "data-testid=email-input", "not-an-email")


def test_result_is_fully_structured():
    runner, _ = _runner()

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert RESULT_FIELDS <= set(result.model_dump())
    assert [s.index for s in result.executed_steps] == [1, 2, 3, 4, 5, 6]
    assert all(s.status == "passed" and s.duration_ms >= 0 for s in result.executed_steps)
    assert result.failed_step is None
    assert result.error is None
    assert result.execution_time >= 0
    assert result.diagnostics is None


def test_failure_stops_at_failed_step_with_diagnostics():
    runner, browser = _runner(console_errors=["TypeError: form is null"])
    browser.texts["data-testid=email-error"] = "Something went wrong"

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.status == "failed"
    assert len(result.executed_steps) == 6
    failed = result.failed_step
    assert (failed.index, failed.action, failed.status) == (6, "verify_text", "failed")
    assert "Actual value: Something went wrong" in failed.error
    assert result.error.startswith("Step 6 (verify_text 'data-testid=email-error') failed:")
    assert result.diagnostics.element_text == "Something went wrong"
    assert result.diagnostics.matching_elements == 1
    assert result.diagnostics.console_errors == ["TypeError: form is null"]
    assert result.diagnostics.page_url == f"{UI_BASE_URL}/forgot-password.html"


def test_steps_after_a_failure_are_not_executed():
    runner, browser = _runner(missing={"data-testid=send-reset-link"})

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.failed_step.index == 4
    assert [s.index for s in result.executed_steps] == [1, 2, 3, 4]
    assert [a[0] for a in browser.actions] == ["navigate", "select", "fill", "click"]
    assert result.diagnostics.matching_elements == 0


def test_navigation_failure_captures_page_diagnostics_without_selector():
    runner, _ = _runner(broken_paths={"/forgot-password.html"})

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.status == "failed"
    assert result.failed_step.action == "navigate"
    assert "HTTP 404" in result.error
    assert result.diagnostics.matching_elements is None


def test_app_is_opened_when_script_does_not_start_with_navigate():
    runner, browser = _runner()

    result = runner.run(_request({"action": "verify_visible", "target": "data-testid=page-title"}))

    assert result.status == "passed"
    assert browser.actions[0] == ("navigate", f"{UI_BASE_URL}/", None)
    assert [s.action for s in result.executed_steps] == ["navigate", "verify_visible"]


def test_each_test_gets_a_fresh_session_that_is_closed():
    runner, browser = _runner(missing={"data-testid=send-reset-link"})

    runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))
    runner.run(_request({"action": "navigate", "target": "/"}))

    assert browser.sessions == browser.closed_sessions == 2


def test_injected_browser_is_not_closed_by_runner():
    runner, browser = _runner()

    runner.close()

    assert browser.closed is False


# --- safety and errors -------------------------------------------------------


def test_missing_base_url_errors_without_opening_a_browser():
    browser = FakeUiBrowser()
    runner = UiTestRunner(UiTestConfig(base_url=None), browser)

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.status == "error"
    assert "UI_TEST_BASE_URL" in result.error
    assert browser.sessions == 0


@pytest.mark.parametrize(
    "target",
    ["https://www.google.com/", "http://production.example.com/login", "file:///etc/passwd"],
)
def test_navigation_outside_test_hosts_is_refused_before_any_step(target):
    runner, browser = _runner()

    result = runner.run(
        _request({"action": "navigate", "target": "/forgot-password.html"}, {"action": "navigate", "target": target})
    )

    assert result.status == "error"
    assert browser.sessions == 0 and browser.actions == []


def test_navigation_to_explicitly_allowed_host_is_permitted():
    browser = FakeUiBrowser()
    runner = fake_ui_runner(browser, allowed_hosts=["auth.test.local"])

    result = runner.run(_request({"action": "navigate", "target": "http://auth.test.local/login"}))

    assert result.status == "passed"


def test_production_environment_is_refused():
    browser = FakeUiBrowser()
    runner = fake_ui_runner(browser, app_env="production")

    result = runner.run(_request({"action": "navigate", "target": "/"}))

    assert result.status == "error"
    assert "UI_TEST_ALLOW_PRODUCTION" in result.error
    assert browser.sessions == 0


def test_browser_launch_failure_is_an_error_result():
    runner, _ = _runner(fail_launch=RuntimeError("Executable doesn't exist at /ms-playwright/chromium\nmore"))

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.status == "error"
    assert result.error == "Could not start browser: Executable doesn't exist at /ms-playwright/chromium"
    assert result.executed_steps == []


def test_unexpected_browser_exception_is_an_error_not_a_failure():
    runner, browser = _runner()

    def crash(selector):
        raise RuntimeError("Target page, context or browser has been closed")

    session_factory = browser.new_session

    def new_session():
        session = session_factory()
        session.click = crash
        return session

    browser.new_session = new_session

    result = runner.run(_request(*FORGOT_PASSWORD_INVALID_EMAIL))

    assert result.status == "error"
    assert result.failed_step.index == 4
    assert result.error == "Browser error at step 4: Target page, context or browser has been closed"
    assert browser.closed_sessions == 1


def test_page_inventory_requires_configuration():
    browser = FakeUiBrowser(pages=["page"])

    with pytest.raises(ValueError, match="UI_TEST_BASE_URL"):
        UiTestRunner(UiTestConfig(base_url=None), browser).page_inventory()
    assert browser.inventory_calls == 0


# --- models, config, messages ------------------------------------------------


@pytest.mark.parametrize("action", ["fill", "select", "verify_text"])
def test_steps_needing_a_value_require_one(action):
    with pytest.raises(ValidationError):
        UiStep(action=action, target="#x")


def test_invalid_steps_are_rejected():
    with pytest.raises(ValidationError):
        UiStep(action="hover", target="#x")
    with pytest.raises(ValidationError):
        UiStep(action="click", target="")
    with pytest.raises(ValidationError):
        UiTestRequest(test_case_id="TC-1", steps=[])


def test_config_from_settings_has_no_default_target():
    config = UiTestConfig.from_settings(Settings(_env_file=None))

    assert config.base_url is None
    assert config.screenshot_dir is None  # screenshots are opt-in
    assert config.permitted_hosts() == set()


def test_config_from_settings_parses_values():
    config = UiTestConfig.from_settings(
        Settings(
            _env_file=None,
            ui_test_base_url="http://127.0.0.1:8765",
            ui_test_allowed_hosts="auth.test.local",
            ui_test_step_timeout_ms=1234,
        )
    )

    assert config.permitted_hosts() == {"127.0.0.1", "auth.test.local"}
    assert config.step_timeout_ms == 1234


def test_clean_playwright_message_drops_call_log_and_colours():
    raw = (
        "Locator expected to contain text 'x'\nActual value: y \n\x1b[2mCall log:\x1b[22m\n"
        "  - waiting for locator"
    )

    assert clean_playwright_message(Exception(raw)) == "Locator expected to contain text 'x' | Actual value: y"


# --- LangChain tool ----------------------------------------------------------


def test_langchain_tool_runs_steps_and_returns_structured_dict():
    runner, _ = _runner()
    tool = create_execute_ui_test_tool(runner)

    output = tool.invoke(
        {"test_case_id": "TC-005", "steps": [dict(s) for s in FORGOT_PASSWORD_INVALID_EMAIL]}
    )

    assert tool.name == "execute_ui_test"
    assert RESULT_FIELDS <= set(output)
    assert output["status"] == "passed"
    assert len(output["executed_steps"]) == 6


def test_langchain_tool_rejects_invalid_input():
    tool = create_execute_ui_test_tool(_runner()[0])

    with pytest.raises(ValidationError):
        tool.invoke({"test_case_id": "TC-1", "steps": [{"action": "fill", "target": "#x"}]})


# --- LLM script conversion ---------------------------------------------------


def _script(index: int = 1, **changes) -> UiScriptSpec:
    raw = load_fixture("password_reset", "ui_script_plan.json")["scripts"][index]
    return UiScriptSpec.model_validate({**raw, **changes})


def test_script_converts_to_request():
    request = script_to_request(_script())

    assert request.test_case_id == "TC-005"
    assert request.steps[0].target == "/forgot-password.html"
    assert request.steps[-1].value == "Please enter a valid email address."


@pytest.mark.parametrize("target", ["https://evil.example.com/", "//evil.example.com", "forgot-password.html"])
def test_script_navigation_must_be_relative(target):
    raw = load_fixture("password_reset", "ui_script_plan.json")["scripts"][1]
    raw["steps"][0]["target"] = target

    with pytest.raises(InvalidUiScript, match="relative path"):
        script_to_request(UiScriptSpec.model_validate(raw))


def test_script_with_invalid_step_is_rejected():
    raw = load_fixture("password_reset", "ui_script_plan.json")["scripts"][1]
    raw["steps"][1]["value"] = None  # fill without a value

    with pytest.raises(InvalidUiScript, match="requires a value"):
        script_to_request(UiScriptSpec.model_validate(raw))
