"""Real-browser tests: Playwright against the local demo app (demo_app/).

Skipped automatically when no Playwright browser is installed
(install with: python -m playwright install chromium).
"""

from pathlib import Path

import pytest

from qa_agent.chains.ui_script_chain import UiScriptSpec, script_to_request
from qa_agent.domain.ui_test import UiStep, UiTestRequest
from qa_agent.tools.demo_server import serve_demo_app
from qa_agent.tools.ui_test_tool import UiTestConfig, UiTestRunner
from tests.conftest import load_fixture

pytestmark = pytest.mark.integration

DEMO_DIR = Path(__file__).parents[2] / "demo_app"


@pytest.fixture(scope="module")
def demo_url():
    with serve_demo_app(DEMO_DIR) as base_url:
        yield base_url


@pytest.fixture(scope="module")
def runner(demo_url):
    runner = UiTestRunner(UiTestConfig(base_url=demo_url, step_timeout_ms=2000))
    try:
        runner.browser  # launch once per module
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"Playwright browser not available: {str(exc).splitlines()[0]}")
    yield runner
    runner.close()


def _request(*steps: dict, test_case_id: str = "TC-X") -> UiTestRequest:
    return UiTestRequest(test_case_id=test_case_id, steps=[UiStep.model_validate(s) for s in steps])


@pytest.mark.parametrize("index", [0, 1], ids=["TC-002-reset-and-login", "TC-005-invalid-email"])
def test_fixture_scripts_pass_against_demo_app(runner, index):
    """The canned LLM scripts used in unit tests really work on the demo pages."""
    raw = load_fixture("password_reset", "ui_script_plan.json")["scripts"][index]

    result = runner.run(script_to_request(UiScriptSpec.model_validate(raw)))

    assert result.status == "passed", result.error
    assert all(step.status == "passed" for step in result.executed_steps)
    assert result.execution_time > 0


def test_select_changes_page_language(runner):
    result = runner.run(
        _request(
            {"action": "navigate", "target": "/forgot-password.html"},
            {"action": "select", "target": "data-testid=language-select", "value": "Español"},
            {"action": "verify_text", "target": "data-testid=page-title", "value": "¿Olvidaste tu contraseña?"},
            {"action": "verify_text", "target": "data-testid=send-reset-link", "value": "Enviar enlace"},
        )
    )

    assert result.status == "passed", result.error


def test_no_account_enumeration_in_ui(runner):
    result = runner.run(
        _request(
            {"action": "navigate", "target": "/forgot-password.html"},
            {"action": "fill", "target": "data-testid=email-input", "value": "nobody@example.com"},
            {"action": "click", "target": "data-testid=send-reset-link"},
            {"action": "verify_text", "target": "data-testid=confirmation",
             "value": "If the email is registered, a reset link has been sent."},
        )
    )

    assert result.status == "passed", result.error


def test_failed_assertion_reports_structured_diagnostics(runner, demo_url):
    result = runner.run(
        _request(
            {"action": "navigate", "target": "/reset-password.html?token=used-token-456"},
            {"action": "fill", "target": "data-testid=new-password", "value": "N3w-Passw0rd!"},
            {"action": "fill", "target": "data-testid=confirm-password", "value": "N3w-Passw0rd!"},
            {"action": "click", "target": "data-testid=reset-submit"},
            {"action": "verify_text", "target": "data-testid=reset-error", "value": "Your password has been reset"},
            {"action": "click", "target": "data-testid=reset-submit"},
        )
    )

    assert result.status == "failed"
    assert result.failed_step.index == 5
    assert len(result.executed_steps) == 5  # the step after the failure did not run
    assert "Your password has been reset" in result.failed_step.error
    assert "Call log" not in result.failed_step.error
    diagnostics = result.diagnostics
    assert diagnostics.element_text == "This reset link has already been used."
    assert diagnostics.matching_elements == 1
    assert diagnostics.page_url.startswith(f"{demo_url}/reset-password.html")
    assert diagnostics.page_title == "Demo Shop - Reset password"
    assert diagnostics.screenshot_path is None


def test_missing_element_fails_quickly_with_zero_matches(runner):
    result = runner.run(
        _request(
            {"action": "navigate", "target": "/login.html"},
            {"action": "click", "target": "data-testid=does-not-exist"},
        )
    )

    assert result.status == "failed"
    assert result.diagnostics.matching_elements == 0
    assert result.failed_step.duration_ms < 10_000


def test_external_sites_are_never_reached(runner):
    result = runner.run(
        _request(
            {"action": "navigate", "target": "/forgot-password.html"},
            {"action": "click", "target": "data-testid=external-help"},
            {"action": "verify_visible", "target": "text=Password help"},
        )
    )

    assert result.status == "failed"
    assert result.diagnostics.blocked_requests == ["GET https://docs.example.com/password-help"]


def test_navigate_to_external_url_is_refused(runner):
    result = runner.run(_request({"action": "navigate", "target": "https://www.example.com/"}))

    assert result.status == "error"
    assert "not an allowed test host" in result.error


def test_missing_page_fails_navigation_step(runner):
    result = runner.run(_request({"action": "navigate", "target": "/no-such-page.html"}))

    assert result.status == "failed"
    assert "HTTP 404" in result.failed_step.error


def test_page_inventory_crawls_same_host_pages(runner, monkeypatch):
    monkeypatch.setattr(runner.config, "max_inventory_pages", 10)
    pages = {page.path: page for page in runner.page_inventory()}

    assert set(pages) == {
        "/",
        "/login.html",
        "/signup.html",
        "/forgot-password.html",
        "/outbox.html",
        "/reset-password.html?token=valid-token-123",
    }
    forgot = pages["/forgot-password.html"]
    test_ids = {e.test_id for e in forgot.elements}
    assert {"email-input", "send-reset-link", "language-select"} <= test_ids
    select = next(e for e in forgot.elements if e.test_id == "language-select")
    assert select.options == ["en: English", "es: Español"]
    email = next(e for e in forgot.elements if e.test_id == "email-input")
    assert email.label == "Email address"
    # Hidden message regions are listed (flagged shown_later); other hidden elements are not.
    hidden = {e.test_id for e in forgot.elements if e.shown_later}
    assert hidden == {"email-error", "confirmation"}
    visible = {e.test_id for e in forgot.elements if not e.shown_later}
    assert "email-error" not in visible
    login = {e.test_id: e for e in pages["/login.html"].elements}
    assert login["login-message"].shown_later is True


def test_screenshot_is_only_taken_when_configured(runner, tmp_path, monkeypatch):
    # Reuse the module's browser: Playwright's sync API allows one instance per thread.
    monkeypatch.setattr(runner.config, "screenshot_dir", str(tmp_path))

    result = runner.run(
        _request({"action": "navigate", "target": "/login.html"},
                 {"action": "verify_visible", "target": "data-testid=login-message"},
                 test_case_id="TC-SHOT")
    )

    assert result.status == "failed"
    assert result.diagnostics.screenshot_path == str(tmp_path / "TC-SHOT-step2.png")
    assert Path(result.diagnostics.screenshot_path).stat().st_size > 0
    # The textual diagnosis stands on its own without the image.
    assert result.failed_step.error and result.diagnostics.matching_elements == 1
