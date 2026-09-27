"""``execute_ui_test``: run a UI test's steps in a real browser with Playwright.

Safety, enforced by ``target_policy`` and a browser-level route guard:

* No default target: ``UI_TEST_BASE_URL`` must point at a local/test app.
* ``navigate`` targets resolve against the base URL and must stay on allowed
  hosts. Every request the page makes to any other host (links, redirects,
  scripts, images) is aborted and recorded, so tests never exercise
  arbitrary external websites.
* Refuses to run when ``APP_ENV=production`` unless explicitly overridden.

Failures are reported as structured data: which step failed, the selector,
Playwright's expected/actual message, how many elements matched, the
element's text, page URL/title, console errors and failed requests.
Screenshots are off by default and are only a supporting artifact.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from qa_agent.config import Settings, get_settings
from qa_agent.domain.ui_test import (
    UiDiagnostics,
    UiStep,
    UiStepResult,
    UiTestRequest,
    UiTestResult,
)
from qa_agent.tools import target_policy
from qa_agent.observability import traced
from qa_agent.tools.target_policy import UnsafeRequestError

logger = logging.getLogger(__name__)

MAX_TEXT = 300
MAX_ELEMENTS_PER_PAGE = 100


class UiStepError(Exception):
    """A step could not be completed on the page (element missing, assertion failed...)."""


class UiTestConfig(BaseModel):
    base_url: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    browser: str = "chromium"
    headless: bool = True
    step_timeout_ms: int = Field(default=5000, gt=0)
    navigation_timeout_ms: int = Field(default=15000, gt=0)
    max_inventory_pages: int = Field(default=5, ge=1)
    screenshot_dir: str | None = None
    app_env: str = "local"
    allow_production: bool = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "UiTestConfig":
        settings = settings or get_settings()
        return cls(
            base_url=settings.ui_test_base_url or None,
            allowed_hosts=target_policy.split_hosts(settings.ui_test_allowed_hosts),
            browser=settings.ui_test_browser,
            headless=settings.ui_test_headless,
            step_timeout_ms=settings.ui_test_step_timeout_ms,
            navigation_timeout_ms=settings.ui_test_navigation_timeout_ms,
            max_inventory_pages=settings.ui_test_max_inventory_pages,
            screenshot_dir=settings.ui_test_screenshot_dir or None,
            app_env=settings.app_env,
            allow_production=settings.ui_test_allow_production,
        )

    def permitted_hosts(self) -> set[str]:
        return target_policy.permitted_hosts(self.base_url, self.allowed_hosts)

    def check_usable(self) -> None:
        target_policy.check_base_url(
            self.base_url,
            setting="UI_TEST_BASE_URL",
            app_env=self.app_env,
            allow_production=self.allow_production,
            override="UI_TEST_ALLOW_PRODUCTION",
        )

    def resolve(self, url: str) -> str:
        return target_policy.resolve(self.base_url, url, self.permitted_hosts())


# --- page inventory (what the LLM may use as selectors) -----------------------


class ElementInfo(BaseModel):
    tag: str
    type: str | None = None
    test_id: str | None = None
    id: str | None = None
    name: str | None = None
    label: str | None = None
    placeholder: str | None = None
    text: str | None = None
    href: str | None = None
    options: list[str] = Field(default_factory=list)
    shown_later: bool = Field(
        default=False,
        description="Hidden message region (role=status/alert, aria-live) revealed after an action.",
    )


class PageInventory(BaseModel):
    path: str
    title: str
    elements: list[ElementInfo] = Field(default_factory=list)


# --- driver abstraction -------------------------------------------------------


class UiSession(Protocol):
    """One isolated browser session (fresh context) for one test."""

    def navigate(self, url: str) -> None: ...
    def click(self, selector: str) -> None: ...
    def fill(self, selector: str, value: str) -> None: ...
    def select(self, selector: str, value: str) -> None: ...
    def verify_text(self, selector: str, expected: str) -> None: ...
    def verify_visible(self, selector: str) -> None: ...
    def diagnostics(self, selector: str | None, artifact_name: str) -> UiDiagnostics: ...
    def close(self) -> None: ...


class UiBrowser(Protocol):
    def new_session(self) -> UiSession: ...
    def inventory(self, start_url: str, max_pages: int) -> list[PageInventory]: ...
    def close(self) -> None: ...


class UiTestRunner:
    """Runs ``UiTestRequest``s. ``browser`` defaults to a lazily started Playwright browser."""

    def __init__(self, config: UiTestConfig, browser: UiBrowser | None = None) -> None:
        self.config = config
        self._browser = browser
        self._owns_browser = browser is None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "UiTestRunner":
        return cls(UiTestConfig.from_settings(settings))

    @property
    def browser(self) -> UiBrowser:
        if self._browser is None:
            self._browser = PlaywrightBrowser(self.config)
        return self._browser

    def close(self) -> None:
        if self._owns_browser and self._browser is not None:
            self._browser.close()
            self._browser = None

    def __enter__(self) -> "UiTestRunner":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @traced("ui_page_inventory", run_type="tool")
    def page_inventory(self) -> list[PageInventory]:
        self.config.check_usable()
        pages = self.browser.inventory(self.config.resolve("/"), self.config.max_inventory_pages)
        logger.info("UI inventory crawled %d pages", len(pages))
        return pages

    @traced("ui_test", run_type="tool")
    def run(self, request: UiTestRequest) -> UiTestResult:
        """Run one browser test (traced in LangSmith)."""
        result = self._run(request)
        log = logger.warning if result.status == "error" else logger.info
        log(
            "UI test %s -> %s after %d/%d steps (%.0f ms)%s",
            request.test_case_id,
            result.status,
            len(result.executed_steps),
            len(request.steps),
            result.execution_time,
            f": {result.error}" if result.error else "",
        )
        return result

    def _run(self, request: UiTestRequest) -> UiTestResult:
        started = time.perf_counter()

        def result(**fields: Any) -> UiTestResult:
            return UiTestResult(
                test_case_id=request.test_case_id,
                execution_time=round((time.perf_counter() - started) * 1000, 3),
                **fields,
            )

        try:
            self.config.check_usable()
            steps = list(request.steps)
            if steps[0].action != "navigate":
                steps.insert(0, UiStep(action="navigate", target="/", description="Open the app"))
            # Validate every navigation target before touching the browser.
            for step in steps:
                if step.action == "navigate":
                    self.config.resolve(step.target)
            session = self.browser.new_session()
        except UnsafeRequestError as exc:
            return result(status="error", error=str(exc))
        except Exception as exc:  # browser missing / failed to launch
            return result(status="error", error=f"Could not start browser: {_first_line(exc)}")

        executed: list[UiStepResult] = []
        try:
            for index, step in enumerate(steps, start=1):
                step_started = time.perf_counter()
                try:
                    _perform(session, step, self.config)
                except UiStepError as exc:
                    failed = _step_result(index, step, step_started, error=str(exc))
                    executed.append(failed)
                    diagnostics = session.diagnostics(
                        None if step.action == "navigate" else step.target,
                        f"{request.test_case_id}-step{index}",
                    )
                    return result(
                        status="failed",
                        executed_steps=executed,
                        failed_step=failed,
                        error=f"Step {index} ({step.action} {step.target!r}) failed: {exc}",
                        diagnostics=diagnostics,
                    )
                except Exception as exc:  # browser crashed, closed, protocol error...
                    failed = _step_result(index, step, step_started, error=_first_line(exc))
                    executed.append(failed)
                    return result(
                        status="error",
                        executed_steps=executed,
                        failed_step=failed,
                        error=f"Browser error at step {index}: {_first_line(exc)}",
                    )
                executed.append(_step_result(index, step, step_started))
            return result(status="passed", executed_steps=executed)
        finally:
            session.close()


def _perform(session: UiSession, step: UiStep, config: UiTestConfig) -> None:
    if step.action == "navigate":
        session.navigate(config.resolve(step.target))
    elif step.action == "click":
        session.click(step.target)
    elif step.action == "fill":
        session.fill(step.target, step.value)
    elif step.action == "select":
        session.select(step.target, step.value)
    elif step.action == "verify_text":
        session.verify_text(step.target, step.value)
    elif step.action == "verify_visible":
        session.verify_visible(step.target)


def _step_result(index: int, step: UiStep, started: float, error: str | None = None) -> UiStepResult:
    return UiStepResult(
        index=index,
        action=step.action,
        target=step.target,
        value=step.value,
        status="failed" if error else "passed",
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        error=error,
    )


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text.splitlines()[0]


def clean_playwright_message(exc: BaseException) -> str:
    """Playwright messages end with a verbose call log; keep the informative head."""
    text = str(exc).split("Call log:")[0].strip()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # strip ANSI colours
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[:4]) or type(exc).__name__


# --- Playwright implementation ------------------------------------------------

_INVENTORY_JS = """
(maxElements) => {
  const visible = (el) => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  const clip = (s) => (s || '').replace(/\\s+/g, ' ').trim().slice(0, 80) || null;
  const nodes = document.querySelectorAll(
    'a[href], button, input, select, textarea, [role=button], [data-testid]');
  const out = [];
  // Hidden message regions (errors, confirmations) are kept: tests verify them after an action.
  const messageRegion = (el) => ['status', 'alert'].includes(el.getAttribute('role'))
    || el.hasAttribute('aria-live');
  for (const el of nodes) {
    if (el.type === 'hidden') continue;
    const shownLater = !visible(el) && messageRegion(el);
    if (!visible(el) && !shownLater) continue;
    const labels = el.labels ? Array.from(el.labels).map(l => l.innerText).join(' ') : '';
    out.push({
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute('type'),
      test_id: el.getAttribute('data-testid'),
      id: el.id || null,
      name: el.getAttribute('name'),
      label: clip(labels || el.getAttribute('aria-label')),
      placeholder: el.getAttribute('placeholder'),
      text: ['input', 'select', 'textarea'].includes(el.tagName.toLowerCase()) ? null : clip(el.innerText),
      href: el.getAttribute('href'),
      options: el.tagName === 'SELECT' ? Array.from(el.options).map(o => `${o.value}: ${o.text}`) : [],
      shown_later: shownLater,
    });
    if (out.length >= maxElements) break;
  }
  return out;
}
"""


class PlaywrightBrowser:
    def __init__(self, config: UiTestConfig) -> None:
        from playwright.sync_api import sync_playwright  # imported lazily: optional at runtime

        self.config = config
        self._playwright = sync_playwright().start()
        try:
            launcher = getattr(self._playwright, config.browser)
            self._browser = launcher.launch(headless=config.headless)
        except Exception:
            self._playwright.stop()
            raise

    def new_session(self) -> "PlaywrightSession":
        return PlaywrightSession(self._browser.new_context(), self.config)

    def inventory(self, start_url: str, max_pages: int) -> list[PageInventory]:
        session = self.new_session()
        try:
            return session.crawl_inventory(start_url, max_pages)
        finally:
            session.close()

    def close(self) -> None:
        try:
            self._browser.close()
        finally:
            self._playwright.stop()


class PlaywrightSession:
    def __init__(self, context: Any, config: UiTestConfig) -> None:
        self.config = config
        self.context = context
        self.hosts = config.permitted_hosts()
        self.console_errors: list[str] = []
        self.failed_requests: list[str] = []
        self.blocked_requests: list[str] = []
        context.set_default_timeout(config.step_timeout_ms)
        context.set_default_navigation_timeout(config.navigation_timeout_ms)
        context.route("**/*", self._guard)
        self.page = context.new_page()
        self.page.on("console", self._on_console)
        self.page.on("pageerror", lambda error: self.console_errors.append(f"pageerror: {error}"))
        self.page.on("requestfailed", self._on_request_failed)

    # --- event handlers ---

    def _guard(self, route: Any) -> None:
        url = route.request.url
        parts = urlsplit(url)
        if parts.scheme in ("data", "blob", "about") or (parts.hostname or "").lower() in self.hosts:
            route.continue_()
        else:
            self.blocked_requests.append(f"{route.request.method} {url}")
            route.abort("blockedbyclient")

    def _on_console(self, message: Any) -> None:
        if message.type == "error":
            self.console_errors.append(message.text[:MAX_TEXT])

    def _on_request_failed(self, request: Any) -> None:
        if any(request.url in blocked for blocked in self.blocked_requests):
            return
        self.failed_requests.append(f"{request.method} {request.url}: {request.failure}")

    # --- actions ---

    def navigate(self, url: str) -> None:
        response = self._call(lambda: self.page.goto(url, wait_until="load"))
        if response is not None and response.status >= 400:
            raise UiStepError(f"Navigation to {url} returned HTTP {response.status}")

    def click(self, selector: str) -> None:
        self._call(lambda: self.page.locator(selector).click())

    def fill(self, selector: str, value: str) -> None:
        self._call(lambda: self.page.locator(selector).fill(value))

    def select(self, selector: str, value: str) -> None:
        self._call(lambda: self.page.locator(selector).select_option(value))

    def verify_text(self, selector: str, expected: str) -> None:
        from playwright.sync_api import expect

        self._call(lambda: expect(self.page.locator(selector)).to_contain_text(expected))

    def verify_visible(self, selector: str) -> None:
        from playwright.sync_api import expect

        self._call(lambda: expect(self.page.locator(selector)).to_be_visible())

    def _call(self, action: Any) -> Any:
        from playwright.sync_api import Error as PlaywrightError

        try:
            return action()
        except (PlaywrightError, AssertionError) as exc:
            if self.page.is_closed():
                raise  # not a page-level failure: surface as a browser error
            raise UiStepError(clean_playwright_message(exc)) from exc

    # --- diagnostics & inventory ---

    def diagnostics(self, selector: str | None, artifact_name: str) -> UiDiagnostics:
        diagnostics = UiDiagnostics(
            page_url=self.page.url,
            console_errors=list(self.console_errors),
            failed_requests=list(self.failed_requests),
            blocked_requests=list(self.blocked_requests),
        )
        try:
            diagnostics.page_title = self.page.title()
            if selector:
                locator = self.page.locator(selector)
                diagnostics.matching_elements = locator.count()
                if diagnostics.matching_elements:
                    text = locator.first.inner_text(timeout=1000)
                    diagnostics.element_text = text[:MAX_TEXT]
            if self.config.screenshot_dir:
                path = Path(self.config.screenshot_dir) / f"{_safe_name(artifact_name)}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.page.screenshot(path=str(path))
                diagnostics.screenshot_path = str(path)
        except Exception:  # diagnostics are best effort and must not mask the failure
            pass
        return diagnostics

    def crawl_inventory(self, start_url: str, max_pages: int) -> list[PageInventory]:
        """Visit the start page and same-host pages it links to (breadth-first)."""
        queue, seen, pages = [start_url], set(), []
        while queue and len(pages) < max_pages:
            url = queue.pop(0).split("#")[0]
            if url in seen:
                continue
            seen.add(url)
            try:
                target_policy.check_target(url, self.hosts)
                response = self.page.goto(url, wait_until="load")
            except Exception:
                continue
            if response is not None and response.status >= 400:
                continue
            elements = [
                ElementInfo.model_validate(e)
                for e in self.page.evaluate(_INVENTORY_JS, MAX_ELEMENTS_PER_PAGE)
            ]
            parts = urlsplit(self.page.url)
            path = parts.path + (f"?{parts.query}" if parts.query else "")
            pages.append(PageInventory(path=path or "/", title=self.page.title(), elements=elements))
            for link in self.page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)"):
                if (urlsplit(link).hostname or "").lower() in self.hosts and link not in seen:
                    queue.append(link)
        return pages

    def close(self) -> None:
        try:
            self.context.close()
        except Exception:
            pass


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def create_execute_ui_test_tool(runner: UiTestRunner) -> BaseTool:
    """``execute_ui_test`` as a LangChain tool bound to ``runner``."""

    def _run(**kwargs: Any) -> dict:
        return runner.run(UiTestRequest.model_validate(kwargs)).model_dump(mode="json")

    return StructuredTool.from_function(
        func=_run,
        name="execute_ui_test",
        description=(
            "Run a UI test in a browser against the configured local/test web app. Steps: "
            "navigate (path), click, fill, select, verify_text, verify_visible (Playwright "
            "selectors such as data-testid=submit). Returns test_case_id, status, "
            "executed_steps, failed_step, error, execution_time (ms) and diagnostics."
        ),
        args_schema=UiTestRequest,
    )
