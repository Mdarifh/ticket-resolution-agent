"""Fake chat model for exercising LangChain structured output without an API.

``with_structured_output(Schema)`` on a ``BaseChatModel`` binds ``Schema`` as a
tool and parses the tool call with ``PydanticToolsParser``. This fake answers
with a tool call whose arguments are the canned payload registered for that
schema name, so tests go through LangChain's real parsing and validation.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field


class FakeStructuredChatModel(BaseChatModel):
    # schema name -> queue of payloads; the last payload repeats once the queue drains.
    responses: dict[str, list[Any]]
    calls: list[tuple[str, list[BaseMessage]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any):
        names = [convert_to_openai_tool(tool)["function"]["name"] for tool in tools]
        return self.bind(tool_names=names)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        tool_names: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not tool_names:
            raise ValueError("FakeStructuredChatModel only supports with_structured_output")
        name = tool_names[0]
        if name not in self.responses:
            raise KeyError(f"No canned response for schema {name!r}")

        queue = self.responses[name]
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        self.calls.append((name, messages))

        args = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        message = AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{len(self.calls)}"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def calls_for(self, name: str) -> list[list[BaseMessage]]:
        return [messages for called, messages in self.calls if called == name]


class FakeUiSession:
    """Scripted stand-in for a Playwright session (see FakeUiBrowser)."""

    def __init__(self, browser: "FakeUiBrowser") -> None:
        self.browser = browser
        self.url: str | None = None

    def _record(self, action: str, target: str, value: str | None = None) -> None:
        self.browser.actions.append((action, target, value))

    def _require(self, selector: str) -> None:
        from qa_agent.tools.ui_test_tool import UiStepError

        if selector in self.browser.missing:
            raise UiStepError(f"Timeout 5000ms exceeded waiting for locator('{selector}')")

    def navigate(self, url: str) -> None:
        from qa_agent.tools.ui_test_tool import UiStepError

        self._record("navigate", url)
        self.url = url
        if any(url.split("?")[0].endswith(path) for path in self.browser.broken_paths):
            raise UiStepError(f"Navigation to {url} returned HTTP 404")

    def click(self, selector: str) -> None:
        self._record("click", selector)
        self._require(selector)

    def fill(self, selector: str, value: str) -> None:
        self._record("fill", selector, value)
        self._require(selector)

    def select(self, selector: str, value: str) -> None:
        self._record("select", selector, value)
        self._require(selector)

    def verify_text(self, selector: str, expected: str) -> None:
        from qa_agent.tools.ui_test_tool import UiStepError

        self._record("verify_text", selector, expected)
        self._require(selector)
        actual = self.browser.texts.get(selector)
        if actual is None or expected not in actual:
            raise UiStepError(f"Locator expected to contain text '{expected}' | Actual value: {actual}")

    def verify_visible(self, selector: str) -> None:
        self._record("verify_visible", selector)
        self._require(selector)

    def diagnostics(self, selector: str | None, artifact_name: str):
        from qa_agent.domain.ui_test import UiDiagnostics

        return UiDiagnostics(
            page_url=self.url,
            page_title="Fake page",
            matching_elements=None if selector is None else (0 if selector in self.browser.missing else 1),
            element_text=self.browser.texts.get(selector) if selector else None,
            console_errors=list(self.browser.console_errors),
        )

    def close(self) -> None:
        self.browser.closed_sessions += 1


class FakeUiBrowser:
    """In-memory UiBrowser: every selector exists and verifies unless configured otherwise."""

    def __init__(
        self,
        *,
        texts: dict[str, str] | None = None,
        missing: set[str] | None = None,
        broken_paths: set[str] | None = None,
        pages: list | None = None,
        console_errors: list[str] | None = None,
        fail_launch: Exception | None = None,
    ) -> None:
        self.texts = texts or {}
        self.missing = missing or set()
        self.broken_paths = broken_paths or set()
        self.pages = pages or []
        self.console_errors = console_errors or []
        self.fail_launch = fail_launch
        self.actions: list[tuple[str, str, str | None]] = []
        self.sessions = 0
        self.closed_sessions = 0
        self.inventory_calls = 0
        self.closed = False

    def new_session(self) -> FakeUiSession:
        if self.fail_launch:
            raise self.fail_launch
        self.sessions += 1
        return FakeUiSession(self)

    def inventory(self, start_url: str, max_pages: int) -> list:
        self.inventory_calls += 1
        return self.pages[:max_pages]

    def close(self) -> None:
        self.closed = True
