"""UI test cases (+ an inventory of the app's pages) -> Playwright step scripts.

The LLM sees the interactive elements actually present on the app's pages
(test ids, labels, texts) so it picks real selectors instead of guessing.
Navigation targets must be relative paths: the host always comes from
configuration.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import TestCase
from qa_agent.domain.ui_test import UiAction, UiStep, UiTestRequest
from qa_agent.tools.ui_test_tool import PageInventory


class UiStepSpec(BaseModel):
    action: UiAction
    target: str = Field(
        description="navigate: path starting with '/'. Other actions: a Playwright selector, "
        "preferably data-testid=<id> from the page inventory."
    )
    value: str | None = Field(
        description="Text to type (fill), option value or label (select), or expected text "
        "(verify_text); null otherwise."
    )
    description: str


class UiScriptSpec(BaseModel):
    test_case_id: str
    steps: list[UiStepSpec]
    skip_reason: str | None = Field(
        description="Why the test cannot be automated on these pages (e.g. the page or element "
        "does not exist, or it needs an email inbox); null when executable."
    )


class UiScriptPlan(BaseModel):
    scripts: list[UiScriptSpec]


UI_SCRIPT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You convert UI test cases into browser automation steps.\n"
            "Allowed actions: navigate, click, fill, select, verify_text, verify_visible.\n"
            "Rules:\n"
            "- Produce exactly one script per test case id.\n"
            "- Start with a navigate step. Paths are relative to the app and start with '/'; "
            "never output a scheme, hostname or full URL.\n"
            "- Only use elements listed in the page inventory. Prefer 'data-testid=<id>' "
            "selectors; otherwise use '#id' or 'text=...'.\n"
            "- End with verify_text or verify_visible steps that check the test's expected "
            "result itself: the message, value or page that proves the outcome. Never verify "
            "something that is present whatever the outcome (such as the page title or a form "
            "label); such a check cannot fail and hides bugs.\n"
            "- Inventory elements with shown_later=true are hidden until an action reveals them "
            "(error, status and confirmation messages). Use them to verify messages that appear "
            "after submitting a form.\n"
            "- If the inventory lacks the page or elements the test needs, or the test needs "
            "something a browser script cannot observe (an email inbox), set skip_reason "
            "and leave steps empty.\n"
            "- Query strings from the test data (e.g. ?token=...) may be added to paths.",
        ),
        ("human", "UI test cases (JSON):\n{test_cases}\n\nPage inventory (JSON):\n{inventory}"),
    ]
)


def build_ui_script_chain(llm: BaseChatModel) -> Runnable[dict, UiScriptPlan]:
    return structured_chain(UI_SCRIPT_PROMPT, llm, UiScriptPlan)


def generate_ui_scripts(
    llm: BaseChatModel, test_cases: list[TestCase], inventory: list[PageInventory]
) -> UiScriptPlan:
    return build_ui_script_chain(llm).invoke(
        {
            "test_cases": "[" + ",".join(tc.model_dump_json() for tc in test_cases) + "]",
            "inventory": json.dumps(
                [page.model_dump(exclude_none=True, exclude_defaults=True) for page in inventory],
                ensure_ascii=False,
            ),
        }
    )


class InvalidUiScript(ValueError):
    pass


def script_to_request(spec: UiScriptSpec) -> UiTestRequest:
    """Validate an LLM script and convert it; raises ``InvalidUiScript``."""
    for step in spec.steps:
        if step.action == "navigate":
            path = step.target.strip()
            if not path.startswith("/") or path.startswith("//") or "://" in path:
                raise InvalidUiScript(f"Navigation target must be a relative path: {step.target!r}")
    try:
        return UiTestRequest(
            test_case_id=spec.test_case_id,
            steps=[
                UiStep(action=s.action, target=s.target, value=s.value, description=s.description)
                for s in spec.steps
            ],
        )
    except ValueError as exc:  # includes pydantic.ValidationError
        raise InvalidUiScript(f"Invalid UI script: {exc}") from exc
