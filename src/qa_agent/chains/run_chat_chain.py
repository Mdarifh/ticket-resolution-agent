"""Answer free-form questions about one QA run ("Why did TC-005 fail?").

The run's workflow state is condensed into a JSON context and sent with the
question and the recent conversation. Answers are plain text (light Markdown).
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

MAX_CONTEXT_CHARS = 60_000
MAX_HISTORY_TURNS = 8

SYSTEM_PROMPT = (
    "You are the assistant of an AI QA agent. You answer questions about ONE QA run: the "
    "requirement, its analysis, the generated test cases and plan, execution results, "
    "failure analysis, root cause analysis, bug reports, confidence score, human review and "
    "final report. The run data is given below as JSON.\n"
    "Rules:\n"
    "- Answer only from the run data. If it does not contain the answer, say so plainly.\n"
    "- Cite test case ids (TC-...) and bug ids (BR-...) when you refer to them.\n"
    "- Be concise: short paragraphs or bullet lists, Markdown allowed.\n"
    "- Answer in the language the user writes in (for example Hindi/Hinglish or English).\n"
    "- Never invent results, status codes or evidence.\n"
    "- If run.metadata.mode is \"bug_verification\", the run verifies a service desk bug ticket "
    "(run.metadata.bug_ticket) and run.verdict says whether the bug was confirmed, not reproduced "
    "or inconclusive. Explain the verdict with the test evidence when asked, and write service desk "
    "replies when asked.\n\n"
    "Run data (JSON):\n{context}"
)

_CONTEXT_KEYS = (
    "requirement_analysis",
    "test_cases",
    "test_plan",
    "execution_results",
    "failures",
    "root_cause_analysis",
    "bug_reports",
    "confidence_score",
    "human_review",
    "final_report",
    "errors",
)


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude={"markdown"})
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def build_run_context(run: dict[str, Any], state: dict[str, Any]) -> str:
    """JSON context for the prompt; the least important parts are trimmed to fit."""
    context: dict[str, Any] = {"run": run}
    for key in _CONTEXT_KEYS:
        if state.get(key):
            context[key] = _plain(state[key])
    text = json.dumps(context, default=str, ensure_ascii=False)
    # Drop the bulkiest, least question-relevant parts first.
    for key in ("test_plan", "final_report", "execution_results"):
        if len(text) <= MAX_CONTEXT_CHARS:
            break
        context.pop(key, None)
        text = json.dumps(context, default=str, ensure_ascii=False)
    return text[:MAX_CONTEXT_CHARS]


def _content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def answer_question(
    llm: BaseChatModel,
    context: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """``history`` holds earlier turns as {"role": "user"|"assistant", "content": ...}."""
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT.replace("{context}", context))]
    for turn in (history or [])[-MAX_HISTORY_TURNS * 2 :]:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        messages.append(cls(turn.get("content", "")))
    messages.append(HumanMessage(question))
    return _content_text(llm.invoke(messages)).strip()
