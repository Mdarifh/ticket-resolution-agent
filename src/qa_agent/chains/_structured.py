"""Shared helper: prompt -> LLM with Pydantic structured output (+ one retry)."""

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError


def structured_chain[T: BaseModel](
    prompt: ChatPromptTemplate, llm: BaseChatModel, schema: type[T]
) -> Runnable[dict, T]:
    """Output is validated against ``schema``; a malformed response is retried once."""
    return (prompt | llm.with_structured_output(schema)).with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException),
        stop_after_attempt=2,
        wait_exponential_jitter=False,
    )
