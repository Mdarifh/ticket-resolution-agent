"""Chat model factory. Every chain receives its model from here (or a test fake)."""

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from qa_agent.config import Settings, get_settings
from qa_agent.errors import LLMConfigurationError

__all__ = ["LLMConfigurationError", "get_chat_model"]


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set; configure it in .env to run LLM-backed nodes."
        )

    def build(model: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    primary = build(settings.llm_model)
    fallbacks = [m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()]
    if not fallbacks:
        return primary
    # RunnableWithFallbacks forwards with_structured_output() to every model, so chains
    # can use it like a chat model.
    return cast(BaseChatModel, primary.with_fallbacks([build(m) for m in fallbacks]))
