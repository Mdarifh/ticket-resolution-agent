"""Shared FastAPI dependencies.

The run manager (graph, checkpointer, optional database) is built on first
use, so the app imports and serves /health without an LLM key or database.
"""

import logging
import threading
from collections.abc import Callable

from fastapi import Request
from langchain_core.language_models import BaseChatModel

from qa_agent.chains import get_chat_model
from qa_agent.config import Settings, get_settings
from qa_agent.graph import build_qa_graph
from qa_agent.observability import redact_text
from qa_agent.persistence.db import engine_from_settings, session_factory
from qa_agent.persistence.service import PersistenceService
from qa_agent.services.run_manager import RunManager
from qa_agent.services.run_service import RunService

logger = logging.getLogger(__name__)

_build_lock = threading.Lock()


def build_default_run_manager(settings: Settings | None = None) -> RunManager:
    settings = settings or get_settings()
    persistence = None
    if settings.database_url:
        persistence = PersistenceService(session_factory(engine_from_settings(settings)))
        logger.info("Persistence enabled: %s", redact_text(settings.database_url))
    else:
        logger.info("Persistence disabled (DATABASE_URL is not set)")
    return RunManager(RunService(build_qa_graph(), persistence))


def get_run_manager(request: Request) -> RunManager:
    state = request.app.state
    if getattr(state, "run_manager", None) is None:
        with _build_lock:
            if getattr(state, "run_manager", None) is None:
                factory: Callable[[], RunManager] = state.run_manager_factory
                state.run_manager = factory()
    return state.run_manager


def get_chat_model_factory() -> Callable[[], BaseChatModel]:
    """Builds the chat model for run Q&A lazily (so a missing key is a 503, not a crash);
    tests override this dependency with a fake."""
    return get_chat_model
