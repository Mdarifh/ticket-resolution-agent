import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

import qa_agent.persistence.models  # noqa: F401  (registers tables)
from qa_agent.persistence.db import Base, create_db_engine, session_factory

from qa_agent.rag.embeddings import HashingEmbeddings
from qa_agent.rag.knowledge_base import QAKnowledgeBase
from qa_agent.tools.ui_test_tool import PageInventory, UiTestConfig, UiTestRunner
from tests.fakes import FakeStructuredChatModel, FakeUiBrowser

FIXTURES = Path(__file__).parent / "fixtures"
KNOWLEDGE_BASE_DIR = Path(__file__).parents[1] / "knowledge_base"

PASSWORD_RESET_REQUIREMENT = "User should be able to reset password using registered email."


@pytest.fixture(autouse=True)
def _planner_ignores_local_env(monkeypatch):
    """The planner reroutes api/ui tests based on configured targets; keep a developer's
    .env (e.g. UI_TEST_BASE_URL) from changing graph test plans."""
    from qa_agent.config import Settings

    monkeypatch.setattr(
        "qa_agent.graph.nodes.test_planner.get_settings", lambda: Settings(_env_file=None)
    )


def load_fixture(*parts: str):
    return json.loads(FIXTURES.joinpath(*parts).read_text(encoding="utf-8"))


def password_reset_responses() -> dict[str, list[dict]]:
    """Canned LLM output per structured-output schema, for the password reset example."""
    return {
        "RequirementAnalysis": [load_fixture("password_reset", "requirement_analysis.json")],
        "TestSuite": [load_fixture("password_reset", "test_suite.json")],
        "TestPlanDraft": [load_fixture("password_reset", "test_plan_draft.json")],
        "RootCauseDraft": [load_fixture("password_reset", "root_cause_draft.json")],
        "ApiRequestPlan": [load_fixture("password_reset", "api_request_plan.json")],
        "UiScriptPlan": [load_fixture("password_reset", "ui_script_plan.json")],
        "BugNarrativeDraft": [load_fixture("password_reset", "bug_narrative_draft.json")],
    }


UI_BASE_URL = "http://127.0.0.1:8765"

# Texts the password-reset UI scripts verify; FakeUiBrowser passes them by default.
PASSING_UI_TEXTS = {
    "data-testid=reset-success": "Your password has been reset. Log in",
    "data-testid=login-message": "Welcome back!",
    "data-testid=email-error": "Please enter a valid email address.",
}


def demo_inventory() -> list[PageInventory]:
    """Inventory crawled from demo_app/ (regenerate if the demo pages change)."""
    return [PageInventory.model_validate(p) for p in load_fixture("password_reset", "ui_inventory.json")]


def fake_ui_runner(browser: FakeUiBrowser | None = None, **config) -> UiTestRunner:
    """Real UiTestRunner logic over an in-memory browser."""
    browser = browser or FakeUiBrowser(texts=dict(PASSING_UI_TEXTS), pages=demo_inventory())
    return UiTestRunner(UiTestConfig(**{"base_url": UI_BASE_URL, **config}), browser)


_THROWAWAY_KNOWLEDGE_BASES: list[QAKnowledgeBase] = []


def make_knowledge_base(**kwargs) -> QAKnowledgeBase:
    """Empty throwaway knowledge base with offline embeddings; dropped after the test."""
    knowledge_base = QAKnowledgeBase.in_memory(HashingEmbeddings(), **kwargs)
    _THROWAWAY_KNOWLEDGE_BASES.append(knowledge_base)
    return knowledge_base


@pytest.fixture(autouse=True)
def _drop_throwaway_knowledge_bases():
    yield
    while _THROWAWAY_KNOWLEDGE_BASES:
        _THROWAWAY_KNOWLEDGE_BASES.pop().drop()


@lru_cache
def repo_knowledge_base() -> QAKnowledgeBase:
    """The repository's knowledge_base/ indexed once per test session. Treat as read-only."""
    knowledge_base = QAKnowledgeBase.in_memory(HashingEmbeddings())
    knowledge_base.index_directory(KNOWLEDGE_BASE_DIR)
    return knowledge_base


@pytest.fixture
def fake_llm() -> FakeStructuredChatModel:
    return FakeStructuredChatModel(responses=password_reset_responses())


@pytest.fixture
def knowledge_base() -> QAKnowledgeBase:
    return repo_knowledge_base()


# --- database ------------------------------------------------------------------

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
ALEMBIC_INI = Path(__file__).parents[1] / "alembic.ini"


def alembic_config(url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False  # keep pytest's logging setup
    return config


@pytest.fixture(scope="session")
def postgres_engine():
    """PostgreSQL from TEST_DATABASE_URL, schema built by the Alembic migrations."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    config = alembic_config(TEST_DATABASE_URL)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    engine = create_db_engine(TEST_DATABASE_URL)
    yield engine
    engine.dispose()


@pytest.fixture(params=["sqlite", "postgresql"])
def db_engine(request):
    """Every repository test runs on SQLite, and on PostgreSQL when configured."""
    if request.param == "sqlite":
        engine = create_db_engine("sqlite://")
        Base.metadata.create_all(engine)
        yield engine
        engine.dispose()
        return
    engine = request.getfixturevalue("postgres_engine")
    yield engine
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} CASCADE"))


@pytest.fixture
def db_sessions(db_engine):
    return session_factory(db_engine)
