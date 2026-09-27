"""Alembic migrations against PostgreSQL (skipped unless TEST_DATABASE_URL is set)."""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect

from qa_agent.persistence.db import Base
from tests.conftest import TEST_DATABASE_URL, alembic_config

pytestmark = pytest.mark.integration

TABLES = {
    "projects",
    "requirements",
    "test_cases",
    "test_runs",
    "test_results",
    "failures",
    "bug_reports",
    "agent_runs",
    "human_reviews",
}


@pytest.fixture
def engine(postgres_engine):
    yield postgres_engine
    command.upgrade(alembic_config(TEST_DATABASE_URL), "head")  # leave the schema migrated


def test_upgrade_creates_every_table(engine):
    assert TABLES <= set(inspect(engine).get_table_names())


def test_schema_matches_the_orm_models(engine):
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_downgrade_and_upgrade_round_trip(engine):
    config = alembic_config(TEST_DATABASE_URL)

    command.downgrade(config, "base")
    assert not TABLES & set(inspect(engine).get_table_names())

    command.upgrade(config, "head")
    assert TABLES <= set(inspect(engine).get_table_names())


def test_foreign_keys_cascade_from_agent_runs(engine):
    inspector = inspect(engine)
    for table in ("test_cases", "test_runs", "bug_reports", "human_reviews"):
        [fk] = [f for f in inspector.get_foreign_keys(table) if f["referred_table"] == "agent_runs"]
        assert fk["options"].get("ondelete") == "CASCADE", table
