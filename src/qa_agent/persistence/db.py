"""Engine, session factory and declarative base.

Configured from ``DATABASE_URL`` (e.g. ``postgresql+psycopg://qa:qa@localhost:5432/qa_agent``).
Column types are portable (JSONB on PostgreSQL, JSON elsewhere) so the
repositories can also be exercised against SQLite in tests.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import JSON, Engine, MetaData, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from qa_agent.config import Settings, get_settings

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

JsonType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DatabaseNotConfiguredError(RuntimeError):
    pass


def create_db_engine(url: str, *, echo: bool = False) -> Engine:
    if url.startswith("sqlite"):
        in_memory = ":memory:" in url or url.rstrip("/") == "sqlite:"
        engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False},
            **({"poolclass": StaticPool} if in_memory else {}),
        )

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):  # SQLite ignores FKs by default
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

        return engine
    return create_engine(url, echo=echo, pool_pre_ping=True)


def engine_from_settings(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    if not settings.database_url:
        raise DatabaseNotConfiguredError("DATABASE_URL is not set")
    return create_db_engine(settings.database_url, echo=settings.database_echo)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One unit of work: commit on success, roll back on any error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
