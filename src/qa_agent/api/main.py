"""FastAPI application entry point.

    uvicorn qa_agent.api.main:app --app-dir src --port 8000

The chat web app in ``web/`` is served at http://localhost:8000/app/.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from qa_agent.api import middleware
from qa_agent.api.dependencies import build_default_run_manager
from qa_agent.api.routers import projects, runs, system
from qa_agent.config import get_settings
from qa_agent.logging_conf import configure_logging
from qa_agent.observability import configure_tracing, flush_traces
from qa_agent.services.run_manager import RunManager

configure_logging()
configure_tracing()
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app(run_manager_factory: Callable[[], RunManager] | None = None) -> FastAPI:
    """``run_manager_factory`` builds the workflow runner on first use (tests pass fakes)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting AI QA Agent API (env=%s)", get_settings().app_env)
        yield
        manager = getattr(app.state, "run_manager", None)
        if manager is not None:
            manager.shutdown()
        flush_traces()

    app = FastAPI(
        title="AI QA Agent",
        description="Agentic AI QA system: requirement analysis, test generation, "
        "execution, root-cause analysis, and reporting.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.run_manager_factory = run_manager_factory or build_default_run_manager
    app.state.run_manager = None
    middleware.install(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    app.include_router(system.router)
    app.include_router(projects.router)
    app.include_router(runs.router)

    if WEB_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/app/")

    return app


app = create_app()
