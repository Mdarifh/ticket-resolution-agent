"""System status: what the backend is configured to do (no secrets)."""

from fastapi import APIRouter

from qa_agent.api.schemas import SystemStatus
from qa_agent.config import get_settings
from qa_agent.observability import tracing_status

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatus)
def system_status() -> SystemStatus:
    settings = get_settings()
    warnings = []
    llm_configured = bool(settings.openai_api_key)
    if not llm_configured:
        warnings.append("OPENAI_API_KEY is not set: LLM-backed steps will fail and be reported as errors.")
    if settings.embedding_provider == "openai" and not llm_configured:
        warnings.append("Knowledge base search uses OpenAI embeddings and is unavailable without an API key.")
    if not settings.api_test_base_url:
        warnings.append("API_TEST_BASE_URL is not set: API tests will report an error.")
    if not settings.ui_test_base_url:
        warnings.append("UI_TEST_BASE_URL is not set: UI tests will report an error.")
    tracing = tracing_status()
    if settings.langsmith_tracing and not tracing.enabled:
        warnings.append(f"LangSmith tracing was requested but is off: {tracing.reason}.")
    if not settings.database_url:
        warnings.append("DATABASE_URL is not set: runs are not persisted to PostgreSQL.")
    return SystemStatus(
        app_env=settings.app_env,
        llm_configured=llm_configured,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        database_configured=bool(settings.database_url),
        api_test_base_url=settings.api_test_base_url,
        ui_test_base_url=settings.ui_test_base_url,
        confidence_threshold=settings.confidence_threshold,
        langsmith_tracing=tracing.enabled,
        langsmith_project=tracing.project if tracing.enabled else None,
        log_format=settings.log_format,
        warnings=warnings,
    )
