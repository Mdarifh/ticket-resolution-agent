"""Centralized application configuration, loaded from environment variables / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # App
    app_env: str = "local"
    log_level: str = "INFO"
    # text (human-readable) or json (one object per line, for log shippers)
    log_format: Literal["text", "json"] = "text"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Streamlit
    streamlit_api_base_url: str = "http://localhost:8000"

    # LLM
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_timeout_s: float = 60.0
    # Retries per model for transient errors (429, 5xx, timeouts), with backoff.
    llm_max_retries: int = 2
    # Models tried in order when the primary still fails (comma-separated, same provider).
    llm_fallback_models: str = ""

    # LangSmith tracing (off unless enabled AND a key is set). The LANGCHAIN_* names are
    # the legacy spellings of the same variables.
    langsmith_tracing: bool = Field(
        default=False, validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    )
    langsmith_project: str = Field(
        default="ai-qa-agent", validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
    )
    langsmith_endpoint: str | None = Field(
        default=None, validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
    )

    # PostgreSQL persistence, e.g. postgresql+psycopg://qa:qa@localhost:5432/qa_agent
    database_url: str | None = None
    database_echo: bool = False

    # Confidence / human-in-the-loop
    confidence_threshold: float = 0.7
    # Proposed actions that always need human approval (comma-separated):
    # report_bugs, report_high_severity_bugs, reclassify_failure_as_test_issue
    sensitive_actions: str = "report_high_severity_bugs,reclassify_failure_as_test_issue"

    # RAG / knowledge base
    knowledge_base_dir: str = "knowledge_base"
    embedding_provider: Literal["openai", "hashing"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "qa-knowledge"
    rag_top_k: int = 4
    rag_min_relevance: float = 0.25
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_stale_after_days: int = 365

    # API test execution (target system under test; never a production default)
    api_test_base_url: str | None = None
    api_test_allowed_hosts: str = ""  # comma-separated extra hosts; base URL host is implied
    api_test_timeout_s: float = 10.0
    api_test_max_response_bytes: int = 1_000_000
    api_test_verify_tls: bool = True
    api_test_follow_redirects: bool = False
    api_test_auth_token: SecretStr | None = None
    api_test_allow_production: bool = False

    # UI test execution with Playwright (local/test web app; never a production default)
    ui_test_base_url: str | None = None
    ui_test_allowed_hosts: str = ""  # comma-separated extra hosts; base URL host is implied
    ui_test_browser: Literal["chromium", "firefox", "webkit"] = "chromium"
    ui_test_headless: bool = True
    ui_test_step_timeout_ms: int = 5000
    ui_test_navigation_timeout_ms: int = 15000
    ui_test_max_inventory_pages: int = 5
    ui_test_screenshot_dir: str | None = None  # screenshots on failure are off by default
    ui_test_allow_production: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
