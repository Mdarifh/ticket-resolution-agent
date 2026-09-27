"""Secret redaction for logs and traces.

Two layers:

* exact values of the configured secrets (OpenAI key, LangSmith key, API test
  token, database password) are replaced wherever they appear;
* well-known credential shapes are masked even when they are not configured
  here (``sk-...`` / ``lsv2_...`` keys, ``Bearer`` tokens, passwords in URLs,
  ``api_key=...``-style pairs).

``redact`` walks dicts/lists/Pydantic models for trace payloads and also blanks
credential-carrying keys (``Authorization``, ``Cookie``, ...). Test data such
as a reset token or a test user's password is deliberately *not* treated as a
secret: it is what a failed test needs to be debugged.
"""

import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, SecretStr

MASK = "***"

SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api_key",
        "apikey",
        "api-key",
        "openai_api_key",
        "langsmith_api_key",
        "api_test_auth_token",
        "auth_token",
        "access_token",
        "refresh_token",
        "client_secret",
        "secret",
    }
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"), "sk-" + MASK),
    (re.compile(r"\blsv2_[A-Za-z0-9_]{12,}"), "lsv2_" + MASK),
    (re.compile(r"\bls__[A-Za-z0-9]{12,}"), "ls__" + MASK),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-~+/=]{8,}"), r"\1 " + MASK),
    (re.compile(r"([a-z][a-z0-9+.\-]*://[^:/\s@]+):[^@\s/]+@"), r"\1:" + MASK + "@"),
    (
        re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|authorization)"
                   r"(\s*[=:]\s*[\"']?)([^\s\"',}&]+)"),
        r"\1\2" + MASK,
    ),
]

_MIN_SECRET_LENGTH = 6  # shorter values would mask ordinary words


@lru_cache(maxsize=1)
def _configured_secrets() -> tuple[str, ...]:
    from qa_agent.config import get_settings

    settings = get_settings()
    values: list[str] = []
    for secret in (settings.openai_api_key, settings.langsmith_api_key, settings.api_test_auth_token):
        if isinstance(secret, SecretStr) and secret.get_secret_value():
            values.append(secret.get_secret_value())
    if settings.database_url:
        password = urlsplit(settings.database_url.replace("+psycopg", "")).password
        if password:
            values.append(password)
    # Longest first, so a secret containing another is masked whole.
    return tuple(sorted((v for v in values if len(v) >= _MIN_SECRET_LENGTH), key=len, reverse=True))


def reset_cache() -> None:
    """Re-read the configured secrets (after settings change, e.g. in tests)."""
    _configured_secrets.cache_clear()


def redact_text(text: str) -> str:
    for secret in _configured_secrets():
        text = text.replace(secret, MASK)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact(value: Any, *, _depth: int = 0) -> Any:
    """A copy of ``value`` safe to log or trace."""
    if _depth > 12:
        return value
    if isinstance(value, SecretStr):
        return MASK
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            k: MASK if isinstance(k, str) and k.lower() in SECRET_KEYS and v not in (None, "") else redact(v, _depth=_depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v, _depth=_depth + 1) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def safe_url(url: str | None) -> str | None:
    """URL for logs: credentials masked, query string dropped (it may carry tokens)."""
    if not url:
        return url
    return redact_text(url.split("?", 1)[0])
