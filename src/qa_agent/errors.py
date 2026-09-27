"""Exceptions shared across layers (kept dependency-free to avoid import cycles)."""


class LLMConfigurationError(RuntimeError):
    """Raised when no LLM or embedding credentials are configured."""
