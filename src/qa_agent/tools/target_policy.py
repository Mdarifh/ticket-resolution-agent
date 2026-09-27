"""Shared rules for which system-under-test URLs the API and UI tools may reach.

There is never a default target: a base URL must be configured. Requests stay
on the base URL's host plus explicitly allowed hosts, use http(s) only, and
are refused when the app runs in production unless explicitly overridden.
"""

from urllib.parse import urljoin, urlsplit


class UnsafeRequestError(ValueError):
    """The request targets something the configuration does not allow."""


def permitted_hosts(base_url: str | None, extra_hosts: list[str]) -> set[str]:
    hosts = {h.strip().lower() for h in extra_hosts if h.strip()}
    if base_url:
        host = urlsplit(base_url).hostname
        if host:
            hosts.add(host.lower())
    return hosts


def split_hosts(value: str) -> list[str]:
    """Parse a comma-separated hosts setting."""
    return [h.strip() for h in value.split(",") if h.strip()]


def check_base_url(
    base_url: str | None, *, setting: str, app_env: str, allow_production: bool, override: str
) -> None:
    if not base_url:
        raise UnsafeRequestError(
            f"Test base URL is not configured; set {setting} to a test environment"
        )
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeRequestError(f"{setting} must be an http(s) URL: {base_url!r}")
    if app_env.lower() in {"prod", "production"} and not allow_production:
        raise UnsafeRequestError(
            f"Tests are disabled when APP_ENV is production "
            f"(set {override}=true to override deliberately)"
        )


def check_target(url: str, hosts: set[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeRequestError(f"Only http(s) URLs are allowed, got {url!r}")
    host = (parts.hostname or "").lower()
    if host not in hosts:
        raise UnsafeRequestError(
            f"Host {host!r} is not an allowed test host (allowed: {sorted(hosts)})"
        )


def resolve(base_url: str, url: str, hosts: set[str]) -> str:
    """Resolve ``url`` (relative path or absolute) against ``base_url`` and check it."""
    parts = urlsplit(url)
    resolved = (
        url if parts.scheme or parts.netloc else urljoin(base_url.rstrip("/") + "/", url.lstrip("/"))
    )
    check_target(resolved, hosts)
    return resolved
