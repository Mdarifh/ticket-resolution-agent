"""Auth backend for the local demo app (the system under test in demos).

Implements the knowledge base's v2 login and password reset API documents, plus
registration and a "current user" endpoint so the demo has real accounts:

    POST /api/v2/register                 {"email", "password"}      -> 201 | 400 | 409 | 422
    POST /api/v2/login                    {"email", "password"}      -> 200 | 401 | 423
    POST /api/v2/logout                   Authorization: Bearer ...  -> 204
    GET  /api/v2/me                       Authorization: Bearer ...  -> 200 | 401
    POST /api/v2/password-reset           {"email"}                  -> 202 | 400 | 429
    POST /api/v2/password-reset/confirm   {"token", "new_password"}  -> 200 | 400 | 422
    GET  /api/demo/outbox?email=...       reset emails "sent" so far (demo only)

Users live in SQLite (a file, or in memory for tests); sessions, reset tokens,
lockouts and rate limits are in memory. Not for production use.
"""

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
GENERIC_RESET_MESSAGE = "If the email is registered, a reset link has been sent."

SESSION_TTL_S = 3600
LOCK_AFTER_FAILURES = 10
LOCK_DURATION_S = 15 * 60
RESET_TOKEN_TTL_S = 30 * 60
RESET_REQUESTS_PER_HOUR = 5

# Seeded demo data. The fixed tokens keep the demo's reset links (and tests) working.
DEMO_EMAIL = "registered.user@example.com"
DEMO_PASSWORD = "N3w-Passw0rd!"
DEMO_TOKENS = {"valid-token-123": "valid", "used-token-456": "used", "expired-token-789": "expired"}

_PBKDF2_ITERATIONS = 200_000


@dataclass
class ApiResponse:
    status: int
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


def _normalize(email: object) -> str:
    return email.strip().lower() if isinstance(email, str) else ""


def password_policy_ok(password: object) -> bool:
    return (
        isinstance(password, str)
        and len(password) >= 12
        and re.search(r"[A-Za-z]", password) is not None
        and re.search(r"\d", password) is not None
        and re.search(r"[^A-Za-z0-9]", password) is not None
    )


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, _ = stored.split("$", 1)
    return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), stored)


def _new_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


@dataclass
class _ResetToken:
    email: str
    created_at: float
    used: bool = False


class DemoAuthService:
    """Thread-safe: the demo server handles requests on several threads."""

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        base_url: str = "",
        clock: Callable[[], float] = time.time,
        seed: bool = True,
    ) -> None:
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password_hash TEXT NOT NULL)"
        )
        self._db.commit()
        self._lock = threading.Lock()
        self._clock = clock
        self.base_url = base_url
        self._sessions: dict[str, tuple[str, float]] = {}  # token -> (email, expires_at)
        self._failures: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}
        self._reset_tokens: dict[str, _ResetToken] = {}
        self._reset_requests: dict[str, list[float]] = {}
        self.outbox: list[dict[str, str]] = []
        if seed:
            self._seed()

    def _seed(self) -> None:
        if not self._user_hash(DEMO_EMAIL):
            self._set_password(DEMO_EMAIL, DEMO_PASSWORD, insert=True)
        now = self._clock()
        for token, state in DEMO_TOKENS.items():
            created = now - RESET_TOKEN_TTL_S - 1 if state == "expired" else now
            self._reset_tokens[token] = _ResetToken(DEMO_EMAIL, created, used=state == "used")

    # --- storage ---------------------------------------------------------------------

    def _user_hash(self, email: str) -> str | None:
        row = self._db.execute("SELECT password_hash FROM users WHERE email = ?", (email,)).fetchone()
        return row[0] if row else None

    def _set_password(self, email: str, password: str, *, insert: bool = False) -> None:
        sql = (
            "INSERT INTO users (password_hash, email) VALUES (?, ?)"
            if insert
            else "UPDATE users SET password_hash = ? WHERE email = ?"
        )
        self._db.execute(sql, (_hash_password(password), email))
        self._db.commit()

    # --- endpoints -------------------------------------------------------------------

    def register(self, body: dict[str, Any]) -> ApiResponse:
        email, password = _normalize(body.get("email")), body.get("password")
        if not EMAIL_RE.match(email):
            return ApiResponse(400, {"error": "invalid_email"})
        if not password_policy_ok(password):
            return ApiResponse(422, {"error": "weak_password"})
        with self._lock:
            if self._user_hash(email):
                return ApiResponse(409, {"error": "email_taken"})
            self._set_password(email, password, insert=True)
        return ApiResponse(201, {"email": email})

    def login(self, body: dict[str, Any]) -> ApiResponse:
        email, password = _normalize(body.get("email")), body.get("password")
        if not email or not isinstance(password, str) or not password:
            # The login API document defines only 200/401/423: missing credentials are
            # simply wrong credentials (and do not count towards the lockout).
            return ApiResponse(401, {"error": "invalid_credentials"})
        with self._lock:
            now = self._clock()
            if self._locked_until.get(email, 0) > now:
                return ApiResponse(423, {"error": "account_locked"})
            stored = self._user_hash(email)
            if not stored or not _verify_password(password, stored):
                self._failures[email] = self._failures.get(email, 0) + 1
                if self._failures[email] >= LOCK_AFTER_FAILURES:
                    self._locked_until[email] = now + LOCK_DURATION_S
                    self._failures[email] = 0
                return ApiResponse(401, {"error": "invalid_credentials"})
            self._failures.pop(email, None)
            token = _new_token()
            self._sessions[token] = (email, now + SESSION_TTL_S)
        cookie = f"refresh_token={_new_token()}; HttpOnly; Secure; SameSite=Strict; Path=/api"
        return ApiResponse(
            200, {"access_token": token, "expires_in": SESSION_TTL_S}, {"Set-Cookie": cookie}
        )

    def logout(self, bearer: str | None) -> ApiResponse:
        with self._lock:
            if bearer:
                self._sessions.pop(bearer, None)
        return ApiResponse(204)

    def me(self, bearer: str | None) -> ApiResponse:
        with self._lock:
            session = self._sessions.get(bearer or "")
            if not session or session[1] <= self._clock():
                return ApiResponse(401, {"error": "unauthorized"})
        return ApiResponse(200, {"email": session[0]})

    def request_password_reset(self, body: dict[str, Any]) -> ApiResponse:
        email = _normalize(body.get("email"))
        if not EMAIL_RE.match(email):
            return ApiResponse(400, {"error": "invalid_email"})
        with self._lock:
            now = self._clock()
            recent = [t for t in self._reset_requests.get(email, []) if now - t < 3600]
            if len(recent) >= RESET_REQUESTS_PER_HOUR:
                retry_after = int(3600 - (now - recent[0])) + 1
                self._reset_requests[email] = recent
                return ApiResponse(
                    429, {"error": "too_many_requests"}, {"Retry-After": str(retry_after)}
                )
            self._reset_requests[email] = [*recent, now]
            if self._user_hash(email):
                token = _new_token()
                self._reset_tokens[token] = _ResetToken(email, now)
                link = f"{self.base_url}/reset-password.html?token={token}"
                self.outbox.append({"to": email, "subject": "Reset your password", "link": link})
        return ApiResponse(202, {"message": GENERIC_RESET_MESSAGE})

    def confirm_password_reset(self, body: dict[str, Any]) -> ApiResponse:
        token, new_password = body.get("token"), body.get("new_password")
        with self._lock:
            record = self._reset_tokens.get(token) if isinstance(token, str) else None
            if record is None:
                return ApiResponse(400, {"error": "token_invalid"})
            if record.used:
                return ApiResponse(400, {"error": "token_used"})
            if self._clock() - record.created_at > RESET_TOKEN_TTL_S:
                return ApiResponse(400, {"error": "token_expired"})
            if not password_policy_ok(new_password):
                return ApiResponse(422, {"error": "weak_password"})
            record.used = True
            self._set_password(record.email, new_password)
            self._sessions = {t: s for t, s in self._sessions.items() if s[0] != record.email}
            self._failures.pop(record.email, None)
            self._locked_until.pop(record.email, None)
        return ApiResponse(200, {"message": "Your password has been reset."})

    def outbox_for(self, email: str) -> ApiResponse:
        wanted = _normalize(email)
        with self._lock:
            return ApiResponse(200, {"emails": [m for m in self.outbox if m["to"] == wanted]})
