from pathlib import Path

import httpx
import pytest

from qa_agent.tools.demo_backend import (
    DEMO_EMAIL,
    DEMO_PASSWORD,
    GENERIC_RESET_MESSAGE,
    LOCK_DURATION_S,
    RESET_TOKEN_TTL_S,
    DemoAuthService,
)
from qa_agent.tools.demo_server import serve_demo_app

DEMO_DIR = Path(__file__).parents[2] / "demo_app"

NEW_PASSWORD = "Br4nd-New-Pass!"


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service(clock):
    return DemoAuthService(clock=clock, base_url="http://demo")


def test_seeded_account_logs_in_and_session_resolves(service):
    login = service.login({"email": " Registered.User@Example.com ", "password": DEMO_PASSWORD})

    assert login.status == 200
    assert login.body["expires_in"] == 3600
    assert "HttpOnly" in login.headers["Set-Cookie"]
    assert service.me(login.body["access_token"]).body == {"email": DEMO_EMAIL}


def test_wrong_password_and_unknown_email_look_the_same(service):
    wrong = service.login({"email": DEMO_EMAIL, "password": "nope"})
    unknown = service.login({"email": "ghost@example.com", "password": "nope"})

    assert (wrong.status, wrong.body) == (unknown.status, unknown.body) == (
        401,
        {"error": "invalid_credentials"},
    )


@pytest.mark.parametrize("body", [{}, {"email": "", "password": ""}, {"email": DEMO_EMAIL}])
def test_missing_credentials_are_invalid_credentials(service, body):
    response = service.login(body)

    assert (response.status, response.body) == (401, {"error": "invalid_credentials"})


def test_account_locks_after_ten_failures_for_fifteen_minutes(service, clock):
    for _ in range(10):
        service.login({"email": DEMO_EMAIL, "password": "nope"})

    assert service.login({"email": DEMO_EMAIL, "password": DEMO_PASSWORD}).status == 423
    clock.now += LOCK_DURATION_S + 1
    assert service.login({"email": DEMO_EMAIL, "password": DEMO_PASSWORD}).status == 200


def test_register_validates_and_rejects_duplicates(service):
    assert service.register({"email": "bad", "password": NEW_PASSWORD}).status == 400
    assert service.register({"email": "a@example.com", "password": "short"}).status == 422
    assert service.register({"email": "A@example.com", "password": NEW_PASSWORD}).status == 201
    assert service.register({"email": "a@example.com", "password": NEW_PASSWORD}).status == 409
    assert service.login({"email": "a@example.com", "password": NEW_PASSWORD}).status == 200


def test_logout_revokes_the_session(service):
    token = service.login({"email": DEMO_EMAIL, "password": DEMO_PASSWORD}).body["access_token"]

    assert service.logout(token).status == 204
    assert service.me(token).status == 401


def test_reset_request_is_generic_and_only_mails_registered_users(service):
    known = service.request_password_reset({"email": DEMO_EMAIL.upper()})
    unknown = service.request_password_reset({"email": "ghost@example.com"})

    assert (known.status, known.body) == (unknown.status, unknown.body) == (
        202,
        {"message": GENERIC_RESET_MESSAGE},
    )
    assert [m["to"] for m in service.outbox] == [DEMO_EMAIL]
    assert service.outbox[0]["link"].startswith("http://demo/reset-password.html?token=")
    assert service.request_password_reset({"email": "nope"}).body == {"error": "invalid_email"}


def test_reset_requests_are_rate_limited_per_email(service):
    for _ in range(5):
        assert service.request_password_reset({"email": DEMO_EMAIL}).status == 202

    limited = service.request_password_reset({"email": DEMO_EMAIL})
    assert limited.status == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_reset_confirm_changes_password_once_and_revokes_sessions(service):
    session = service.login({"email": DEMO_EMAIL, "password": DEMO_PASSWORD}).body["access_token"]
    service.request_password_reset({"email": DEMO_EMAIL})
    token = service.outbox[-1]["link"].split("token=")[1]

    assert service.confirm_password_reset({"token": token, "new_password": "weak"}).status == 422
    assert service.confirm_password_reset({"token": token, "new_password": NEW_PASSWORD}).status == 200
    assert service.confirm_password_reset({"token": token, "new_password": NEW_PASSWORD}).body == {
        "error": "token_used"
    }
    assert service.me(session).status == 401
    assert service.login({"email": DEMO_EMAIL, "password": DEMO_PASSWORD}).status == 401
    assert service.login({"email": DEMO_EMAIL, "password": NEW_PASSWORD}).status == 200


@pytest.mark.parametrize(
    ("token", "error"),
    [("used-token-456", "token_used"), ("expired-token-789", "token_expired"), ("x", "token_invalid")],
)
def test_seeded_reset_tokens(service, token, error):
    response = service.confirm_password_reset({"token": token, "new_password": NEW_PASSWORD})

    assert (response.status, response.body) == (400, {"error": error})


def test_reset_token_expires_after_thirty_minutes(service, clock):
    clock.now += RESET_TOKEN_TTL_S + 1

    response = service.confirm_password_reset({"token": "valid-token-123", "new_password": NEW_PASSWORD})

    assert response.body == {"error": "token_expired"}


def test_users_persist_in_a_sqlite_file(tmp_path):
    db = str(tmp_path / "users.db")
    DemoAuthService(db).register({"email": "kept@example.com", "password": NEW_PASSWORD})

    assert DemoAuthService(db).login({"email": "kept@example.com", "password": NEW_PASSWORD}).status == 200


def test_server_routes_api_requests_and_serves_pages():
    with serve_demo_app(DEMO_DIR) as base_url, httpx.Client(base_url=base_url) as http:
        assert http.get("/login.html").status_code == 200
        login = http.post("/api/v2/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
        assert login.status_code == 200
        token = login.json()["access_token"]
        assert http.get("/api/v2/me", headers={"Authorization": f"Bearer {token}"}).json() == {
            "email": DEMO_EMAIL
        }
        assert http.post("/api/v2/logout", headers={"Authorization": f"Bearer {token}"}).status_code == 204
        bad_json = http.post(
            "/api/v2/login", content=b"{not json", headers={"Content-Type": "application/json"}
        )
        assert bad_json.status_code == 400
        assert http.get("/api/v2/unknown").status_code == 404
        http.post("/api/v2/password-reset", json={"email": DEMO_EMAIL})
        mails = http.get("/api/demo/outbox", params={"email": DEMO_EMAIL}).json()["emails"]
        assert mails[0]["link"].startswith(f"{base_url}/reset-password.html?token=")
