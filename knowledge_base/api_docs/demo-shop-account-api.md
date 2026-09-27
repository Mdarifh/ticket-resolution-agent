---
title: Demo Shop account API (local demo environment)
doc_type: api_doc
component: auth
version: v2
last_updated: 2026-09-27
tags: [login, registration, session, password-reset, demo]
---
# Demo Shop account API (local demo environment)

The local Demo Shop (`python -m qa_agent.tools.demo_server`, default `http://127.0.0.1:8765`)
serves both the web pages and this JSON API from the same base URL. Login, logout and password
reset follow the v2 login and password reset API documents; this document adds the endpoints
that exist only for accounts in the demo.

Seeded account: `registered.user@example.com` with password `N3w-Passw0rd!`.

## POST /api/v2/register

Request body: `{"email": "...", "password": "..."}`

- `201 Created` with `{"email": "<normalized email>"}`. Emails are stored lower-case and trimmed.
- `400 Bad Request` with `{"error": "invalid_email"}`.
- `409 Conflict` with `{"error": "email_taken"}` when the email is already registered.
- `422 Unprocessable Entity` with `{"error": "weak_password"}` when the password fails the policy
  (minimum 12 characters, at least one letter, one digit and one symbol).

## GET /api/v2/me

Header: `Authorization: Bearer <access_token from login>`.

- `200 OK` with `{"email": "..."}` for a valid, unexpired session.
- `401 Unauthorized` with `{"error": "unauthorized"}` otherwise, including after logout or a
  password reset.

## POST /api/v2/logout

Header: `Authorization: Bearer <access_token>`. Always returns `204 No Content`.

## GET /api/demo/outbox?email=...

The demo sends no real email. Returns `{"emails": [{"to", "subject", "link"}]}` with the password
reset emails "sent" to that address. The web page `/outbox.html` shows the same list.

## Web pages

`/login.html`, `/signup.html`, `/forgot-password.html`, `/reset-password.html?token=...`,
`/outbox.html` and `/dashboard.html` (signed-in area; redirects to login without a session).
After a successful login the page shows "Welcome back!" and opens `/dashboard.html`.

## Known differences from staging

- The password history rule (not equal to the last 5 passwords) is not enforced.
- Sessions, reset tokens, lockouts and rate limits are kept in memory and reset when the
  demo server restarts. Registered users persist in `data/demo_users.db`.
- Demo reset tokens: `valid-token-123` (valid for 30 minutes after server start),
  `used-token-456` (already used), `expired-token-789` (expired).
