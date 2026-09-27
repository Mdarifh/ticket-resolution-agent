---
title: User login API
doc_type: api_doc
component: auth
version: v2
last_updated: 2026-03-02
tags: [login, auth, session]
---
# User login API (v2)

## POST /api/v2/login

Request body: `{"email": "...", "password": "..."}`

Responses:

- `200 OK` with `{"access_token": "...", "expires_in": 3600}` and a `refresh_token` cookie
  (`HttpOnly`, `Secure`, `SameSite=Strict`).
- `401 Unauthorized` with `{"error": "invalid_credentials"}` for a wrong email or password.
  The message does not reveal which of the two was wrong.
- `423 Locked` with `{"error": "account_locked"}` after 10 consecutive failed attempts; the lock
  lasts 15 minutes.

## POST /api/v2/logout

Revokes the current session. Returns `204 No Content`.

Test accounts in the staging environment are reset nightly at 02:00 UTC.
