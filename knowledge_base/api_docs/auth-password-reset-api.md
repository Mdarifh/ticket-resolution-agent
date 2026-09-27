---
title: Password reset API
doc_type: api_doc
component: auth
version: v2
last_updated: 2026-06-12
tags: [password-reset, auth, email]
---
# Password reset API (v2)

## POST /api/v2/password-reset

Starts a password reset for the account that owns `email`.

Request body:

```json
{ "email": "registered.user@example.com" }
```

Responses:

- `202 Accepted` with body `{"message": "If the email is registered, a reset link has been sent."}`.
  The same status and body are returned whether or not the email is registered, to prevent
  account enumeration.
- `400 Bad Request` with `{"error": "invalid_email"}` when `email` is missing or not a valid address.
- `429 Too Many Requests` after 5 requests for the same email within one hour. Includes a
  `Retry-After` header in seconds.

Email lookup is case-insensitive and ignores leading/trailing whitespace.

## POST /api/v2/password-reset/confirm

Sets a new password using the token from the reset email.

Request body:

```json
{ "token": "<reset token>", "new_password": "N3w-Passw0rd!" }
```

Responses:

- `200 OK` when the password was changed. All existing sessions for the account are revoked.
- `400 Bad Request` with `{"error": "token_expired"}` when the token is older than 30 minutes.
- `400 Bad Request` with `{"error": "token_used"}` when the token was already used.
- `400 Bad Request` with `{"error": "token_invalid"}` for unknown or malformed tokens.
- `422 Unprocessable Entity` with `{"error": "weak_password"}` when the password fails the policy:
  minimum 12 characters, at least one letter, one digit and one symbol, and not equal to any of
  the last 5 passwords.

Tokens are single-use, 32 bytes of random data, URL-safe base64 encoded.
