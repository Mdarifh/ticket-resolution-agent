---
title: Authentication security testing checklist
doc_type: qa_guideline
component: auth
last_updated: 2026-04-08
tags: [security, auth, checklist]
---
# Authentication security testing checklist

Apply to login, registration and password reset features.

1. **Account enumeration**: responses (status, body, headers, timing) must be identical for
   registered and unregistered identifiers.
2. **Rate limiting**: repeated requests must be throttled; verify the limit and the `Retry-After`
   header.
3. **Token handling**: reset and verification tokens must be single-use, expire, be unguessable,
   and be invalidated when a newer token is issued.
4. **Session handling**: changing a password must revoke existing sessions.
5. **Transport**: tokens must never appear in logs, analytics events or `Referer` headers.
6. **Input handling**: test SQL/NoSQL injection strings and overly long input in every field.
