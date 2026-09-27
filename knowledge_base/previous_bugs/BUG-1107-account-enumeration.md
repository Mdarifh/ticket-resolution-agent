---
title: Password reset leaked which emails are registered
doc_type: previous_bug
bug_id: BUG-1107
component: auth
severity: high
status: resolved
last_updated: 2026-03-30
tags: [password-reset, security, enumeration]
---
# BUG-1107: Password reset leaked which emails are registered

**Symptom:** `POST /api/v2/password-reset` returned `202` for registered emails and `404` for
unregistered ones, letting attackers discover valid accounts.

**Root cause:** The handler returned early with a not-found error before the generic response.
Response time also differed by about 300 ms because email sending was synchronous.

**Fix:** Always return `202` with the generic message, and send the email from a background job so
response times match.

**Regression test:** Compare status, body and response time for a registered and an unregistered
email; they must match (timing within 50 ms at the median of 20 requests).
