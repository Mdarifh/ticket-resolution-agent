---
title: Password reset token could be used more than once
doc_type: previous_bug
bug_id: BUG-1042
component: auth
severity: high
status: resolved
last_updated: 2026-02-14
tags: [password-reset, token, regression]
---
# BUG-1042: Password reset token could be used more than once

**Symptom:** After a successful reset, calling `POST /api/v2/password-reset/confirm` again with the
same token returned `200 OK` and changed the password again.

**Root cause:** The token was marked as used in a cache write that happened after the password
update. When the cache write failed (Redis timeout), the token stayed valid. There was no
database-level used flag.

**Fix:** Added a `used_at` column on `password_reset_tokens`, set in the same transaction as the
password update. The confirm endpoint now returns `400 token_used` when `used_at` is set.

**Regression test:** Confirm a token, then confirm it again and expect `400` with `token_used`;
verify the first new password still works.
