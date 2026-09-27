---
title: Password reset failed for emails entered with capital letters
doc_type: previous_bug
bug_id: BUG-1188
component: auth
severity: medium
status: resolved
last_updated: 2026-07-19
tags: [password-reset, email, validation]
---
# BUG-1188: Password reset failed for emails entered with capital letters

**Symptom:** Users typing `Jane.Doe@Example.com` never received a reset email, although the account
was registered as `jane.doe@example.com`. The API still returned the generic `202`, so the failure
was silent.

**Root cause:** The reset handler looked up users with a case-sensitive query, while registration
lower-cased emails before saving.

**Fix:** Normalize emails (trim and lower-case) in a shared helper used by registration, login and
password reset.

**Regression test:** Request a reset with a mixed-case version of a registered email and assert a
reset email is queued for that account.
