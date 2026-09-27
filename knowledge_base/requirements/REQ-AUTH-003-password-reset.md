---
title: "REQ-AUTH-003: Password reset via registered email"
doc_type: requirement
requirement_id: REQ-AUTH-003
component: auth
status: approved
last_updated: 2026-01-22
tags: [password-reset, auth]
---
# REQ-AUTH-003: Password reset via registered email

A user who forgot their password can reset it using their registered email address.

Acceptance criteria:

1. The user enters their email on the "Forgot password" page and submits it.
2. If the email is registered, a reset link is emailed within 2 minutes.
3. The page shows the same confirmation message whether or not the email is registered.
4. The reset link expires after 30 minutes and can be used only once.
5. The new password must satisfy the password policy (see the password reset API document).
6. After a successful reset, all existing sessions are signed out and the user can log in with the
   new password.
7. No more than 5 reset requests per email per hour are accepted.
