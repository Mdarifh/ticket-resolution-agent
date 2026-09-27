---
title: Troubleshooting email delivery in test environments
doc_type: troubleshooting
component: notifications
last_updated: 2026-08-05
tags: [email, staging, mailhog]
---
# Troubleshooting email delivery in test environments

Staging does not send real email. All outgoing mail is captured by MailHog at
`http://mailhog.staging.internal:8025` (API: `GET /api/v2/messages`).

**Reset email not received**

1. Check the `notification-worker` logs for the request id. `SMTP connection refused` means
   MailHog is down; restart the `mailhog` deployment.
2. If the worker never picked up the job, check the `email` queue depth in RabbitMQ. A growing
   queue with no consumers means the worker crashed; look for `OOMKilled` in pod events.
3. If the job ran but no message exists, the user lookup probably found no account. Check how the
   email was normalized (see BUG-1188).

**Links in emails point to the wrong host**: `PUBLIC_BASE_URL` in the worker's config is wrong for
the environment.
