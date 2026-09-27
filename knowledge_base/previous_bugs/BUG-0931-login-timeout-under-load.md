---
title: Login requests timed out under load
doc_type: previous_bug
bug_id: BUG-0931
component: auth
severity: critical
status: resolved
last_updated: 2020-11-03
tags: [login, performance, timeout]
---
# BUG-0931: Login requests timed out under load

**Symptom:** During peak traffic, `POST /api/v1/login` took over 30 seconds and API tests failed
with `ReadTimeout`.

**Root cause:** Password hashing used bcrypt with cost 14 on a CPU-constrained pod, and the
connection pool to the user database had only 5 connections.

**Fix:** Lowered bcrypt cost to 12, raised the pool to 20 connections, and added autoscaling on CPU.

Note: this incident predates the v2 auth service; v1 endpoints are retired.
