---
title: Diagnosing API test failures
doc_type: troubleshooting
component: general
last_updated: 2026-07-01
tags: [api, 5xx, timeout, assertion]
---
# Diagnosing API test failures

**5xx responses**

- `502`/`503` right after a deploy usually mean pods are still starting; retry after the readiness
  probe passes before calling it a bug.
- `500` with a stack trace mentioning `redis.exceptions.TimeoutError` points at the cache cluster;
  check its CPU and eviction metrics.

**Timeouts (`ReadTimeout`)**

- Compare with the service's p95 latency dashboard. If the whole service is slow, it is an
  environment problem, not a test failure.
- A single slow endpoint often means a missing database index or an external call without a timeout.

**Assertion mismatches**

- Check whether the API contract changed: compare the response with the current API documentation
  before filing a bug against the service.
- Status `400` where `200` was expected often means test data drifted, for example a test account
  was reset by the nightly job.

**Token and auth errors**

- `401` in tests that passed yesterday: the shared test token may have expired; regenerate it.
