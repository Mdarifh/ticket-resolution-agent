---
title: Flaky UI tests
doc_type: troubleshooting
component: web
last_updated: 2026-06-25
tags: [ui, playwright, flaky]
---
# Flaky UI tests

Common causes of Playwright tests that pass locally and fail in CI:

- **Element not found**: the test clicked before the element rendered. Use locator auto-waiting
  (`page.get_by_role(...)`) instead of fixed sleeps or raw CSS selectors.
- **Animations**: toasts and modals animate in; assert on visibility, not position.
- **Shared test data**: two tests using the same account change each other's state. Create data
  per test.
- **Time-dependent logic**: expiry windows (such as 30-minute reset links) need a controllable
  clock, not real waiting.

Before filing a UI bug, rerun the test up to 2 times. A test that fails only sometimes should be
tagged as flaky and investigated separately from product bugs.
