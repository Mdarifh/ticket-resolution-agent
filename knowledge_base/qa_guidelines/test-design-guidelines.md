---
title: Test design guidelines
doc_type: qa_guideline
component: general
last_updated: 2026-05-20
tags: [test-design, coverage]
---
# Test design guidelines

## Coverage

- Every acceptance criterion needs at least one positive and one negative test.
- Add edge-case tests for boundaries: minimum and maximum lengths, limits such as rate limits and
  expiry windows (test just before and just after the limit), empty and whitespace-only input,
  letter casing and Unicode.
- Every previously fixed bug in the same component gets a regression test.

## Writing test cases

- Steps are atomic: one action per step.
- Expected results must be observable: status codes, response fields, UI messages, database state.
- Use concrete test data. Never write "valid email"; write `registered.user@example.com`.
- Keep each test independent: create its own data in preconditions rather than relying on
  another test having run first.

## Choosing automation

- Prefer API tests for business rules; they are faster and less flaky than UI tests.
- Use UI tests for user journeys, client-side validation and rendering.
- Keep manual tests for things automation cannot observe reliably, such as real email inboxes,
  SMS, and visual or usability judgement.
