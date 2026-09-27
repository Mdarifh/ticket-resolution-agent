"""Service desk bug tickets: verify whether a reported bug is real.

A ticket is turned into a verification requirement whose acceptance criteria are
the *expected* (correct) behaviour. The normal workflow then generates tests that
follow the reproduction steps and runs them, so:

- a failing test means the reported behaviour occurs     -> bug confirmed
- only passing tests means the expected behaviour occurs -> not reproducible
- nothing conclusive executed (errors, skips, manual)     -> inconclusive
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TicketVerdictStatus = Literal["pending", "confirmed", "not_reproduced", "inconclusive"]

BUG_VERIFICATION_MODE = "bug_verification"


class BugTicket(BaseModel):
    ticket_id: str = Field(default="", max_length=100, description="Service desk ticket number.")
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=10_000)
    steps_to_reproduce: list[str] = Field(default_factory=list, max_length=50)
    expected_result: str = Field(min_length=1, max_length=2_000)
    actual_result: str = Field(min_length=1, max_length=2_000)
    environment: str = Field(default="", max_length=500, description="URL, build or environment.")
    priority: str = Field(default="", max_length=50)

    @field_validator("steps_to_reproduce", mode="before")
    @classmethod
    def _split_steps(cls, value: object) -> object:
        """Accept one step per line; drop numbering and blank lines."""
        if isinstance(value, str):
            value = value.splitlines()
        if isinstance(value, list):
            steps = []
            for step in value:
                text = str(step).strip().lstrip("-*•").strip()
                head, _, rest = text.partition(" ")
                if head.rstrip(".)").isdigit() and rest:
                    text = rest.strip()
                if text:
                    steps.append(text)
            return steps
        return value

    @field_validator(
        "ticket_id", "title", "description", "expected_result", "actual_result", "environment", "priority",
        mode="before",
    )
    @classmethod
    def _strip(cls, value: object) -> object:
        # Before the length checks, so whitespace-only required fields are rejected.
        return value.strip() if isinstance(value, str) else value


def ticket_to_requirement(ticket: BugTicket) -> str:
    """Requirement text for the QA workflow; the correct behaviour is what gets tested."""
    label = f"{ticket.ticket_id}: {ticket.title}" if ticket.ticket_id else ticket.title
    lines = [
        f"Service desk bug ticket {label}",
        "",
        "Goal: verify whether this reported bug is real by reproducing it.",
    ]
    if ticket.description:
        lines += ["", f"Reported problem: {ticket.description}"]
    if ticket.environment:
        lines += ["", f"Environment / URL: {ticket.environment}"]
    if ticket.steps_to_reproduce:
        lines += ["", "Steps to reproduce:"]
        lines += [f"{i}. {step}" for i, step in enumerate(ticket.steps_to_reproduce, start=1)]
    lines += [
        "",
        f"Expected behaviour (correct): {ticket.expected_result}",
        f"Actual behaviour reported by the user: {ticket.actual_result}",
        "",
        "Acceptance criteria (the correct behaviour that the tests must check):",
        f"1. Following the steps to reproduce results in: {ticket.expected_result}",
        f"2. The reported behaviour does not occur: {ticket.actual_result}",
        "",
        "Test guidance: the FIRST test case (TC-001) is the reproduction test: it follows the steps "
        "to reproduce exactly, in one test, and checks the expected behaviour. Add at most two closely "
        "related variations of the same scenario; do not test unrelated features.",
    ]
    return "\n".join(lines)


class TicketVerdict(BaseModel):
    status: TicketVerdictStatus
    headline: str
    explanation: str
    failed_tests: list[str] = Field(default_factory=list)
    passed_tests: list[str] = Field(default_factory=list)
    inconclusive_tests: list[str] = Field(default_factory=list)
    reproduction_test: str | None = Field(
        default=None, description="The test that follows the ticket's steps exactly."
    )


def _ids(ids: list[str]) -> str:
    return ", ".join(ids)


def ticket_verdict(
    run_status: str,
    results: list[Any],
    human_decision: str | None = None,
    reproduction_test: str | None = None,
) -> TicketVerdict:
    """Verdict from execution results (objects with ``test_case_id`` and ``status``).

    ``reproduction_test`` is the test that follows the ticket's steps exactly (the first
    generated test case). It decides the verdict; the other tests are variations whose
    outcome is reported as a note, so a broken variation cannot "confirm" a bug on its own.
    When the reproduction test could not run, the variations decide instead.
    """
    failed = [r.test_case_id for r in results if r.status == "failed"]
    passed = [r.test_case_id for r in results if r.status == "passed"]
    other = [r.test_case_id for r in results if r.status not in ("failed", "passed")]
    lists = {"failed_tests": failed, "passed_tests": passed, "inconclusive_tests": other}

    if not results:
        if run_status == "error":
            return TicketVerdict(
                status="inconclusive",
                headline="Inconclusive: the run stopped before any test ran",
                explanation="Fix the run error and verify the ticket again.",
            )
        if run_status == "completed":
            return TicketVerdict(
                status="inconclusive",
                headline="Inconclusive: no test was executed",
                explanation="No test results were produced, so the ticket could not be checked.",
            )
        return TicketVerdict(
            status="pending",
            headline="Verification in progress",
            explanation="The verdict appears once the reproduction tests have run.",
        )

    status_of = {r.test_case_id: r.status for r in results}
    primary = reproduction_test if reproduction_test in status_of else None
    primary_status = status_of.get(primary) if primary else None
    variation_failed = [t for t in failed if t != primary]
    variation_passed = [t for t in passed if t != primary]

    if primary_status == "failed":
        status = "confirmed"
        explanation = (
            f"The reproduction test {primary}, which follows the ticket's steps, failed: the application "
            "does not show the expected behaviour."
        )
        if variation_failed:
            explanation += f" Related variations also failed: {_ids(variation_failed)}."
    elif primary_status == "passed":
        status = "not_reproduced"
        explanation = (
            f"The reproduction test {primary}, which follows the ticket's steps, passed: the application "
            "showed the expected behaviour, so the reported bug did not occur."
        )
        if variation_failed:
            explanation += (
                f" Note: related variation(s) {_ids(variation_failed)} failed. Check them separately; they "
                "may be a different issue or a problem in the generated test."
            )
    else:
        cannot_run = f"The reproduction test {primary} could not run" if primary else "No reproduction test ran"
        if failed:
            status = "confirmed"
            explanation = (
                f"{cannot_run}, but related test(s) {_ids(failed)} failed: the application does not show "
                "the expected behaviour. Confirm with the reproduction steps manually if needed."
            )
        elif passed:
            status = "not_reproduced"
            explanation = (
                f"{cannot_run}, but related test(s) {_ids(passed)} passed and none failed. Verify the exact "
                "steps manually to be sure."
            )
        else:
            return TicketVerdict(
                status="inconclusive",
                headline="Inconclusive: no test could check the ticket",
                explanation=(
                    f"All {len(other)} test(s) errored or were skipped ({_ids(other)}). Check the test "
                    "environment configuration or verify the ticket manually."
                ),
                reproduction_test=primary,
                **lists,
            )

    if status == "confirmed" and human_decision == "REJECT":
        return TicketVerdict(
            status="inconclusive",
            headline="Inconclusive: the reviewer rejected the agent's findings",
            explanation=f"{explanation} A human reviewer rejected this analysis.",
            reproduction_test=primary,
            **lists,
        )
    headline = (
        "Bug confirmed: the reported behaviour was reproduced"
        if status == "confirmed"
        else "Not reproducible: the application behaves as expected"
    )
    return TicketVerdict(
        status=status, headline=headline, explanation=explanation, reproduction_test=primary, **lists
    )
