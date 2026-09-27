from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qa_agent.domain.bug_ticket import BugTicket, ticket_to_requirement, ticket_verdict

TICKET = {
    "ticket_id": "SD-1042",
    "title": "Account not locked after failed logins",
    "description": "Customer says they can keep guessing passwords.",
    "steps_to_reproduce": "1. Open the login page\n2) Enter a wrong password 3 times\n\n- Try once more",
    "expected_result": "The 4th attempt shows the account is locked",
    "actual_result": "The 4th attempt still says invalid password",
    "environment": "http://127.0.0.1:8765",
}


def _results(*statuses):
    return [SimpleNamespace(test_case_id=f"TC-00{i}", status=s) for i, s in enumerate(statuses, start=1)]


def test_steps_are_split_per_line_without_numbering():
    ticket = BugTicket.model_validate(TICKET)

    assert ticket.steps_to_reproduce == ["Open the login page", "Enter a wrong password 3 times", "Try once more"]


def test_expected_and_actual_results_are_required():
    with pytest.raises(ValidationError):
        BugTicket.model_validate({**TICKET, "expected_result": " "})
    with pytest.raises(ValidationError):
        BugTicket.model_validate({k: v for k, v in TICKET.items() if k != "actual_result"})


def test_requirement_tests_the_expected_behaviour_by_following_the_steps():
    text = ticket_to_requirement(BugTicket.model_validate(TICKET))

    assert text.startswith("Service desk bug ticket SD-1042: Account not locked after failed logins")
    assert "2. Enter a wrong password 3 times" in text
    assert "results in: The 4th attempt shows the account is locked" in text
    assert "does not occur: The 4th attempt still says invalid password" in text
    assert "Environment / URL: http://127.0.0.1:8765" in text


@pytest.mark.parametrize(
    ("run_status", "statuses", "decision", "verdict"),
    [
        ("running", (), None, "pending"),
        ("awaiting_execution", (), None, "pending"),
        ("error", (), None, "inconclusive"),
        ("completed", (), None, "inconclusive"),
        ("completed", ("passed", "failed", "skipped"), None, "confirmed"),
        ("awaiting_review", ("failed",), None, "confirmed"),
        ("completed", ("failed",), "REJECT", "inconclusive"),
        ("completed", ("passed", "passed", "error"), None, "not_reproduced"),
        ("completed", ("error", "skipped"), None, "inconclusive"),
    ],
)
def test_verdict(run_status, statuses, decision, verdict):
    assert ticket_verdict(run_status, _results(*statuses), decision).status == verdict


def test_reproduction_test_decides_over_failing_variations():
    # TC-001 follows the ticket's steps and passed; a variation failed (e.g. a broken locator).
    verdict = ticket_verdict("completed", _results("passed", "passed", "failed"), reproduction_test="TC-001")

    assert verdict.status == "not_reproduced"
    assert verdict.reproduction_test == "TC-001"
    assert "TC-003 failed" in verdict.explanation


def test_failing_reproduction_test_confirms_the_bug():
    verdict = ticket_verdict("completed", _results("failed", "passed"), reproduction_test="TC-001")

    assert verdict.status == "confirmed"
    assert "reproduction test TC-001" in verdict.explanation


def test_variations_decide_when_the_reproduction_test_could_not_run():
    verdict = ticket_verdict("completed", _results("skipped", "failed"), reproduction_test="TC-001")

    assert verdict.status == "confirmed"
    assert "could not run" in verdict.explanation


def test_reviewer_rejection_makes_a_confirmed_bug_inconclusive():
    verdict = ticket_verdict("completed", _results("failed"), "REJECT", reproduction_test="TC-001")

    assert verdict.status == "inconclusive"


def test_confirmed_verdict_names_the_failing_tests():
    verdict = ticket_verdict("completed", _results("passed", "failed"))

    assert verdict.failed_tests == ["TC-002"]
    assert verdict.passed_tests == ["TC-001"]
    assert "TC-002" in verdict.explanation
