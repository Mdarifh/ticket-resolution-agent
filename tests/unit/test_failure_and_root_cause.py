"""Failure Analyzer + Root Cause Analyzer on realistic failures.

Execution evidence is produced by the real API/UI tools (mocked HTTP, in-memory
browser) so the analyzers see exactly what they would in a run.
"""

from uuid import uuid4

import httpx
from langgraph.types import Command

from qa_agent.analysis.failure_facts import analyze_failure
from qa_agent.chains.root_cause_chain import RootCauseDraft, finalize_root_cause, render_root_cause_markdown
from qa_agent.domain import (
    ExecutionResult,
    FailureDetail,
    KnowledgeSearchResult,
    PlannedTest,
    Requirement,
    RequirementAnalysis,
)
from qa_agent.domain.ui_test import UiStep, UiTestRequest
from qa_agent.graph import build_qa_graph, initial_state, run_config
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.failure_analyzer import failure_analyzer
from qa_agent.graph.nodes.root_cause_analyzer import make_root_cause_analyzer
from qa_agent.graph.nodes.test_executor import from_ui_result
from qa_agent.tools.api_test_tool import ApiTestConfig, ApiTestExecutor
from tests.conftest import (
    PASSWORD_RESET_REQUIREMENT,
    fake_ui_runner,
    load_fixture,
    password_reset_responses,
    repo_knowledge_base,
)
from tests.fakes import FakeStructuredChatModel, FakeUiBrowser
from tests.scenarios import (
    api_500,
    api_run,
    api_timeout,
    make_draft,
    make_finding,
    requirement_mismatch,
    case_for,
    ui_element_missing,
    validation_failure,
)

API_DOC = "api_docs/auth-password-reset-api.md"
RESET_REQUIREMENT_DOC = "requirements/REQ-AUTH-003-password-reset.md"


# --- failure analyzer ---------------------------------------------------------


def _facts(failure) -> list[str]:
    return [f"{fact.source}: {fact.detail}" for fact in failure.observed_facts]


def test_api_500_is_a_server_error_with_response_facts():
    result, test_case = api_500()

    failure = analyze_failure(result, test_case)

    assert failure.category == "server_error"
    facts = _facts(failure)
    assert facts[0] == "api.status_code: POST https://sut.test.local/api/v2/password-reset returned HTTP 500"
    assert "api.validation: Expected status 202, got 500" in facts
    assert 'api.response_body: Response body: {"error": "internal_error", "trace_id": "abc123"}' in facts
    assert failure.expected_result == test_case.expected_result
    assert failure.actual_result == "HTTP 500; Expected status 202, got 500"
    assert failure.signal == facts[0].split(": ", 1)[1]


def test_validation_failure_is_an_assertion_mismatch():
    result, test_case = validation_failure()

    failure = analyze_failure(result, test_case)

    assert failure.category == "assertion_mismatch"
    assert """api.validation: JSON path 'error': expected "token_used", got "token_invalid\"""" in _facts(failure)
    assert failure.test_title == "Reset link cannot be used twice"


def test_ui_element_missing_is_element_not_found_with_page_facts():
    result, test_case = ui_element_missing()

    failure = analyze_failure(result, test_case)

    assert failure.category == "element_not_found"
    facts = _facts(failure)
    assert facts[0].startswith("ui.failed_step: Step 3 click 'data-testid=send-reset-link' failed: Timeout")
    assert "ui.selector: Selector matched 0 element(s)" in facts
    assert "ui.console: Browser console error: Uncaught ReferenceError: submitForm is not defined" in facts
    assert "ui.progress: 2 of the executed steps passed before the failure" in facts
    assert failure.automation_type == "ui"


def test_api_timeout_is_a_timeout_without_response():
    result, test_case = api_timeout()

    failure = analyze_failure(result, test_case)

    assert failure.category == "timeout"
    assert _facts(failure)[0] == (
        "api.error: POST https://sut.test.local/api/v2/password-reset produced no response: "
        "Request timed out after 2s (ReadTimeout)"
    )
    assert failure.status == "error"


def test_ui_navigation_timeout_is_a_timeout():
    class SlowBrowser(FakeUiBrowser):
        def new_session(self):
            session = super().new_session()

            def navigate(url):
                from qa_agent.tools.ui_test_tool import UiStepError

                raise UiStepError(f"Timeout 15000ms exceeded. navigating to \"{url}\"")

            session.navigate = navigate
            return session

    runner = fake_ui_runner(SlowBrowser())
    ui = runner.run(UiTestRequest(test_case_id="TC-005", steps=[UiStep(action="navigate", target="/forgot-password.html")]))
    entry = PlannedTest(test_case_id="TC-005", automation_type="ui", execution_order=1, rationale="r")

    failure = analyze_failure(from_ui_result(entry, ui), case_for("TC-005"))

    assert failure.category == "timeout"


def test_requirement_mismatch_is_observed_as_assertion_mismatch():
    result, test_case = requirement_mismatch()

    failure = analyze_failure(result, test_case)

    # The analyzer only records what happened; judging the test wrong is the RCA's job.
    assert failure.category == "assertion_mismatch"
    assert failure.expected_result == "Response is 404 Not Found for an unregistered email"
    assert failure.actual_result == "HTTP 202; Expected status 404, got 202"


def test_connection_refused_is_environment():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    result = api_run("TC-001", refuse, {"method": "GET", "url": "/health", "expected_status": 200})

    assert analyze_failure(result, None).category == "environment"


def test_generic_logs_and_execution_details_become_facts():
    result = ExecutionResult(
        test_case_id="TC-009",
        status="failed",
        duration_ms=12,
        message="assert 1 == 2",
        evidence={"automation_type": "api", "logs": ["worker restarted", "cache miss"]},
    )

    failure = analyze_failure(result, None)

    facts = _facts(failure)
    assert facts[0] == "result: Test reported status 'failed': assert 1 == 2"
    assert "logs: worker restarted" in facts and "logs: cache miss" in facts
    assert facts[-1] == "execution: Executed as api test; status 'failed'; duration 12 ms"
    assert [f.id for f in failure.observed_facts] == [f"TC-009:F{i}" for i in range(1, 5)]


def test_long_facts_are_truncated():
    result = api_run(
        "TC-001", lambda r: httpx.Response(500, text="x" * 2000), {"method": "GET", "url": "/", "expected_status": 200}
    )

    body_fact = next(f for f in analyze_failure(result, None).observed_facts if f.source == "api.response_body")
    assert body_fact.detail.endswith("[truncated]")
    assert len(body_fact.detail) < 600


def test_failure_analyzer_node_uses_execution_results_and_test_cases():
    result, test_case = api_500()
    state = {
        "execution_results": [result],
        "test_cases": [test_case],
        "failures": [FailureDetail(test_case_id="TC-001", status="failed", message=result.message)],
    }

    [failure] = failure_analyzer(state)["failures"]

    assert failure.category == "server_error"
    assert failure.expected_result == test_case.expected_result


# --- root cause analyzer ------------------------------------------------------


def _rca(failure_builder, draft: dict, knowledge_base=repo_knowledge_base):
    result, test_case = failure_builder()
    failure = analyze_failure(result, test_case)
    llm = FakeStructuredChatModel(responses={"RootCauseDraft": [draft]})
    state = {
        "requirement": Requirement(text=PASSWORD_RESET_REQUIREMENT),
        "requirement_analysis": RequirementAnalysis.model_validate(load_fixture("password_reset", "requirement_analysis.json")),
        "test_cases": [test_case],
        "failures": [failure],
    }
    update = make_root_cause_analyzer(llm, knowledge_base)(state)
    return update, failure, llm


REQUIRED_FIELDS = {
    "summary",
    "observed_behavior",
    "expected_behavior",
    "probable_root_cause",
    "evidence",
    "affected_component",
    "severity_suggestion",
    "reproduction_steps",
    "recommended_next_investigation",
    "confidence",
}


def test_api_500_root_cause_keeps_observed_probable_and_unknown_apart():
    draft = make_draft(
        make_finding(
            ["TC-001"],
            observed_behavior="POST /api/v2/password-reset returned HTTP 500 with error internal_error.",
            suspected_origin="product_defect",
            probable_root_cause={
                "description": "Likely an unhandled exception in the reset request handler, possibly a cache timeout.",
                "reasoning": "A 500 with a trace id indicates a server-side exception.",
                "likelihood": "likely",
                "supporting_evidence": ["TC-001:F1", "TC-001:F3"],
            },
            evidence=[
                {"reference": "TC-001:F1", "interpretation": "Server error, not a validation response"},
                {"reference": "TC-001:F3", "interpretation": "Trace id available for log lookup"},
            ],
            unknowns=["Server logs for trace abc123 are not available", "Whether the error is intermittent"],
            affected_component="auth: POST /api/v2/password-reset",
            severity_suggestion="critical",
            confidence=0.55,
        )
    )

    update, failure, _ = _rca(api_500, draft)

    [finding] = update["root_cause_analysis"].findings
    assert REQUIRED_FIELDS <= set(finding.model_dump())
    assert finding.failure_category == "server_error"
    observed = [e for e in finding.evidence if e.kind == "observed"]
    # Evidence text is the recorded fact, not the LLM's wording.
    assert observed[0].detail == failure.observed_facts[0].detail
    assert observed[0].interpretation == "Server error, not a validation response"
    assert finding.probable_root_cause.likelihood == "likely"
    assert "Server logs for trace abc123 are not available" in finding.unknowns
    assert finding.confidence == 0.55
    assert finding.adjustments == []


def test_validation_failure_root_cause():
    draft = make_draft(
        make_finding(
            ["TC-007"],
            suspected_origin="product_defect",
            probable_root_cause={
                "description": "Possibly the used-token check runs after token lookup, returning token_invalid.",
                "reasoning": "Status matched but the error code differs from the documented token_used.",
                "likelihood": "possible",
                "supporting_evidence": ["TC-007:F2", API_DOC],
            },
            evidence=[{"reference": "TC-007:F2", "interpretation": "Wrong error code"}, {"reference": API_DOC, "interpretation": "Documents token_used"}],
            confidence=0.5,
        )
    )

    update, _, _ = _rca(validation_failure, draft)

    [finding] = update["root_cause_analysis"].findings
    assert finding.failure_category == "assertion_mismatch"
    assert [e.kind for e in finding.evidence] == ["observed", "knowledge_base"]
    assert finding.evidence[1].detail == "Password reset API"  # document title, not LLM text
    assert update["retrieved_knowledge"][0].cited_sources == [API_DOC]


def test_ui_element_missing_root_cause_uses_console_logs():
    draft = make_draft(
        make_finding(
            ["TC-005"],
            suspected_origin="product_defect",
            probable_root_cause={
                "description": "Likely the submit button failed to render because page script errored.",
                "reasoning": "Selector matched nothing and the console shows a ReferenceError.",
                "likelihood": "likely",
                "supporting_evidence": ["TC-005:F4", "TC-005:F5"],
            },
            # F1 failed step, F2 progress, F3 page, F4 selector matches, F5 console error
            evidence=[{"reference": "TC-005:F4", "interpretation": "Button absent"}, {"reference": "TC-005:F5", "interpretation": "Script error"}],
            alternative_causes=[
                {"description": "Possibly the test id changed in a redesign", "reasoning": "Common after UI refactors", "likelihood": "possible", "supporting_evidence": []}
            ],
            confidence=0.6,
        )
    )

    update, failure, llm = _rca(ui_element_missing, draft)

    [finding] = update["root_cause_analysis"].findings
    assert finding.failure_category == "element_not_found"
    assert {e.detail for e in finding.evidence} == {
        "Selector matched 0 element(s)",
        "Browser console error: Uncaught ReferenceError: submitForm is not defined",
    }
    [alternative] = finding.alternative_causes
    assert alternative.likelihood == "speculative"  # no evidence offered
    [messages] = llm.calls_for("RootCauseDraft")
    assert "Uncaught ReferenceError: submitForm is not defined" in messages[-1].content


def test_timeout_root_cause_suspects_environment():
    draft = make_draft(
        make_finding(
            ["TC-004"],
            suspected_origin="environment",
            probable_root_cause={
                "description": "Possibly the service was slow or unreachable in the test environment.",
                "reasoning": "No response within 2 s; nothing indicates a product defect yet.",
                "likelihood": "possible",
                "supporting_evidence": ["TC-004:F1"],
            },
            evidence=[{"reference": "TC-004:F1", "interpretation": "No response"}],
            unknowns=["Service latency during the run", "Whether other endpoints were also slow"],
            severity_suggestion="medium",
            confidence=0.35,
        )
    )

    update, failure, _ = _rca(api_timeout, draft)

    [finding] = update["root_cause_analysis"].findings
    assert failure.category == finding.failure_category == "timeout"
    assert finding.suspected_origin == "environment"
    assert "Whether other endpoints were also slow" in finding.unknowns


def test_requirement_mismatch_root_cause_blames_the_test_expectation():
    draft = make_draft(
        make_finding(
            ["TC-004"],
            summary="Test expectation contradicts the no-enumeration requirement",
            observed_behavior="The API returned 202 with the generic message for an unregistered email.",
            expected_behavior="The test expected 404, but REQ-AUTH-003 requires the same response for any email.",
            suspected_origin="requirement_mismatch",
            probable_root_cause={
                "description": "The test case likely encodes an outdated expectation; the product behaves as the requirement demands.",
                "reasoning": "Observed 202 + generic message matches the documented contract.",
                "likelihood": "likely",
                "supporting_evidence": ["TC-004:F1", "TC-004:F3", RESET_REQUIREMENT_DOC],
            },
            evidence=[
                {"reference": "TC-004:F1", "interpretation": "202 returned"},
                {"reference": RESET_REQUIREMENT_DOC, "interpretation": "Requires identical response"},
            ],
            severity_suggestion="low",
            recommended_next_investigation=["Update TC-004 to expect 202 and the generic message"],
            confidence=0.75,
        )
    )

    update, _, llm = _rca(requirement_mismatch, draft)

    [finding] = update["root_cause_analysis"].findings
    assert finding.suspected_origin == "requirement_mismatch"
    usage = update["retrieved_knowledge"][0]
    assert RESET_REQUIREMENT_DOC in usage.retrieved_sources  # requirements are searched
    assert [e.reference for e in finding.evidence if e.kind == "knowledge_base"] == [RESET_REQUIREMENT_DOC]
    human = llm.calls_for("RootCauseDraft")[0][-1].content
    assert "Response is 404 Not Found for an unregistered email" in human  # test expectation
    assert PASSWORD_RESET_REQUIREMENT in human  # requirement
    assert "A request for an email that is not registered does not reset any password" in human  # analysis


# --- guards -------------------------------------------------------------------


def _finalize(draft: dict, failure_builder=api_500, knowledge: KnowledgeSearchResult | None = None):
    result, test_case = failure_builder()
    failure = analyze_failure(result, test_case)
    knowledge = knowledge or repo_knowledge_base().search("password reset api 500 error", k=6)
    return finalize_root_cause(RootCauseDraft.model_validate(draft), [failure], knowledge)


def test_unsupported_cause_becomes_speculative_with_capped_confidence():
    finding = make_finding(["TC-001"], confidence=0.9)
    finding["probable_root_cause"]["supporting_evidence"] = []

    analysis, _ = _finalize(make_draft(finding))

    [result] = analysis.findings
    assert result.probable_root_cause.likelihood == "speculative"
    assert result.confidence == 0.3
    assert "No observed evidence directly supports the probable cause" in result.unknowns
    assert any("marked speculative" in note for note in result.adjustments)


def test_cause_resting_only_on_knowledge_base_is_capped():
    finding = make_finding(["TC-001"], confidence=0.9)
    finding["probable_root_cause"]["supporting_evidence"] = [API_DOC]

    analysis, _ = _finalize(make_draft(finding))

    [result] = analysis.findings
    assert result.probable_root_cause.likelihood == "possible"
    assert result.confidence == 0.5
    assert "No observed evidence directly supports the probable cause" in result.unknowns


def test_fabricated_fact_ids_and_unretrieved_sources_are_dropped():
    finding = make_finding(
        ["TC-001"],
        evidence=[
            {"reference": "TC-001:F1", "interpretation": "real"},
            {"reference": "TC-001:F42", "interpretation": "Logs show a null pointer"},
            {"reference": "previous_bugs/BUG-0000-made-up.md", "interpretation": "similar"},
        ],
    )
    finding["probable_root_cause"]["supporting_evidence"] = ["TC-001:F42"]

    analysis, discarded = _finalize(make_draft(finding))

    [result] = analysis.findings
    assert [e.reference for e in result.evidence] == ["TC-001:F1"]
    assert result.probable_root_cause.supporting_evidence == []
    assert result.probable_root_cause.likelihood == "speculative"
    assert discarded == ["previous_bugs/BUG-0000-made-up.md"]  # fabricated fact ids are not "citations"
    assert any("TC-001:F42" in note for note in result.adjustments)


def test_findings_only_name_failed_tests_and_every_failure_is_covered():
    analysis, _ = _finalize(make_draft(make_finding(["TC-777"])))

    [placeholder] = analysis.findings
    assert placeholder.test_case_ids == ["TC-001"]
    assert placeholder.probable_root_cause.description == "Not analyzed"
    assert placeholder.confidence == 0.0
    assert placeholder.unknowns == ["Root cause not analyzed"]
    assert [e.kind for e in placeholder.evidence] == ["observed"] * len(placeholder.evidence)
    assert analysis.confidence == 0.0


def test_confidence_is_clamped_and_overall_is_the_minimum():
    first = make_finding(["TC-001"], confidence=1.7)
    analysis, _ = _finalize(make_draft(first))

    assert analysis.findings[0].confidence == 1.0
    assert analysis.confidence == 1.0


def test_blank_component_becomes_unknown():
    analysis, _ = _finalize(make_draft(make_finding(["TC-001"], affected_component="  ")))

    assert analysis.findings[0].affected_component == "unknown"


def test_markdown_labels_observed_probable_and_unknown():
    analysis, _ = _finalize(make_draft(make_finding(["TC-001"])))

    text = render_root_cause_markdown(analysis)

    assert "**Observed (recorded by test execution):** observed" in text
    assert "**Probable cause (not confirmed, likely):** Possibly X" in text
    assert "- **Unknown:**\n  - Server logs are not available" in text
    assert "[observed] TC-001:F1: POST https://sut.test.local/api/v2/password-reset returned HTTP 500" in text


def test_system_prompt_demands_separation_and_hedging():
    update, _, llm = _rca(api_500, make_draft(make_finding(["TC-001"])))

    system = llm.calls_for("RootCauseDraft")[0][0].content
    assert "never as established fact" in system
    assert "Unknowns" in system
    assert "requirement_mismatch" in system


def test_rca_prompt_contains_all_failure_inputs():
    _, failure, llm = _rca(ui_element_missing, make_draft(make_finding(["TC-005"])))

    human = llm.calls_for("RootCauseDraft")[0][-1].content
    assert PASSWORD_RESET_REQUIREMENT in human
    assert '"expected_result": "A validation error is shown and no request is sent"' in human
    assert '"steps": [' in human  # test case
    assert "TC-005:F1" in human and "Selector matched 0 element(s)" in human
    assert "<reference_material>" in human


# --- graph --------------------------------------------------------------------


def test_graph_api_500_flows_through_both_analyzers_into_the_report():
    def server(request):
        if request.url.path == "/api/v2/password-reset":
            return httpx.Response(500, json={"error": "internal_error"})
        return httpx.Response(400, json={"error": "token_used"})

    responses = password_reset_responses()
    responses["RootCauseDraft"] = [
        make_draft(
            make_finding(["TC-001", "TC-004", "TC-008"], summary="Reset request endpoint returns 500", confidence=0.8),
            summary="All reset requests fail with HTTP 500",
        )
    ]
    responses["BugNarrativeDraft"] = [
        {
            "reports": [
                {
                    "finding_id": "RC-1",
                    "title": "Password reset request returns HTTP 500",
                    "summary": "Every reset request failed with HTTP 500; users cannot start a password reset.",
                    "severity": "critical",
                    "severity_rationale": "Blocks a core account-recovery flow for all users.",
                    "priority": "p0",
                    "priority_rationale": "Core flow is down.",
                    "related_bugs": [],
                }
            ]
        }
    ]
    graph = build_qa_graph(
        {"confidence_checker": make_confidence_checker(threshold=0.7)},
        llm=FakeStructuredChatModel(responses=responses),
        knowledge_base=repo_knowledge_base(),
        api_executor=ApiTestExecutor(ApiTestConfig(base_url="https://sut.test.local"), httpx.Client(transport=httpx.MockTransport(server))),
        ui_runner=fake_ui_runner(),
    )
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)
    config = run_config(requirement.id)

    paused = graph.invoke(initial_state(requirement), config)
    # A critical bug is a sensitive action -> review.
    assert paused["confidence_score"].reasons == ["Sensitive action requires approval: report_high_severity_bugs (BR-1)"]
    state = graph.invoke(Command(resume={"decision": "approve"}), config)

    assert {f.test_case_id: f.category for f in state["failures"]} == {
        "TC-001": "server_error", "TC-004": "server_error", "TC-008": "server_error",
    }
    [finding] = state["root_cause_analysis"].findings
    assert finding.test_case_ids == ["TC-001", "TC-004", "TC-008"]
    assert finding.failure_category == "server_error"
    report = state["final_report"]
    assert report.status == "failed"
    assert "## Root cause analysis" in report.markdown
    assert "Probable cause (not confirmed" in report.markdown
    assert "## Bug reports" in report.markdown
    assert "Probable root cause (UNCONFIRMED" in report.markdown
