"""Test Executor node.

* api: the planned API test cases are turned into HTTP requests by an LLM
  (grounded in API docs from the knowledge base) and run with
  ``execute_api_test`` against the configured test environment.
* ui: an LLM turns the planned UI test cases into Playwright steps, using an
  inventory of the elements on the app's pages, and ``UiTestRunner`` runs them
  in a browser against the configured local/test app.
* manual: never executed by the agent; reported as skipped.

A failure preparing one kind of test (configuration, LLM, browser) turns
those tests into ``error`` results without losing the other kind's results.

``make_mock_test_executor`` remains for tests that need fixed outcomes.
"""

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from qa_agent.chains import get_chat_model
from qa_agent.chains.api_request_chain import (
    InvalidApiRequestSpec,
    generate_api_requests,
    spec_to_request,
)
from qa_agent.chains.ui_script_chain import InvalidUiScript, generate_ui_scripts, script_to_request
from qa_agent.domain import ExecutionResult, PlannedTest, TestCase
from qa_agent.domain.api_test import ApiTestResult
from qa_agent.domain.execution import ExecutionStatus
from qa_agent.domain.ui_test import UiTestResult
from qa_agent.graph.instrumentation import NodeFn
from qa_agent.graph.state import QAAgentState
from qa_agent.rag.context import KnowledgeBaseProvider, retrieve, usage_record
from qa_agent.rag.knowledge_base import get_default_knowledge_base
from qa_agent.tools.api_test_tool import ApiTestExecutor
from qa_agent.tools.target_policy import UnsafeRequestError
from qa_agent.tools.ui_test_tool import UiTestRunner

ApiExecutorProvider = Callable[[], ApiTestExecutor]
UiRunnerProvider = Callable[[], UiTestRunner]

MANUAL_MESSAGE = "Manual test: requires a human tester"
MAX_BODY_CHARS_IN_EVIDENCE = 2000


def make_test_executor(
    llm: BaseChatModel | None = None,
    knowledge_base: KnowledgeBaseProvider | None = None,
    api_executor: ApiExecutorProvider | None = None,
    ui_runner: UiRunnerProvider | None = None,
) -> NodeFn:
    """Defaults (configured LLM, knowledge base, API/UI settings) resolve when the node runs.

    A provided ``api_executor`` / ``ui_runner`` is left open; ones created
    from settings are closed after the node finishes.
    """
    kb_provider = knowledge_base or get_default_knowledge_base

    def test_executor(state: QAAgentState) -> dict[str, Any]:
        plan = state.get("test_plan")
        entries = sorted(plan.entries, key=lambda e: e.execution_order) if plan else []
        by_id = {tc.test_case_id: tc for tc in state.get("test_cases") or []}

        results: dict[str, ExecutionResult] = {}
        api_entries: list[PlannedTest] = []
        ui_entries: list[PlannedTest] = []
        for entry in entries:
            if entry.automation_type == "manual":
                results[entry.test_case_id] = _skipped(entry, MANUAL_MESSAGE)
            elif entry.automation_type == "ui":
                ui_entries.append(entry)
            else:
                api_entries.append(entry)

        update: dict[str, Any] = {}
        if api_entries:
            executor = api_executor() if api_executor else ApiTestExecutor.from_settings()
            try:
                api_results, usage = _run_api_tests(
                    api_entries, by_id, executor, llm, kb_provider
                )
            except Exception as exc:
                api_results, usage = _all_errors(api_entries, "API tests could not run", exc), None
            finally:
                if api_executor is None:
                    executor.close()
            results.update(api_results)
            if usage is not None:
                update["retrieved_knowledge"] = [usage]

        if ui_entries:
            runner = ui_runner() if ui_runner else UiTestRunner.from_settings()
            try:
                results.update(_run_ui_tests(ui_entries, by_id, runner, llm))
            except Exception as exc:
                results.update(_all_errors(ui_entries, "UI tests could not run", exc))
            finally:
                if ui_runner is None:
                    runner.close()

        update["execution_results"] = [results[e.test_case_id] for e in entries]
        return update

    return test_executor


def _run_api_tests(
    entries: list[PlannedTest],
    by_id: dict[str, TestCase],
    executor: ApiTestExecutor,
    llm: BaseChatModel | None,
    kb_provider: KnowledgeBaseProvider,
):
    # Fail fast on unusable configuration: no LLM call, no retrieval, no traffic.
    try:
        executor.config.check_usable()
    except UnsafeRequestError as exc:
        return {e.test_case_id: _error(e, str(exc)) for e in entries}, None

    results: dict[str, ExecutionResult] = {}
    test_cases = []
    for entry in entries:
        if entry.test_case_id in by_id:
            test_cases.append(by_id[entry.test_case_id])
        else:
            results[entry.test_case_id] = _error(entry, "Test case not found in state")
    if not test_cases:
        return results, None

    query = "\n".join(f"{tc.title} {tc.description}" for tc in test_cases)
    knowledge = retrieve(kb_provider, query, doc_types=["api_doc"])
    request_plan = generate_api_requests(llm or get_chat_model(), test_cases, knowledge)

    specs: dict[str, Any] = {}
    for spec in request_plan.requests:
        specs.setdefault(spec.test_case_id, spec)  # first spec per test case wins

    for entry in entries:
        if entry.test_case_id in results:
            continue
        spec = specs.get(entry.test_case_id)
        if spec is None:
            results[entry.test_case_id] = _error(entry, "No API request was generated for this test")
        elif spec.skip_reason:
            results[entry.test_case_id] = _skipped(entry, f"Not executable: {spec.skip_reason}")
        else:
            try:
                request = spec_to_request(spec)
            except InvalidApiRequestSpec as exc:
                results[entry.test_case_id] = _error(entry, str(exc))
                continue
            results[entry.test_case_id] = from_api_result(entry, executor.execute(request))

    return results, usage_record("test_executor", knowledge, [], [])


def _run_ui_tests(
    entries: list[PlannedTest],
    by_id: dict[str, TestCase],
    runner: UiTestRunner,
    llm: BaseChatModel | None,
) -> dict[str, ExecutionResult]:
    # Fail fast on unusable configuration: no browser, no LLM call.
    try:
        runner.config.check_usable()
    except UnsafeRequestError as exc:
        return {e.test_case_id: _error(e, str(exc)) for e in entries}

    results: dict[str, ExecutionResult] = {}
    test_cases = []
    for entry in entries:
        if entry.test_case_id in by_id:
            test_cases.append(by_id[entry.test_case_id])
        else:
            results[entry.test_case_id] = _error(entry, "Test case not found in state")
    if not test_cases:
        return results

    inventory = runner.page_inventory()
    if not inventory:
        for entry in entries:
            results.setdefault(
                entry.test_case_id,
                _error(entry, "No pages of the app could be loaded for UI testing"),
            )
        return results
    script_plan = generate_ui_scripts(llm or get_chat_model(), test_cases, inventory)

    scripts: dict[str, Any] = {}
    for script in script_plan.scripts:
        scripts.setdefault(script.test_case_id, script)  # first script per test case wins

    for entry in entries:
        if entry.test_case_id in results:
            continue
        script = scripts.get(entry.test_case_id)
        if script is None:
            results[entry.test_case_id] = _error(entry, "No UI script was generated for this test")
        elif script.skip_reason:
            results[entry.test_case_id] = _skipped(entry, f"Not executable: {script.skip_reason}")
        else:
            try:
                request = script_to_request(script)
            except InvalidUiScript as exc:
                results[entry.test_case_id] = _error(entry, str(exc))
                continue
            results[entry.test_case_id] = from_ui_result(entry, runner.run(request))
    return results


def from_ui_result(entry: PlannedTest, result: UiTestResult) -> ExecutionResult:
    evidence = result.model_dump(mode="json")
    evidence["automation_type"] = entry.automation_type
    if result.status == "error":
        evidence["reached_target"] = bool(result.executed_steps)
    return ExecutionResult(
        test_case_id=entry.test_case_id,
        status=result.status,
        duration_ms=round(result.execution_time),
        message=result.error,
        evidence=evidence,
    )


def _all_errors(entries: list[PlannedTest], prefix: str, exc: Exception) -> dict[str, ExecutionResult]:
    message = f"{prefix}: {type(exc).__name__}: {exc}"
    return {e.test_case_id: _error(e, message) for e in entries}


def from_api_result(entry: PlannedTest, result: ApiTestResult) -> ExecutionResult:
    if result.status == "passed":
        message = None
    elif result.status == "failed":
        message = "; ".join(result.validation_errors)
    else:
        message = result.error_message

    evidence = result.model_dump(mode="json")
    body = evidence.get("response_body")
    if isinstance(body, str) and len(body) > MAX_BODY_CHARS_IN_EVIDENCE:
        evidence["response_body"] = body[:MAX_BODY_CHARS_IN_EVIDENCE] + " [truncated]"
    evidence["automation_type"] = entry.automation_type
    if result.status == "error":
        # Refused before sending (config, unsafe target) vs. sent and timed out / connection failed.
        evidence["reached_target"] = result.status_code is not None or result.response_time is not None

    return ExecutionResult(
        test_case_id=entry.test_case_id,
        status=result.status,
        duration_ms=round(result.response_time or 0),
        message=message,
        evidence=evidence,
    )


def _skipped(entry: PlannedTest, message: str) -> ExecutionResult:
    return ExecutionResult(
        test_case_id=entry.test_case_id,
        status="skipped",
        message=message,
        evidence={"automation_type": entry.automation_type},
    )


def _error(entry: PlannedTest, message: str) -> ExecutionResult:
    """The test could not run at all (no request/script, tool unusable): nothing reached the target."""
    return ExecutionResult(
        test_case_id=entry.test_case_id,
        status="error",
        message=message,
        evidence={"automation_type": entry.automation_type, "reached_target": False},
    )


def make_mock_test_executor(outcomes: Mapping[str, ExecutionStatus] | None = None) -> NodeFn:
    """No I/O: api/ui tests get ``outcomes[test_case_id]`` (default ``passed``);
    manual tests are skipped."""
    configured = dict(outcomes or {})

    def test_executor(state: QAAgentState) -> dict[str, Any]:
        plan = state.get("test_plan")
        entries = sorted(plan.entries, key=lambda e: e.execution_order) if plan else []

        results = []
        for entry in entries:
            if entry.automation_type == "manual":
                status: ExecutionStatus = "skipped"
                message: str | None = MANUAL_MESSAGE
            else:
                status = configured.get(entry.test_case_id, "passed")
                message = None if status == "passed" else f"Mock {status} for {entry.test_case_id}"
            results.append(
                ExecutionResult(
                    test_case_id=entry.test_case_id,
                    status=status,
                    duration_ms=0 if status == "skipped" else 10,
                    message=message,
                    evidence={"automation_type": entry.automation_type, "mock": True},
                )
            )
        return {"execution_results": results}

    return test_executor
