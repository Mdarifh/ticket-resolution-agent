"""Failure Analyzer node: execution evidence -> observed facts + failure category.

Deterministic (no LLM): see ``qa_agent.analysis.failure_facts``.
"""

from typing import Any

from qa_agent.analysis.failure_facts import analyze_failure
from qa_agent.domain import ExecutionResult
from qa_agent.graph.state import QAAgentState


def failure_analyzer(state: QAAgentState) -> dict[str, Any]:
    results = {r.test_case_id: r for r in state.get("execution_results") or []}
    test_cases = {tc.test_case_id: tc for tc in state.get("test_cases") or []}

    analyzed = []
    for failure in state.get("failures") or []:
        result = results.get(failure.test_case_id) or ExecutionResult(
            test_case_id=failure.test_case_id, status=failure.status, message=failure.message
        )
        analyzed.append(analyze_failure(result, test_cases.get(failure.test_case_id)))
    return {"failures": analyzed}
