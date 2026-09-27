"""Result Analyzer node: extracts failed/errored results for the failure path.

Deterministic; the pass/fail routing decision itself is made by the graph's
conditional edge (``routing.route_after_result_analysis``).

Errored tests that never reached the system under test (missing configuration,
unsafe target, no request could be planned, browser did not start) are not
failures of the product, so they stay out of root cause analysis and bug
reports; the final report counts them as errors.
"""

from typing import Any

from qa_agent.domain import FailureDetail
from qa_agent.graph.state import QAAgentState


def result_analyzer(state: QAAgentState) -> dict[str, Any]:
    failures = [
        FailureDetail(test_case_id=r.test_case_id, status=r.status, message=r.message)
        for r in state.get("execution_results") or []
        if r.status == "failed" or (r.status == "error" and r.evidence.get("reached_target", True))
    ]
    return {"failures": failures}
