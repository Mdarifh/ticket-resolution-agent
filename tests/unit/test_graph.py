from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from qa_agent.domain import (
    BugReport,
    ConfidenceScore,
    FailureDetail,
    HumanReview,
    Requirement,
)
from qa_agent.graph import NODE_NAMES, build_qa_graph, initial_state, run_config
from qa_agent.graph.nodes.bug_report_generator import make_bug_report_generator
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.root_cause_analyzer import make_mock_root_cause_analyzer
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from qa_agent.graph.routing import (
    route_after_confidence_check,
    route_after_human_review,
    route_after_result_analysis,
)
from tests.conftest import (
    PASSWORD_RESET_REQUIREMENT,
    password_reset_responses,
    repo_knowledge_base,
)
from tests.fakes import FakeStructuredChatModel

# The canned password-reset LLM output yields TC-001..TC-008. Planned order:
# TC-001 api, TC-004 api, TC-002 ui, TC-006 api, TC-007 api, TC-005 ui,
# TC-003 manual (skipped by the executor), TC-008 api.

HAPPY_PATH = [
    "requirement_analyzer",
    "test_case_generator",
    "test_planner",
    "test_executor",
    "result_analyzer",
    "final_report_generator",
]
FAILURE_PATH_PREFIX = [
    "requirement_analyzer",
    "test_case_generator",
    "test_planner",
    "test_executor",
    "result_analyzer",
    "failure_analyzer",
    "root_cause_analyzer",
    "bug_report_generator",
    "confidence_checker",
]


def _graph(nodes=None) -> CompiledStateGraph:
    """Routing tests use the mock executor and LLM-free bug reports unless overridden."""
    llm = FakeStructuredChatModel(responses=password_reset_responses())
    nodes = {
        "test_executor": make_mock_test_executor(),
        "bug_report_generator": make_bug_report_generator(use_llm=False),
        **(nodes or {}),
    }
    return build_qa_graph(nodes, llm=llm, knowledge_base=repo_knowledge_base())


def _failing_graph(rca_confidence: float = 0.9) -> CompiledStateGraph:
    return _graph(
        {
            "test_executor": make_mock_test_executor({"TC-002": "failed", "TC-006": "error"}),
            "root_cause_analyzer": make_mock_root_cause_analyzer(confidence=rca_confidence),
            "confidence_checker": make_confidence_checker(threshold=0.7),
        }
    )


def _start(graph: CompiledStateGraph):
    requirement = Requirement(id=uuid4().hex, text=PASSWORD_RESET_REQUIREMENT)
    config = run_config(requirement.id)
    return graph.invoke(initial_state(requirement), config), config


def _visited(state) -> list[str]:
    return state["execution_metadata"].nodes_visited


# --- graph creation ----------------------------------------------------------


def test_graph_compiles_with_all_nodes():
    graph = _graph()

    assert isinstance(graph, CompiledStateGraph)
    assert set(graph.get_graph().nodes) == {*NODE_NAMES, "__start__", "__end__"}
    assert len(NODE_NAMES) == 11


def test_graph_edges_and_conditional_routes():
    edges = {(e.source, e.target, e.conditional) for e in _graph().get_graph().edges}

    assert edges == {
        ("__start__", "requirement_analyzer", False),
        ("requirement_analyzer", "test_case_generator", False),
        ("test_case_generator", "test_planner", False),
        ("test_planner", "test_executor", False),
        ("test_executor", "result_analyzer", False),
        ("result_analyzer", "failure_analyzer", True),
        ("result_analyzer", "final_report_generator", True),
        ("failure_analyzer", "root_cause_analyzer", False),
        ("root_cause_analyzer", "bug_report_generator", False),
        ("bug_report_generator", "confidence_checker", False),
        ("confidence_checker", "human_review", True),
        ("confidence_checker", "final_report_generator", True),
        ("human_review", "root_cause_analyzer", True),
        ("human_review", "final_report_generator", True),
        ("final_report_generator", "__end__", False),
    }


def test_graph_attaches_checkpointer_for_interrupts():
    assert _graph().checkpointer is not None


def test_unknown_node_override_is_rejected():
    with pytest.raises(ValueError, match="not_a_node"):
        _graph({"not_a_node": lambda state: {}})


# --- routing (conditional edge functions) ------------------------------------


def _failure() -> FailureDetail:
    return FailureDetail(test_case_id="TC-001", status="failed")


def _confidence(requires_review: bool) -> ConfidenceScore:
    return ConfidenceScore(
        score=0.5 if requires_review else 0.9, threshold=0.7, requires_human_review=requires_review
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"failures": []}, "final_report_generator"),
        ({}, "final_report_generator"),
        ({"failures": [_failure()]}, "failure_analyzer"),
    ],
)
def test_route_after_result_analysis(state, expected):
    assert route_after_result_analysis(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"confidence_score": _confidence(requires_review=False)}, "final_report_generator"),
        ({"confidence_score": _confidence(requires_review=True)}, "human_review"),
        ({}, "human_review"),
    ],
)
def test_route_after_confidence_check(state, expected):
    assert route_after_confidence_check(state) == expected


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("APPROVE", "final_report_generator"),
        ("REJECT", "final_report_generator"),
        ("REQUEST_REANALYSIS", "root_cause_analyzer"),
        (None, "final_report_generator"),
    ],
)
def test_route_after_human_review(decision, expected):
    review = HumanReview(
        review_id="r-1", reason="r", ai_recommendation="a", confidence=0.5, proposed_action="p",
        requested_at=datetime.now(UTC), human_decision=decision,
    )
    state = {"human_review": review}
    assert route_after_human_review(state) == expected


# --- state transitions -------------------------------------------------------


def test_stream_emits_one_update_per_node_in_order():
    graph = _graph()
    requirement = Requirement(text=PASSWORD_RESET_REQUIREMENT)

    chunks = list(
        graph.stream(initial_state(requirement), run_config(requirement.id), stream_mode="updates")
    )

    assert [next(iter(chunk)) for chunk in chunks] == HAPPY_PATH


def test_each_node_populates_its_state_slice():
    state, _ = _start(_graph())

    assert len(state["requirement_analysis"].acceptance_criteria) == 4
    assert [tc.test_case_id for tc in state["test_cases"]] == [f"TC-{i:03d}" for i in range(1, 9)]
    assert [e.execution_order for e in state["test_plan"].entries] == list(range(1, 9))
    assert [r.status for r in state["execution_results"]].count("skipped") == 1
    assert len(state["execution_results"]) == 8
    assert state["failures"] == []


def test_paused_run_is_checkpointed_at_human_review():
    graph = _failing_graph(rca_confidence=0.3)
    state, config = _start(graph)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert "final_report" not in state
    [pending] = state["__interrupt__"]
    assert pending.value["review"]["reason"] == "Confidence 0.30 is below threshold 0.70"
    assert [r["related_test_case_ids"] for r in pending.value["bug_reports"]] == [["TC-002"], ["TC-006"]]


def test_node_exception_is_recorded_and_run_still_completes():
    def broken_executor(state):
        raise RuntimeError("executor exploded")

    state, _ = _start(_graph({"test_executor": broken_executor}))

    [error] = state["errors"]
    assert error.node == "test_executor"
    assert error.error_type == "RuntimeError"
    assert state["final_report"].status == "error"
    assert _visited(state) == HAPPY_PATH


def test_invalid_review_payload_is_recorded_as_error():
    graph = _failing_graph(rca_confidence=0.3)
    _, config = _start(graph)

    state = graph.invoke(Command(resume={"decision": "maybe"}), config)

    assert [e.node for e in state["errors"]] == ["human_review"]
    assert state["final_report"].status == "error"


# --- success path ------------------------------------------------------------


def test_success_path_goes_straight_to_final_report():
    state, _ = _start(_graph())

    assert _visited(state) == HAPPY_PATH
    report = state["final_report"]
    assert report.status == "passed"
    assert (report.total_tests, report.passed, report.failed, report.skipped) == (7, 7, 0, 1)
    assert report.pass_rate == 1.0
    assert report.bug_reports == []
    assert state.get("bug_reports") in (None, [])
    assert state["errors"] == []
    assert state["execution_metadata"].status == "completed"
    assert state["execution_metadata"].completed_at is not None


def test_success_path_final_snapshot_has_no_pending_nodes():
    graph = _graph()
    _, config = _start(graph)

    assert graph.get_state(config).next == ()


# --- failure path ------------------------------------------------------------


def test_failure_path_with_high_confidence_skips_human_review():
    state, _ = _start(_failing_graph(rca_confidence=0.9))

    assert _visited(state) == [*FAILURE_PATH_PREFIX, "final_report_generator"]
    assert [(f.test_case_id, f.category) for f in state["failures"]] == [
        ("TC-002", "assertion_mismatch"),
        ("TC-006", "unknown"),  # an error with no details is not blamed on anything
    ]
    assert state["confidence_score"].requires_human_review is False
    report = state["final_report"]
    assert report.status == "failed"
    assert (report.total_tests, report.passed, report.failed, report.skipped) == (7, 5, 2, 1)
    assert all(isinstance(r, BugReport) for r in report.bug_reports)
    assert [r.related_test_case_ids for r in report.bug_reports] == [["TC-002"], ["TC-006"]]
    assert report.human_decision is None


def test_failure_path_low_confidence_pauses_then_approve_completes():
    graph = _failing_graph(rca_confidence=0.3)
    _, config = _start(graph)

    state = graph.invoke(Command(resume={"decision": "approve", "reviewer": "qa-lead"}), config)

    assert _visited(state) == [*FAILURE_PATH_PREFIX, "human_review", "final_report_generator"]
    assert state["human_review"].reviewer == "qa-lead"
    assert state["final_report"].status == "failed"
    assert state["final_report"].human_decision == "APPROVE"
    assert graph.get_state(config).next == ()


def test_failure_path_reject_produces_rejected_report():
    graph = _failing_graph(rca_confidence=0.3)
    _, config = _start(graph)

    state = graph.invoke(Command(resume={"decision": "reject", "comment": "false alarm"}), config)

    assert state["final_report"].status == "rejected"
    assert state["final_report"].human_decision == "REJECT"


def test_failure_path_reanalysis_loops_back_to_root_cause_analysis():
    graph = _failing_graph(rca_confidence=0.3)
    _, config = _start(graph)

    paused_again = graph.invoke(
        Command(resume={"decision": "REQUEST_REANALYSIS", "comment": "add logs"}), config
    )

    assert graph.get_state(config).next == ("human_review",)
    assert {r.revision for r in paused_again["bug_reports"]} == {1}
    assert paused_again["bug_reports"][0].reviewer_feedback == ["add logs"]

    state = graph.invoke(Command(resume={"decision": "approve"}), config)

    assert _visited(state) == [
        *FAILURE_PATH_PREFIX,
        "human_review",
        "root_cause_analyzer",
        "bug_report_generator",
        "confidence_checker",
        "human_review",
        "final_report_generator",
    ]
    assert state["final_report"].status == "failed"
    assert {r.revision for r in state["final_report"].bug_reports} == {1}


def test_high_severity_forces_review_even_with_high_confidence():
    def critical_bug_report(state):
        update = make_bug_report_generator(use_llm=False)(state)
        return {"bug_reports": [r.model_copy(update={"severity": "critical"}) for r in update["bug_reports"]]}

    graph = _graph(
        {
            "test_executor": make_mock_test_executor({"TC-001": "failed"}),
            "root_cause_analyzer": make_mock_root_cause_analyzer(confidence=0.95),
            "bug_report_generator": critical_bug_report,
            "confidence_checker": make_confidence_checker(threshold=0.7),
        }
    )
    state, config = _start(graph)

    assert graph.get_state(config).next == ("human_review",)
    assert state["confidence_score"].reasons == [
        "Sensitive action requires approval: report_high_severity_bugs (BR-1)"
    ]
