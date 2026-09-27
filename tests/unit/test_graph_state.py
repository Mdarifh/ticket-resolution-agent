from datetime import UTC, datetime

from qa_agent.domain import ExecutionMetadata, Requirement
from qa_agent.graph.state import initial_state, merge_execution_metadata


def test_initial_state_carries_requirement_and_run_id():
    requirement = Requirement(text="Users can log in.")

    state = initial_state(requirement)

    assert state["requirement"] is requirement
    assert state["execution_metadata"].run_id == requirement.id
    assert state["execution_metadata"].nodes_visited == []


def test_merge_metadata_appends_nodes_visited():
    current = ExecutionMetadata(run_id="r1", nodes_visited=["a"])

    merged = merge_execution_metadata(current, {"nodes_visited": ["b"]})

    assert merged.nodes_visited == ["a", "b"]
    assert merged.run_id == "r1"


def test_merge_metadata_patches_only_supplied_fields():
    started = datetime(2026, 1, 1, tzinfo=UTC)
    finished = datetime(2026, 1, 2, tzinfo=UTC)
    current = ExecutionMetadata(run_id="r1", started_at=started)

    merged = merge_execution_metadata(current, {"status": "completed", "completed_at": finished})

    assert merged.status == "completed"
    assert merged.completed_at == finished
    assert merged.started_at == started
    assert merged.run_id == "r1"


def test_merge_metadata_model_update_ignores_unset_defaults():
    current = ExecutionMetadata(run_id="r1", nodes_visited=["a"])

    merged = merge_execution_metadata(current, ExecutionMetadata(nodes_visited=["b"]))

    assert merged.run_id == "r1"
    assert merged.nodes_visited == ["a", "b"]


def test_merge_metadata_handles_missing_current():
    merged = merge_execution_metadata(None, ExecutionMetadata(run_id="r1"))

    assert merged.run_id == "r1"
    assert merged.nodes_visited == []
