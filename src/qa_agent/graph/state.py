"""Typed LangGraph state for the QA agent workflow.

Scalar fields are overwritten by whichever node returns them. Two fields use
reducers so that LangGraph merges updates instead of replacing them:

* ``errors``, ``retrieved_knowledge`` and ``review_history`` accumulate across
  nodes (``operator.add``).
* ``execution_metadata`` is patched field-by-field, with ``nodes_visited``
  appended, so every node can record itself without clobbering run metadata.
"""

import operator
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from qa_agent.domain import (
    BugReport,
    ConfidenceScore,
    ExecutionMetadata,
    ExecutionResult,
    FailureDetail,
    FinalReport,
    HumanReview,
    KnowledgeUsage,
    Requirement,
    RequirementAnalysis,
    RootCauseAnalysis,
    TestCase,
    TestExecutionPlan,
    WorkflowError,
)


def merge_execution_metadata(
    current: ExecutionMetadata | None,
    update: ExecutionMetadata | Mapping[str, Any] | None,
) -> ExecutionMetadata:
    """Reducer: apply only the fields present in ``update``; append nodes_visited."""
    if current is None:
        current = ExecutionMetadata()
    if update is None:
        return current

    if isinstance(update, ExecutionMetadata):
        patch = update.model_dump(exclude_unset=True)
    else:
        patch = dict(update)

    visited = patch.pop("nodes_visited", [])
    return current.model_copy(
        update={**patch, "nodes_visited": [*current.nodes_visited, *visited]}
    )


class QAAgentState(TypedDict, total=False):
    requirement: Requirement
    requirement_analysis: RequirementAnalysis | None
    test_cases: list[TestCase]
    test_plan: TestExecutionPlan | None
    execution_results: list[ExecutionResult]
    failures: list[FailureDetail]
    root_cause_analysis: RootCauseAnalysis | None
    bug_reports: list[BugReport]
    confidence_score: ConfidenceScore | None
    human_review: HumanReview | None
    final_report: FinalReport | None
    errors: Annotated[list[WorkflowError], operator.add]
    retrieved_knowledge: Annotated[list[KnowledgeUsage], operator.add]
    review_history: Annotated[list[HumanReview], operator.add]
    execution_metadata: Annotated[ExecutionMetadata, merge_execution_metadata]


def initial_state(requirement: Requirement) -> QAAgentState:
    """Input state for a new run."""
    return QAAgentState(
        requirement=requirement,
        execution_metadata=ExecutionMetadata(run_id=requirement.id),
    )
