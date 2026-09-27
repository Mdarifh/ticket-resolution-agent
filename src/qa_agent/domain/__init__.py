"""Framework-agnostic domain models (no LangChain/LangGraph imports)."""

from qa_agent.domain.analysis import (
    EvidenceItem,
    ProbableCause,
    RootCauseAnalysis,
    RootCauseFinding,
)
from qa_agent.domain.bug_report import BugEnvironment, BugReport
from qa_agent.domain.bug_ticket import BugTicket, TicketVerdict
from qa_agent.domain.confidence import ConfidenceScore
from qa_agent.domain.execution import ExecutionResult, FailureDetail, ObservedFact
from qa_agent.domain.knowledge import KnowledgeSearchResult, KnowledgeUsage, RetrievedKnowledge
from qa_agent.domain.qa_report import FinalReport
from qa_agent.domain.requirement import Requirement, RequirementAnalysis
from qa_agent.domain.review import HumanReview, HumanReviewDecision
from qa_agent.domain.test_case import TestCase, TestDataItem, TestSuite
from qa_agent.domain.test_plan import PlannedTest, TestExecutionPlan
from qa_agent.domain.workflow import ExecutionMetadata, WorkflowError

__all__ = [
    "BugEnvironment",
    "BugReport",
    "BugTicket",
    "ConfidenceScore",
    "EvidenceItem",
    "ExecutionMetadata",
    "ExecutionResult",
    "FailureDetail",
    "FinalReport",
    "HumanReview",
    "HumanReviewDecision",
    "KnowledgeSearchResult",
    "KnowledgeUsage",
    "ObservedFact",
    "PlannedTest",
    "ProbableCause",
    "Requirement",
    "RequirementAnalysis",
    "RetrievedKnowledge",
    "RootCauseAnalysis",
    "RootCauseFinding",
    "TestCase",
    "TestDataItem",
    "TestExecutionPlan",
    "TestSuite",
    "TicketVerdict",
    "WorkflowError",
]
