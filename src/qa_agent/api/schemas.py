"""HTTP request/response schemas.

Run-level views (summary, node progress, system status) are API contracts
defined here. Workflow artifacts (test cases, results, bug reports, ...) are
returned as their domain models, which are already plain Pydantic data.
"""

from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from qa_agent.domain import HumanReview, KnowledgeUsage, RootCauseAnalysis, WorkflowError
from qa_agent.domain.bug_ticket import BugTicket, TicketVerdict

RunStatus = Literal["running", "awaiting_execution", "awaiting_review", "completed", "error"]
NodeStatus = Literal["pending", "running", "waiting", "completed", "error", "skipped"]


class SystemStatus(BaseModel):
    app_env: str
    llm_configured: bool
    llm_model: str
    embedding_provider: str
    database_configured: bool
    api_test_base_url: str | None
    ui_test_base_url: str | None
    confidence_threshold: float
    langsmith_tracing: bool
    langsmith_project: str | None
    log_format: str
    warnings: list[str] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value


class ProjectOut(BaseModel):
    name: str
    description: str | None = None
    run_count: int = 0


class RunCreate(BaseModel):
    project: str = Field(default="default", min_length=1, max_length=200)
    requirement: str = Field(default="", max_length=20_000)
    bug_ticket: BugTicket | None = Field(
        default=None,
        description="Service desk bug ticket to verify; replaces the requirement text.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    pause_before_execution: bool = Field(
        default=True, description="Stop after test planning until POST /runs/{id}/execute."
    )

    @model_validator(mode="after")
    def _requirement_or_ticket(self) -> "RunCreate":
        self.requirement = self.requirement.strip()
        if self.bug_ticket is None and not self.requirement:
            raise ValueError("requirement must not be blank (or send a bug_ticket)")
        return self


class TestCounts(BaseModel):
    __test__: ClassVar[bool] = False  # not a pytest test class

    total: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0


class NodeProgress(BaseModel):
    name: str
    status: NodeStatus
    visits: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None


class RunSummary(BaseModel):
    run_id: str
    project: str
    requirement: str
    status: RunStatus
    outcome: str | None = Field(default=None, description="Final report status once completed.")
    current_nodes: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    pause_before_execution: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    test_case_count: int = 0
    test_counts: TestCounts = Field(default_factory=TestCounts)
    confidence: float | None = None
    bug_count: int = 0
    node_error_count: int = 0
    verdict: TicketVerdict | None = Field(
        default=None, description="Bug ticket verification verdict (bug_verification runs only)."
    )


class TraceInfo(BaseModel):
    """One LangSmith trace: a workflow segment of the run."""

    segment: str
    trace_id: str
    started_at: datetime
    url: str | None = Field(default=None, description="Link to the trace in LangSmith, when resolvable.")


class RunDetail(RunSummary):
    nodes: list[NodeProgress] = Field(default_factory=list)
    errors: list[WorkflowError] = Field(default_factory=list)
    traces: list[TraceInfo] = Field(default_factory=list)
    langsmith_tracing: bool = False
    langsmith_project: str | None = None


class RootCauseOut(BaseModel):
    analysis: RootCauseAnalysis | None = None
    knowledge_usage: list[KnowledgeUsage] = Field(default_factory=list)


class ReviewOut(BaseModel):
    pending: HumanReview | None = None
    latest: HumanReview | None = None
    history: list[HumanReview] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
