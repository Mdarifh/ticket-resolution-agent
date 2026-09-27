"""Execution plan: how and in what order each test case runs."""

from typing import ClassVar

from pydantic import BaseModel, Field

from qa_agent.domain.test_case import AutomationType


class PlannedTest(BaseModel):
    test_case_id: str
    automation_type: AutomationType
    execution_order: int = Field(ge=1)
    rationale: str


class TestExecutionPlan(BaseModel):
    __test__: ClassVar[bool] = False  # not a pytest test class

    strategy: str
    entries: list[PlannedTest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def ids_for(self, automation_type: AutomationType) -> list[str]:
        """Test case ids of one automation type, in execution order."""
        ordered = sorted(self.entries, key=lambda e: e.execution_order)
        return [e.test_case_id for e in ordered if e.automation_type == automation_type]
