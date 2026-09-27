"""Requirement input and its structured analysis.

``RequirementAnalysis`` is also the LLM structured-output schema, so field
descriptions double as instructions to the model.
"""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequirementAnalysis(BaseModel):
    summary: str = Field(description="One or two sentence restatement of what the requirement asks for.")
    feature: str = Field(description="Short name of the feature or component under test.")
    actors: list[str] = Field(description="Users or systems that interact with the feature.")
    preconditions: list[str] = Field(
        description="State that must hold before the feature can be used."
    )
    acceptance_criteria: list[str] = Field(
        description="Atomic, testable statements of expected behaviour."
    )
    business_rules: list[str] = Field(
        description="Constraints and rules the feature must enforce."
    )
    data_inputs: list[str] = Field(description="Input fields or data the feature consumes.")
    edge_cases: list[str] = Field(
        description="Unusual or boundary situations worth testing."
    )
    risk_areas: list[str] = Field(
        description="Security, reliability or usability risks raised by the requirement."
    )
    ambiguities: list[str] = Field(
        description="Details the requirement leaves unspecified; do not invent answers for them."
    )
