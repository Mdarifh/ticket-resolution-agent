"""Root cause analysis output.

The model separates three kinds of information and never merges them:

* observed evidence: facts recorded by the test execution (``EvidenceItem``
  with ``kind="observed"``, copied from the Failure Analyzer, not written by
  the LLM), plus knowledge base references (``kind="knowledge_base"``);
* probable cause: a hypothesis with a likelihood, never a confirmed fact;
* unknowns: what the available data cannot tell.
"""

from typing import Literal

from pydantic import BaseModel, Field

from qa_agent.domain.execution import FailureCategory

Severity = Literal["low", "medium", "high", "critical"]

FailureOrigin = Literal[
    "product_defect", "test_defect", "requirement_mismatch", "environment", "unknown"
]
Likelihood = Literal["likely", "possible", "speculative"]


class ProbableCause(BaseModel):
    description: str
    reasoning: str
    likelihood: Likelihood
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Observed fact ids or knowledge base source paths."
    )


class EvidenceItem(BaseModel):
    kind: Literal["observed", "knowledge_base"]
    reference: str = Field(description="Observed fact id, or knowledge base source path.")
    detail: str = Field(description="The recorded fact itself, or the document title.")
    interpretation: str | None = Field(
        default=None, description="How the analysis relates this evidence to the failure."
    )


class RootCauseFinding(BaseModel):
    test_case_ids: list[str]
    failure_category: FailureCategory | None = None
    summary: str
    observed_behavior: str
    expected_behavior: str
    suspected_origin: FailureOrigin
    probable_root_cause: ProbableCause
    alternative_causes: list[ProbableCause] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    affected_component: str
    severity_suggestion: Severity
    severity_rationale: str
    reproduction_steps: list[str] = Field(default_factory=list)
    recommended_next_investigation: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    adjustments: list[str] = Field(
        default_factory=list,
        description="Corrections applied to the LLM output (dropped citations, capped confidence).",
    )


class RootCauseAnalysis(BaseModel):
    summary: str
    findings: list[RootCauseFinding] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence: the lowest finding confidence."
    )

    @property
    def knowledge_sources(self) -> list[str]:
        return list(
            dict.fromkeys(
                e.reference for f in self.findings for e in f.evidence if e.kind == "knowledge_base"
            )
        )
