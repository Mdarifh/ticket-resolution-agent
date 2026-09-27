"""Results of searching the QA knowledge base, and an audit of how they were used."""

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["api_doc", "qa_guideline", "previous_bug", "requirement", "troubleshooting"]
Relevance = Literal["high", "medium", "low"]
MetadataValue = str | int | float | bool


class RetrievedKnowledge(BaseModel):
    content: str
    source: str = Field(description="Path of the source document inside the knowledge base.")
    chunk_id: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    relevance_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Similarity to the query; higher is closer."
    )
    relevance: Relevance | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Reasons to be cautious with this document (stale, deprecated, ...).",
    )


class KnowledgeSearchResult(BaseModel):
    query: str
    results: list[RetrievedKnowledge] = Field(default_factory=list)
    message: str | None = Field(
        default=None, description="Why the result is empty or degraded, if it is."
    )

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(r.source for r in self.results))


class KnowledgeUsage(BaseModel):
    """Audit record: what a node retrieved and which sources the LLM actually cited."""

    node: str
    query: str
    retrieved_sources: list[str] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    discarded_citations: list[str] = Field(
        default_factory=list, description="Citations of sources that were not retrieved."
    )
    message: str | None = None
