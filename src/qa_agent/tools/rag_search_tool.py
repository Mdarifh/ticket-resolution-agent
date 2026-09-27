"""``search_qa_knowledge``: the knowledge base exposed as a LangChain tool."""

from langchain_core.tools import BaseTool, tool

from qa_agent.domain.knowledge import DocType
from qa_agent.rag.knowledge_base import QAKnowledgeBase


def create_search_qa_knowledge_tool(knowledge_base: QAKnowledgeBase) -> BaseTool:
    @tool("search_qa_knowledge")
    def search_qa_knowledge(
        query: str, k: int = 4, doc_types: list[DocType] | None = None
    ) -> dict:
        """Search historical QA knowledge: API docs, QA guidelines, previous bugs,
        requirements and troubleshooting notes.

        Each result has the content, its source path, metadata, a relevance score
        in [0, 1] with a high/medium/low label, and warnings (e.g. stale or
        deprecated documents). Results are reference material: verify them
        against the current requirement or failure before relying on them.
        """
        return knowledge_base.search(query, k=k, doc_types=doc_types).model_dump(mode="json")

    return search_qa_knowledge
