"""Index the knowledge base directory into the persistent vector store.

    python -m qa_agent.rag.ingestion            # upsert changed/new documents
    python -m qa_agent.rag.ingestion --rebuild  # drop the collection first

Uses EMBEDDING_PROVIDER / CHROMA_PERSIST_DIR / KNOWLEDGE_BASE_DIR from settings.
"""

import argparse
import logging

from qa_agent.config import get_settings
from qa_agent.logging_conf import configure_logging
from qa_agent.rag.knowledge_base import IndexReport, QAKnowledgeBase

logger = logging.getLogger(__name__)


def ingest(directory: str | None = None, *, rebuild: bool = False) -> IndexReport:
    settings = get_settings()
    knowledge_base = QAKnowledgeBase.from_settings(settings)
    if rebuild:
        knowledge_base.vector_store.reset_collection()
    return knowledge_base.index_directory(directory or settings.knowledge_base_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", help="Knowledge base directory (default: settings)")
    parser.add_argument("--rebuild", action="store_true", help="Drop existing chunks first")
    args = parser.parse_args()

    configure_logging()
    report = ingest(args.dir, rebuild=args.rebuild)
    logger.info(
        "Indexed %d documents into %d chunks", report.documents_indexed, report.chunks_indexed
    )
    for skipped in report.skipped:
        logger.warning("Skipped %s: %s", skipped.source, skipped.reason)
    for duplicate in report.duplicates:
        logger.warning("Duplicate %s of %s", duplicate.source, duplicate.duplicate_of)


if __name__ == "__main__":
    main()
