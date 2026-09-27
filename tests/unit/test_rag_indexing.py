from pathlib import Path

import pytest
from langchain_core.documents import Document

from qa_agent.rag.loader import load_knowledge_documents
from tests.conftest import KNOWLEDGE_BASE_DIR, make_knowledge_base

EXPECTED_DOC_TYPES = {"api_doc", "qa_guideline", "previous_bug", "requirement", "troubleshooting"}


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


VALID_BUG = """---
title: Checkout total rounding error
doc_type: previous_bug
bug_id: BUG-7
severity: medium
last_updated: 2026-01-05
tags: [checkout, rounding]
---
# BUG-7

Totals were rounded per line item instead of once per order.
"""


# --- loading -----------------------------------------------------------------


def test_repository_knowledge_base_loads_cleanly():
    report = load_knowledge_documents(KNOWLEDGE_BASE_DIR)

    assert len(report.documents) == 14
    assert report.skipped == []
    assert report.duplicates == []
    assert {d.metadata["doc_type"] for d in report.documents} == EXPECTED_DOC_TYPES


def test_front_matter_becomes_scalar_metadata(tmp_path):
    _write(tmp_path, "previous_bugs/bug-7.md", VALID_BUG)

    [document] = load_knowledge_documents(tmp_path).documents

    assert document.metadata["source"] == "previous_bugs/bug-7.md"
    assert document.metadata["title"] == "Checkout total rounding error"
    assert document.metadata["bug_id"] == "BUG-7"
    assert document.metadata["tags"] == "checkout, rounding"
    assert document.metadata["last_updated"] == "2026-01-05"
    assert len(document.metadata["content_hash"]) == 64
    assert not document.page_content.startswith("---")


def test_doc_type_and_title_are_inferred_without_front_matter(tmp_path):
    _write(tmp_path, "troubleshooting/cache.md", "# Cache misses\n\nFlush the CDN.\n")

    [document] = load_knowledge_documents(tmp_path).documents

    assert document.metadata["doc_type"] == "troubleshooting"
    assert document.metadata["title"] == "Cache misses"


@pytest.mark.parametrize(
    ("relative", "content", "reason"),
    [
        ("qa_guidelines/empty.md", "", "no content"),
        ("qa_guidelines/blank.md", "   \n\n  ", "no content"),
        ("qa_guidelines/only-front-matter.md", "---\ntitle: x\n---\n", "no content"),
        ("qa_guidelines/bad-yaml.md", "---\ntitle: [unclosed\n---\nbody", "invalid front matter"),
        ("qa_guidelines/unclosed.md", "---\ntitle: x\nbody without end", "not closed"),
        ("qa_guidelines/list-front-matter.md", "---\n- a\n- b\n---\nbody", "key/value mapping"),
        ("misc/unknown-folder.md", "# Notes\n\nSomething", "unknown doc_type"),
        ("api_docs/bad-type.md", "---\ndoc_type: memo\n---\nbody", "unknown doc_type"),
        ("api_docs/latin1.md", "caf\xe9 menu".encode("latin-1"), "unreadable"),
    ],
)
def test_malformed_documents_are_skipped_with_reason(tmp_path, relative, content, reason):
    _write(tmp_path, "previous_bugs/good.md", VALID_BUG)
    _write(tmp_path, relative, content)

    report = load_knowledge_documents(tmp_path)

    assert [d.metadata["source"] for d in report.documents] == ["previous_bugs/good.md"]
    [skipped] = report.skipped
    assert skipped.source == relative
    assert reason in skipped.reason


def test_unsupported_file_types_are_ignored(tmp_path):
    _write(tmp_path, "previous_bugs/good.md", VALID_BUG)
    _write(tmp_path, "previous_bugs/screenshot.png", b"\x89PNG\r\n")

    report = load_knowledge_documents(tmp_path)

    assert len(report.documents) == 1
    assert report.skipped == []


def test_duplicate_documents_are_loaded_once(tmp_path):
    _write(tmp_path, "previous_bugs/a.md", VALID_BUG)
    # Same body, different whitespace and case: still a duplicate.
    _write(tmp_path, "previous_bugs/b.md", VALID_BUG.replace("Totals were", "TOTALS   were"))

    report = load_knowledge_documents(tmp_path)

    assert [d.metadata["source"] for d in report.documents] == ["previous_bugs/a.md"]
    [duplicate] = report.duplicates
    assert (duplicate.source, duplicate.duplicate_of) == ("previous_bugs/b.md", "previous_bugs/a.md")


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_knowledge_documents(tmp_path / "nope")


# --- indexing ----------------------------------------------------------------


def test_index_directory_chunks_every_document():
    knowledge_base = make_knowledge_base(chunk_size=400, chunk_overlap=50)

    report = knowledge_base.index_directory(KNOWLEDGE_BASE_DIR)

    assert report.documents_indexed == 14
    assert report.chunks_indexed > 14  # long documents are split
    assert knowledge_base.count() == report.chunks_indexed


def test_chunks_carry_source_and_chunk_metadata():
    knowledge_base = make_knowledge_base(chunk_size=400, chunk_overlap=50)
    knowledge_base.index_directory(KNOWLEDGE_BASE_DIR)

    stored = knowledge_base.vector_store.get(include=["metadatas"])

    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"], strict=True):
        assert metadata["chunk_id"] == chunk_id
        assert {"source", "title", "doc_type", "chunk_index", "content_hash"} <= set(metadata)
    api_chunks = [m for m in stored["metadatas"] if m["source"] == "api_docs/auth-password-reset-api.md"]
    assert sorted(m["chunk_index"] for m in api_chunks) == list(range(len(api_chunks)))


def test_reindexing_is_idempotent():
    knowledge_base = make_knowledge_base()
    first = knowledge_base.index_directory(KNOWLEDGE_BASE_DIR)

    knowledge_base.index_directory(KNOWLEDGE_BASE_DIR)

    assert knowledge_base.count() == first.chunks_indexed


def test_reindexing_a_changed_document_replaces_its_chunks(tmp_path):
    path = _write(tmp_path, "previous_bugs/bug-7.md", VALID_BUG)
    knowledge_base = make_knowledge_base()
    knowledge_base.index_directory(tmp_path)

    path.write_text(VALID_BUG.replace("per line item", "per basket row"), encoding="utf-8")
    knowledge_base.index_directory(tmp_path)

    stored = knowledge_base.vector_store.get(include=["documents"])
    assert len(stored["ids"]) == 1
    assert "per basket row" in stored["documents"][0]


def test_duplicates_and_malformed_files_are_reported_not_indexed(tmp_path):
    _write(tmp_path, "previous_bugs/a.md", VALID_BUG)
    _write(tmp_path, "previous_bugs/b.md", VALID_BUG)
    _write(tmp_path, "previous_bugs/broken.md", "---\ntitle: [\n---\nbody")
    knowledge_base = make_knowledge_base()

    report = knowledge_base.index_directory(tmp_path)

    assert report.documents_indexed == 1
    assert [d.source for d in report.duplicates] == ["previous_bugs/b.md"]
    assert [s.source for s in report.skipped] == ["previous_bugs/broken.md"]
    assert knowledge_base.count() == 1


def test_index_documents_with_no_documents_is_a_no_op():
    knowledge_base = make_knowledge_base()

    report = knowledge_base.index_documents([])

    assert (report.documents_indexed, report.chunks_indexed) == (0, 0)
    assert knowledge_base.count() == 0


def test_identical_content_indexed_twice_does_not_duplicate_chunks():
    knowledge_base = make_knowledge_base()
    document = Document(
        page_content="Reset tokens expire after 30 minutes.",
        metadata={"source": "api_docs/x.md", "doc_type": "api_doc", "title": "x", "content_hash": "ab" * 32},
    )

    knowledge_base.index_documents([document])
    knowledge_base.index_documents([document.model_copy(deep=True)])

    assert knowledge_base.count() == 1
