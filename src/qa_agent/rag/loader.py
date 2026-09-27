"""Load knowledge base files into LangChain ``Document``s.

Each file is Markdown/text with optional YAML front matter::

    ---
    title: Password reset API
    doc_type: api_doc
    component: auth
    ---
    # body...

``doc_type`` falls back to the parent folder (``api_docs/`` -> ``api_doc``) and
``title`` to the first ``# heading`` or the file name. Files that cannot be
used are skipped with a reason instead of aborting the load, and files whose
body duplicates an earlier file are skipped as duplicates.
"""

import hashlib
import logging
import re
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any, get_args

import yaml
from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from qa_agent.domain.knowledge import DocType

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".md", ".txt")
DOC_TYPES: frozenset[str] = frozenset(get_args(DocType))
FOLDER_DOC_TYPES = {
    "api_docs": "api_doc",
    "qa_guidelines": "qa_guideline",
    "previous_bugs": "previous_bug",
    "requirements": "requirement",
    "troubleshooting": "troubleshooting",
}

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class SkippedDocument(BaseModel):
    source: str
    reason: str


class DuplicateDocument(BaseModel):
    source: str
    duplicate_of: str


class LoadReport(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    skipped: list[SkippedDocument] = Field(default_factory=list)
    duplicates: list[DuplicateDocument] = Field(default_factory=list)


class MalformedDocumentError(ValueError):
    pass


class TextFileLoader(BaseLoader):
    """Minimal LangChain document loader for one UTF-8 text/Markdown file.

    Built on ``langchain_core`` so the project does not depend on the
    deprecated ``langchain-community`` package for a one-file read.
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        blob = Blob.from_path(self.path, encoding=self.encoding)
        yield Document(page_content=blob.as_string(), metadata={"source": str(self.path)})


def load_knowledge_documents(directory: str | Path) -> LoadReport:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {root}")

    report = LoadReport()
    seen_hashes: dict[str, str] = {}

    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES):
        source = path.relative_to(root).as_posix()
        try:
            document = _load_file(path, source)
        except MalformedDocumentError as exc:
            logger.warning("Skipping %s: %s", source, exc)
            report.skipped.append(SkippedDocument(source=source, reason=str(exc)))
            continue

        content_hash = document.metadata["content_hash"]
        if content_hash in seen_hashes:
            report.duplicates.append(
                DuplicateDocument(source=source, duplicate_of=seen_hashes[content_hash])
            )
            continue
        seen_hashes[content_hash] = source
        report.documents.append(document)

    return report


def _load_file(path: Path, source: str) -> Document:
    try:
        [raw] = TextFileLoader(path).load()
    except (OSError, UnicodeDecodeError) as exc:
        raise MalformedDocumentError(f"unreadable file ({exc})") from exc

    front_matter, body = _split_front_matter(raw.page_content)
    body = body.strip()
    if not body:
        raise MalformedDocumentError("document has no content")

    doc_type = front_matter.get("doc_type") or FOLDER_DOC_TYPES.get(path.parent.name)
    if doc_type not in DOC_TYPES:
        raise MalformedDocumentError(
            f"unknown doc_type {doc_type!r}; expected one of {sorted(DOC_TYPES)}"
        )

    heading = _HEADING.search(body)
    title = front_matter.get("title") or (heading.group(1).strip() if heading else path.stem)

    metadata: dict[str, Any] = {
        key: _scalar(value) for key, value in front_matter.items() if value is not None
    }
    metadata.update(
        source=source,
        title=str(title),
        doc_type=doc_type,
        content_hash=hashlib.sha256(_normalize(body).encode()).hexdigest(),
    )
    return Document(page_content=body, metadata=metadata)


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    text = text.lstrip("﻿")
    match = _FRONT_MATTER.match(text)
    if not match:
        if text.startswith("---"):
            raise MalformedDocumentError("front matter is not closed with '---'")
        return {}, text

    try:
        parsed = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise MalformedDocumentError(f"invalid front matter YAML ({exc})") from exc
    if not isinstance(parsed, dict):
        raise MalformedDocumentError("front matter must be a key/value mapping")
    return {str(k): v for k, v in parsed.items()}, text[match.end() :]


def _scalar(value: Any) -> str | int | float | bool:
    """Vector store metadata must be scalar: flatten lists, stringify dates/objects."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list | tuple | set):
        return ", ".join(str(v) for v in value)
    return str(value)


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()
