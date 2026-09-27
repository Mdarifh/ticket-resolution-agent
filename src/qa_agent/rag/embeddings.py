"""Embedding model factory.

``openai`` is the production choice. ``hashing`` is a deterministic, offline
bag-of-words embedding: it only captures word overlap, but needs no API key,
which makes it suitable for tests and local demos.
"""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from qa_agent.config import Settings, get_settings
from qa_agent.errors import LLMConfigurationError

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can for from has have in is it its of on or should "
    "that the their this to was were will with".split()
)


class HashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # Chroma rejects zero vectors under cosine distance; use a fixed unit vector.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddings()
    if not settings.openai_api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set; set it or use EMBEDDING_PROVIDER=hashing."
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
