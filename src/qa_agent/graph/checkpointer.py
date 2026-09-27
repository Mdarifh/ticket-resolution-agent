"""Checkpointer factory.

State holds Pydantic domain models, which LangGraph serializes into
checkpoints. Newer LangGraph only deserializes explicitly allow-listed
types, so every domain model is registered here. The Postgres checkpointer
will reuse the same serializer.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import qa_agent.domain as domain

DOMAIN_TYPES: tuple[type, ...] = tuple(getattr(domain, name) for name in domain.__all__)


def create_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=DOMAIN_TYPES)


def create_default_checkpointer() -> BaseCheckpointSaver:
    return InMemorySaver(serde=create_serializer())
