"""Canonical memory record and projection types."""
from __future__ import annotations

from typing import Any, TypedDict


class MemoryBlobRef(TypedDict, total=False):
    """Reference to a large memory payload stored outside the document record."""

    blob_id: str
    name: str
    uri: str
    artifact_id: str
    content_type: str
    metadata: dict[str, Any]


class MemoryEntity(TypedDict, total=False):
    """Semantic graph entity derived from a memory record."""

    entity_type: str
    key: str
    name: str
    metadata: dict[str, Any]


class MemoryRelation(TypedDict, total=False):
    """Semantic graph relation derived from a memory record."""

    relation_type: str
    source_type: str
    source_key: str
    target_type: str
    target_key: str
    metadata: dict[str, Any]


class MemoryRecord(TypedDict, total=False):
    """Canonical domain-agnostic memory record."""

    record_id: str
    domain: str
    kind: str
    object_type: str
    object_key: str
    object_role: str
    schema_version: str
    title: str
    summary: str
    content: dict[str, Any]
    metadata: dict[str, Any]
    tags: list[str]
    created_at: str
    source_run_id: str | None
    source_record_id: str | None
    blob_refs: list[MemoryBlobRef]
    entities: list[MemoryEntity]
    relations: list[MemoryRelation]


class MemoryProjection(TypedDict, total=False):
    """Backend-facing projection derived from a memory record."""

    embedding_text: str
    vector_metadata: dict[str, Any]
    graph_nodes: list[dict[str, Any]]
    graph_edges: list[dict[str, Any]]


class MemorySearchHit(TypedDict, total=False):
    """Hydrated memory retrieval result."""

    record: MemoryRecord
    distance: float | None
    document: str
    vector_metadata: dict[str, Any]
