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


class MemoryLineage(TypedDict, total=False):
    """How a memory record was produced and what it depends on."""

    node: str
    run_id: str
    source_state_keys: list[str]
    source_record_ids: list[str]
    input_fingerprints: dict[str, str]
    code_fingerprints: dict[str, str]
    config_fingerprint: str
    parent_record_ids: list[str]


class MemoryValidity(TypedDict, total=False):
    """Reuse and freshness metadata for a memory record."""

    status: str
    reusable: bool
    expires_at: str | None
    checked_at: str | None
    invalidated_at: str | None
    invalidated_reason: str
    superseded_by: str
    checks: dict[str, Any]


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
    lineage: MemoryLineage
    validity: MemoryValidity
    blob_refs: list[MemoryBlobRef]
    entities: list[MemoryEntity]
    relations: list[MemoryRelation]


class MemoryProjection(TypedDict, total=False):
    """Backend-facing projection derived from a memory record."""

    embedding_text: str
    vector_metadata: dict[str, Any]
    kg_update: "ResearchKGUpdate"


class ResearchKGNode(TypedDict, total=False):
    """Distilled research knowledge node."""

    node_type: str
    canonical_id: str
    display_name: str
    aliases: list[str]
    properties: dict[str, Any]


class ResearchKGRelation(TypedDict, total=False):
    """Distilled research knowledge relation."""

    relation_type: str
    source_id: str
    target_id: str
    properties: dict[str, Any]


class ResearchKGUpdate(TypedDict, total=False):
    """Graph payload derived from canonical memory records."""

    record_id: str
    domain: str
    nodes: list[ResearchKGNode]
    relations: list[ResearchKGRelation]
    provenance: dict[str, Any]


class CanonicalizationResult(TypedDict, total=False):
    """Resolved canonical identity for a KG concept."""

    canonical_id: str
    display_name: str
    aliases: list[str]
    normalized_fields: dict[str, Any]
    strategy: str
    confidence: float
    matched_existing_id: str | None
    rationale: str


class MemorySearchHit(TypedDict, total=False):
    """Hydrated memory retrieval result."""

    record: MemoryRecord
    distance: float | None
    document: str
    vector_metadata: dict[str, Any]


class MemoryObjectSpec(TypedDict, total=False):
    """Profile-declared behavior for a typed memory object."""

    object_type: str
    kind: str
    schema_version: str
    reusable: bool
    fingerprint_fields: list[str]
    fingerprint_metadata_key: str
    status_metadata_key: str
    ready_statuses: list[str]
    vector_fields: list[str]
    required_blob_names: list[str]


class MemoryQuery(TypedDict, total=False):
    """Backend-agnostic memory retrieval request."""

    query: str
    domain: str
    kind: str
    object_type: str
    object_key: str
    object_role: str
    tags: list[str]
    filters: dict[str, Any]
    fingerprint: str
    fingerprint_metadata_key: str
    include_blobs: bool
    n_results: int
    limit: int


class MemoryReuseResult(TypedDict, total=False):
    """Exact or policy-driven reuse lookup result."""

    reusable: bool
    record: MemoryRecord | None
    reason: str
    fingerprint: str
    resolved_blobs: dict[str, str]
