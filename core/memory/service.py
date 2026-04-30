"""Orchestration layer for canonical memory persistence and retrieval."""
from __future__ import annotations

from typing import Any

from core.memory.backends import (
    MemoryDocumentStore,
    MemoryGraphStore,
    MemoryVectorStore,
    get_memory_document_store,
    get_memory_graph_store,
    get_memory_vector_store,
)
from core.memory.defaults import default_memory_projection, record_from_vector_hit
from core.memory.models import MemoryProjection, MemoryRecord, MemorySearchHit


class MemoryService:
    """Coordinates document, vector, and graph memory backends."""

    def __init__(
        self,
        *,
        document_store: MemoryDocumentStore,
        vector_store: MemoryVectorStore,
        graph_store: MemoryGraphStore | None = None,
    ) -> None:
        self.document_store = document_store
        self.vector_store = vector_store
        self.graph_store = graph_store

    @classmethod
    def for_profile(cls, profile: dict[str, Any]) -> "MemoryService":
        return cls(
            document_store=get_memory_document_store(profile),
            vector_store=get_memory_vector_store(profile),
            graph_store=get_memory_graph_store(profile),
        )

    def persist_record(
        self,
        record: MemoryRecord,
        *,
        projection: MemoryProjection | None = None,
    ) -> None:
        plan = projection or default_memory_projection(record)
        self.document_store.upsert(record)
        self.vector_store.upsert(
            str(record.get("record_id") or ""),
            str(plan.get("embedding_text") or ""),
            dict(plan.get("vector_metadata") or {}),
        )
        if self.graph_store is not None:
            self.graph_store.upsert(
                record,
                nodes=list(plan.get("graph_nodes") or []),
                edges=list(plan.get("graph_edges") or []),
            )

    def persist_records(
        self,
        records: list[MemoryRecord],
        *,
        projections: dict[str, MemoryProjection] | None = None,
    ) -> None:
        for record in records:
            record_id = str(record.get("record_id") or "")
            self.persist_record(record, projection=(projections or {}).get(record_id))

    def search(self, query: str, *, n_results: int = 5) -> list[MemorySearchHit]:
        hits = self.vector_store.query_similar(query, n_results=n_results)
        output: list[MemorySearchHit] = []
        for hit in hits:
            record_id = str(hit.get("id") or "")
            try:
                record = self.document_store.get(record_id) if record_id else None
            except Exception:
                record = None
            if record is None:
                record = record_from_vector_hit(hit)
            output.append({
                "record": record,
                "distance": hit.get("distance"),
                "document": str(hit.get("document") or ""),
                "vector_metadata": dict(hit.get("metadata") or {}),
            })
        return output

    def find_records(self, filters: dict[str, Any], *, limit: int = 50) -> list[MemoryRecord]:
        return self.document_store.find(filters, limit=limit)

    def find_one_record(self, filters: dict[str, Any]) -> MemoryRecord | None:
        records = self.find_records(filters, limit=1)
        return records[0] if records else None
