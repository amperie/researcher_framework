"""Storage backend abstractions for canonical memory records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pymongo

from configs.config import get_config
from core.memory.models import MemoryRecord
from core.tools.chroma_tool import ChromaStore


class MemoryDocumentStore(Protocol):
    """Store for the full canonical memory record."""

    def upsert(self, record: MemoryRecord) -> None: ...

    def get(self, record_id: str) -> MemoryRecord | None: ...

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[MemoryRecord]: ...


class MemoryVectorStore(Protocol):
    """Store for semantic retrieval projections."""

    def upsert(self, record_id: str, document: str, metadata: dict[str, Any]) -> None: ...

    def query_similar(self, text: str, n_results: int) -> list[dict[str, Any]]: ...

    def get_by_id(self, record_id: str) -> dict[str, Any] | None: ...


class MemoryGraphStore(Protocol):
    """Store for graph projections."""

    def upsert(
        self,
        record: MemoryRecord,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None: ...


@dataclass
class MongoMemoryDocumentStore:
    """Mongo-backed source of truth for canonical memory records."""

    mongo_url: str
    db_name: str
    collection_name: str = "memory_records"
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = pymongo.MongoClient(self.mongo_url)

    @property
    def collection(self) -> Any:
        return self.client[self.db_name][self.collection_name]

    def upsert(self, record: MemoryRecord) -> None:
        doc = dict(record)
        self.collection.replace_one({"record_id": doc["record_id"]}, doc, upsert=True)

    def get(self, record_id: str) -> MemoryRecord | None:
        doc = self.collection.find_one({"record_id": record_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return doc

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[MemoryRecord]:
        docs = list(self.collection.find(filters).limit(limit))
        for doc in docs:
            doc.pop("_id", None)
        return docs


@dataclass
class ChromaMemoryVectorStore:
    """Chroma-backed semantic retrieval projection store."""

    collection_name: str | None = None
    store: ChromaStore | None = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = ChromaStore(collection_name=self.collection_name)

    def upsert(self, record_id: str, document: str, metadata: dict[str, Any]) -> None:
        self.store.upsert(record_id, document, metadata)

    def query_similar(self, text: str, n_results: int) -> list[dict[str, Any]]:
        return self.store.query_similar(text, n_results)

    def get_by_id(self, record_id: str) -> dict[str, Any] | None:
        return self.store.get_by_id(record_id)


class NoopMemoryGraphStore:
    """Placeholder graph store until a concrete backend is configured."""

    def upsert(
        self,
        record: MemoryRecord,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        return None


def get_memory_document_store(profile: dict[str, Any]) -> MongoMemoryDocumentStore:
    """Build the configured document store for memory records."""
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    db_name = (
        storage_cfg.get("memory_mongodb_db")
        or storage_cfg.get("mongodb_results_db")
        or "researcher_results"
    )
    collection_name = storage_cfg.get("memory_mongodb_collection", "memory_records")
    return MongoMemoryDocumentStore(
        mongo_url=cfg.mongo_url,
        db_name=db_name,
        collection_name=collection_name,
    )


def get_memory_vector_store(profile: dict[str, Any]) -> ChromaMemoryVectorStore:
    """Build the configured vector store for memory records."""
    storage_cfg = profile.get("storage") or {}
    collection_name = storage_cfg.get("memory_chroma_collection") or storage_cfg.get("chroma_collection")
    return ChromaMemoryVectorStore(collection_name=collection_name)


def get_memory_graph_store(profile: dict[str, Any]) -> MemoryGraphStore:
    """Build the graph projection store for memory records."""
    return NoopMemoryGraphStore()
