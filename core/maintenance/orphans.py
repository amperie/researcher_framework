"""Scan and delete backend artifacts that are orphaned from canonical Mongo memory records."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from core.artifacts.store import ArtifactBackend, ArtifactMetadataStore, get_artifact_store
from core.memory.backends import (
    MemoryDocumentStore,
    MemoryGraphStore,
    NoopMemoryGraphStore,
    MemoryVectorStore,
    get_memory_document_store,
    get_memory_graph_store,
    get_memory_vector_store,
)
from core.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrphanCleanupService:
    """Maintains derived storage by treating Mongo memory records as the master set."""

    profile_name: str
    profile: dict[str, Any]
    document_store: MemoryDocumentStore | None = None
    vector_store: MemoryVectorStore | None = None
    graph_store: MemoryGraphStore | None = None
    artifact_metadata_store: ArtifactMetadataStore | None = None
    artifact_backend: ArtifactBackend | None = None
    backend_errors: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.document_store is None:
            self.document_store = get_memory_document_store(self.profile)
        if self.vector_store is None:
            self.vector_store = get_memory_vector_store(self.profile)
        if self.graph_store is None:
            try:
                self.graph_store = get_memory_graph_store(self.profile)
            except Exception as exc:
                self.backend_errors.append({
                    "backend": "neo4j",
                    "id": self.profile_name,
                    "error": str(exc),
                })
                log.warning(
                    "maintenance.orphans | Falling back to noop graph store for profile=%r: %s",
                    self.profile_name,
                    exc,
                )
                self.graph_store = NoopMemoryGraphStore()
        if self.artifact_metadata_store is None or self.artifact_backend is None:
            artifact_store = get_artifact_store(self.profile)
            self.artifact_metadata_store = artifact_store.metadata_store
            self.artifact_backend = artifact_store.backend

    def scan(self) -> dict[str, Any]:
        memory_record_ids, referenced_artifact_ids = _memory_master_sets(self.document_store)
        chroma_ids = _list_vector_ids(self.vector_store, domain=self.profile_name)
        neo4j_ids = _list_graph_record_ids(self.graph_store, domain=self.profile_name)
        artifact_docs = _artifact_metadata_records(self.artifact_metadata_store)
        storage_keys = _list_artifact_storage_keys(self.artifact_backend)

        artifact_docs_by_id = {
            str(doc.get("artifact_id") or ""): doc
            for doc in artifact_docs
            if str(doc.get("artifact_id") or "")
        }
        artifact_keys_by_id = {
            artifact_id: str(doc.get("storage_key") or "")
            for artifact_id, doc in artifact_docs_by_id.items()
            if str(doc.get("storage_key") or "")
        }

        orphan_chroma_ids = sorted(chroma_ids - memory_record_ids)
        orphan_neo4j_ids = sorted(neo4j_ids - memory_record_ids)
        orphan_artifact_ids = sorted(set(artifact_docs_by_id) - referenced_artifact_ids)
        orphan_artifact_docs = [artifact_docs_by_id[artifact_id] for artifact_id in orphan_artifact_ids]
        orphan_storage_keys = sorted(
            {
                key
                for key in storage_keys
                if key not in set(artifact_keys_by_id.values())
            }.union(
                key
                for artifact_id, key in artifact_keys_by_id.items()
                if artifact_id in orphan_artifact_ids and key
            )
        )

        return {
            "profile_name": self.profile_name,
            "counts": {
                "memory_records": len(memory_record_ids),
                "referenced_artifacts": len(referenced_artifact_ids),
                "chroma_records": len(chroma_ids),
                "neo4j_records": len(neo4j_ids),
                "artifact_metadata_records": len(artifact_docs_by_id),
                "artifact_storage_objects": len(storage_keys),
                "orphan_chroma_records": len(orphan_chroma_ids),
                "orphan_neo4j_records": len(orphan_neo4j_ids),
                "orphan_artifact_metadata_records": len(orphan_artifact_docs),
                "orphan_artifact_storage_objects": len(orphan_storage_keys),
            },
            "orphans": {
                "chroma_record_ids": orphan_chroma_ids,
                "neo4j_record_ids": orphan_neo4j_ids,
                "artifact_metadata_records": orphan_artifact_docs,
                "artifact_storage_keys": orphan_storage_keys,
            },
            "errors": list(self.backend_errors),
        }

    def delete(self) -> dict[str, Any]:
        scan = self.scan()
        deleted = {
            "chroma_records": 0,
            "neo4j_records": 0,
            "artifact_metadata_records": 0,
            "artifact_storage_objects": 0,
        }
        errors: list[dict[str, str]] = []
        errors.extend(self.backend_errors)

        for record_id in scan["orphans"]["chroma_record_ids"]:
            try:
                self.vector_store.delete(record_id)
                deleted["chroma_records"] += 1
            except Exception as exc:
                errors.append({"backend": "chroma", "id": str(record_id), "error": str(exc)})

        for record_id in scan["orphans"]["neo4j_record_ids"]:
            try:
                self.graph_store.delete_record(record_id)
                deleted["neo4j_records"] += 1
            except Exception as exc:
                errors.append({"backend": "neo4j", "id": str(record_id), "error": str(exc)})

        try:
            self.graph_store.prune_orphan_entities()
        except Exception as exc:
            errors.append({"backend": "neo4j", "id": "prune_orphan_entities", "error": str(exc)})

        for artifact in scan["orphans"]["artifact_metadata_records"]:
            artifact_id = str(artifact.get("artifact_id") or "")
            if not artifact_id:
                continue
            try:
                self.artifact_metadata_store.delete(artifact_id)
                deleted["artifact_metadata_records"] += 1
            except Exception as exc:
                errors.append({"backend": "artifact_metadata", "id": artifact_id, "error": str(exc)})

        for key in scan["orphans"]["artifact_storage_keys"]:
            try:
                self.artifact_backend.delete(str(key))
                deleted["artifact_storage_objects"] += 1
            except Exception as exc:
                errors.append({"backend": "artifact_storage", "id": str(key), "error": str(exc)})

        return {
            **scan,
            "deleted": deleted,
            "errors": errors,
        }


def scan_profiles(profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = [OrphanCleanupService(profile_name=name, profile=profile).scan() for name, profile in profiles.items()]
    return _aggregate_results(results, mode="scan")


def delete_profile_orphans(profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = [OrphanCleanupService(profile_name=name, profile=profile).delete() for name, profile in profiles.items()]
    return _aggregate_results(results, mode="delete")


def _aggregate_results(results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    totals = {
        "memory_records": 0,
        "referenced_artifacts": 0,
        "chroma_records": 0,
        "neo4j_records": 0,
        "artifact_metadata_records": 0,
        "artifact_storage_objects": 0,
        "orphan_chroma_records": 0,
        "orphan_neo4j_records": 0,
        "orphan_artifact_metadata_records": 0,
        "orphan_artifact_storage_objects": 0,
    }
    deleted = {
        "chroma_records": 0,
        "neo4j_records": 0,
        "artifact_metadata_records": 0,
        "artifact_storage_objects": 0,
    }
    errors: list[dict[str, str]] = []

    for result in results:
        for key in totals:
            totals[key] += int((result.get("counts") or {}).get(key, 0))
        for key in deleted:
            deleted[key] += int((result.get("deleted") or {}).get(key, 0))
        errors.extend(result.get("errors") or [])

    payload = {
        "mode": mode,
        "profiles": results,
        "totals": totals,
        "errors": errors,
    }
    if mode == "delete":
        payload["deleted"] = deleted
    return payload


def _memory_master_sets(document_store: MemoryDocumentStore) -> tuple[set[str], set[str]]:
    record_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for doc in _iter_document_records(document_store):
        record_id = str(doc.get("record_id") or "")
        if record_id:
            record_ids.add(record_id)
        for ref in doc.get("blob_refs") or []:
            if not isinstance(ref, dict):
                continue
            artifact_id = str(ref.get("artifact_id") or "")
            if artifact_id:
                artifact_ids.add(artifact_id)
    return record_ids, artifact_ids


def _iter_document_records(document_store: MemoryDocumentStore) -> list[dict[str, Any]]:
    collection = getattr(document_store, "collection", None)
    if collection is not None:
        docs = list(collection.find({}, {"record_id": 1, "blob_refs": 1}))
        for doc in docs:
            doc.pop("_id", None)
        return docs
    find = getattr(document_store, "find", None)
    if callable(find):
        return list(find({}, limit=1_000_000))
    return []


def _list_vector_ids(vector_store: MemoryVectorStore, *, domain: str | None = None) -> set[str]:
    list_ids = getattr(vector_store, "list_ids", None)
    if callable(list_ids):
        try:
            values = list_ids(domain=domain)
        except TypeError:
            values = list_ids()
        return {str(item) for item in values if item}
    return set()


def _list_graph_record_ids(graph_store: MemoryGraphStore, *, domain: str | None = None) -> set[str]:
    list_record_ids = getattr(graph_store, "list_record_ids", None)
    if callable(list_record_ids):
        try:
            values = list_record_ids(domain=domain)
        except TypeError:
            values = list_record_ids()
        return {str(item) for item in values if item}
    return set()


def _artifact_metadata_records(metadata_store: ArtifactMetadataStore) -> list[dict[str, Any]]:
    collection = getattr(metadata_store, "collection", None)
    if collection is not None:
        docs = list(collection.find({}, {"artifact_id": 1, "storage_key": 1, "storage_backend": 1, "storage_bucket": 1, "file_name": 1, "uri": 1}))
        for doc in docs:
            doc.pop("_id", None)
        return docs
    find = getattr(metadata_store, "find", None)
    if callable(find):
        return list(find({}, limit=1_000_000))
    return []


def _list_artifact_storage_keys(artifact_backend: ArtifactBackend) -> set[str]:
    list_keys = getattr(artifact_backend, "list_keys", None)
    if callable(list_keys):
        return {str(item) for item in list_keys() if item}
    return set()
