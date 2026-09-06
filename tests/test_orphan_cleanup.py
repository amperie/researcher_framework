"""Tests for orphan cleanup across Mongo-backed memory and derived storage."""
from __future__ import annotations

from dataclasses import dataclass, field

import mongomock

from core.artifacts.store import MongoArtifactMetadataStore
from core.maintenance.orphans import OrphanCleanupService
from core.memory.backends import MongoMemoryDocumentStore


@dataclass
class FakeArtifactBackend:
    keys: set[str] = field(default_factory=set)
    deleted: list[str] = field(default_factory=list)

    def list_keys(self) -> list[str]:
        return sorted(self.keys)

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.keys.discard(key)


@dataclass
class FakeVectorStore:
    ids: set[str] = field(default_factory=set)
    deleted: list[str] = field(default_factory=list)

    def list_ids(self) -> list[str]:
        return sorted(self.ids)

    def delete(self, record_id: str) -> None:
        self.deleted.append(record_id)
        self.ids.discard(record_id)


@dataclass
class FakeGraphStore:
    ids: set[str] = field(default_factory=set)
    deleted: list[str] = field(default_factory=list)
    pruned: bool = False

    def list_record_ids(self) -> list[str]:
        return sorted(self.ids)

    def delete_record(self, record_id: str) -> None:
        self.deleted.append(record_id)
        self.ids.discard(record_id)

    def prune_orphan_entities(self) -> None:
        self.pruned = True


def test_orphan_cleanup_scan_and_delete_derived_storage():
    mongo_client = mongomock.MongoClient()
    document_store = MongoMemoryDocumentStore(
        mongo_url="mongodb://localhost:27017",
        db_name="researcher",
        collection_name="memory_records",
        client=mongo_client,
    )
    document_store.upsert({
        "record_id": "record-1",
        "blob_refs": [{"artifact_id": "artifact-1", "name": "dataset"}],
    })

    metadata_store = MongoArtifactMetadataStore(
        mongo_url="mongodb://localhost:27017",
        db_name="researcher",
        collection_name="artifacts",
        client=mongo_client,
    )
    metadata_store.put({
        "artifact_id": "artifact-1",
        "storage_key": "trading_researcher/dataset/artifact-1/file.csv",
        "storage_backend": "s3",
        "storage_bucket": "artifacts",
        "file_name": "file.csv",
        "uri": "http://minio/artifacts/trading_researcher/dataset/artifact-1/file.csv",
    })
    metadata_store.put({
        "artifact_id": "artifact-2",
        "storage_key": "trading_researcher/dataset/artifact-2/file.csv",
        "storage_backend": "s3",
        "storage_bucket": "artifacts",
        "file_name": "file.csv",
        "uri": "http://minio/artifacts/trading_researcher/dataset/artifact-2/file.csv",
    })

    service = OrphanCleanupService(
        profile_name="trading_researcher",
        profile={"name": "trading_researcher"},
        document_store=document_store,
        vector_store=FakeVectorStore(ids={"record-1", "record-2"}),
        graph_store=FakeGraphStore(ids={"record-1", "record-3"}),
        artifact_metadata_store=metadata_store,
        artifact_backend=FakeArtifactBackend(
            keys={
                "trading_researcher/dataset/artifact-1/file.csv",
                "trading_researcher/dataset/artifact-2/file.csv",
                "trading_researcher/dataset/dangling/file.csv",
            }
        ),
    )

    scan = service.scan()

    assert scan["counts"]["orphan_chroma_records"] == 1
    assert scan["counts"]["orphan_neo4j_records"] == 1
    assert scan["counts"]["orphan_artifact_metadata_records"] == 1
    assert scan["counts"]["orphan_artifact_storage_objects"] == 1
    assert scan["counts"]["untracked_artifact_storage_objects"] == 1
    assert scan["orphans"]["chroma_record_ids"] == ["record-2"]
    assert scan["orphans"]["neo4j_record_ids"] == ["record-3"]
    assert [item["artifact_id"] for item in scan["orphans"]["artifact_metadata_records"]] == ["artifact-2"]
    assert scan["orphans"]["artifact_storage_keys"] == [
        "trading_researcher/dataset/artifact-2/file.csv",
    ]
    assert scan["untracked"]["artifact_storage_keys"] == [
        "trading_researcher/dataset/dangling/file.csv",
    ]

    deleted = service.delete()

    assert deleted["deleted"] == {
        "chroma_records": 1,
        "neo4j_records": 1,
        "artifact_metadata_records": 1,
        "artifact_storage_objects": 1,
    }
    assert deleted["errors"] == []
    assert service.vector_store.deleted == ["record-2"]
    assert service.graph_store.deleted == ["record-3"]
    assert service.graph_store.pruned is True
    assert service.artifact_backend.deleted == [
        "trading_researcher/dataset/artifact-2/file.csv",
    ]
    assert metadata_store.get("artifact-1") is not None
    assert metadata_store.get("artifact-2") is None
