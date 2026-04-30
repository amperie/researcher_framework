"""Tests for the typed memory service APIs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from types import SimpleNamespace
from unittest.mock import patch

from core.memory import MemoryService, Neo4jMemoryGraphStore, fingerprint_for_spec, memory_object_specs
from core.memory.backends import get_memory_graph_store


@dataclass
class InMemoryDocumentStore:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def upsert(self, record: dict[str, Any]) -> None:
        self.records[record["record_id"]] = dict(record)

    def get(self, record_id: str) -> dict[str, Any] | None:
        return self.records.get(record_id)

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
        return [record for record in self.records.values() if _matches(record, filters)][:limit]


@dataclass
class InMemoryVectorStore:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    upsert_count: int = 0

    def upsert(self, record_id: str, document: str, metadata: dict[str, Any]) -> None:
        self.upsert_count += 1
        self.records[record_id] = {"id": record_id, "document": document, "metadata": metadata, "distance": 0.1}

    def query_similar(self, text: str, n_results: int) -> list[dict[str, Any]]:
        return list(self.records.values())[:n_results]

    def get_by_id(self, record_id: str) -> dict[str, Any] | None:
        return self.records.get(record_id)

    def delete(self, record_id: str) -> None:
        self.records.pop(record_id, None)


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def upsert(self, record: dict[str, Any], *, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        self.records.append({"record_id": record["record_id"], "nodes": nodes, "edges": edges})

    def query(self, *, node_type=None, node_key=None, edge_type=None, limit=50):
        return self.records[:limit]


PROFILE = {
    "name": "demo",
    "memory": {
        "objects": [
            {
                "object_type": "dataset",
                "kind": "demo_dataset",
                "reusable": True,
                "fingerprint_fields": ["config.source", "config.limit"],
                "fingerprint_metadata_key": "dataset_config_fingerprint",
                "status_metadata_key": "dataset_status",
                "ready_statuses": ["ready"],
            }
        ]
    },
}


def test_memory_specs_and_emit_build_typed_record(tmp_path):
    service = MemoryService(
        document_store=InMemoryDocumentStore(),
        vector_store=InMemoryVectorStore(),
        graph_store=InMemoryGraphStore(),
    )
    spec = memory_object_specs(PROFILE)["dataset"]
    payload = {"config": {"source": "mongo", "limit": 10}, "name": "dataset-a"}

    record = service.emit(
        profile=PROFILE,
        object_type="dataset",
        payload=payload,
        node="prepare_experiment",
        spec=spec,
        metadata={"dataset_status": "ready"},
    )

    assert record["kind"] == "demo_dataset"
    assert record["validity"]["reusable"] is True
    assert record["lineage"]["node"] == "prepare_experiment"
    assert record["metadata"]["dataset_config_fingerprint"] == fingerprint_for_spec(spec, payload)
    assert service.document_store.get(record["record_id"]) == record


def test_query_supports_structured_and_semantic_lookup():
    document_store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    service = MemoryService(document_store=document_store, vector_store=vector_store)
    service.persist_record({
        "record_id": "dataset:1",
        "domain": "demo",
        "kind": "demo_dataset",
        "object_type": "dataset",
        "object_key": "1",
        "title": "Dataset",
        "summary": "A reusable dataset",
        "content": {},
        "metadata": {"dataset_status": "ready", "dataset_config_fingerprint": "abc"},
        "tags": ["demo", "dataset"],
    })

    structured = service.query(domain="demo", object_type="dataset", fingerprint="abc", fingerprint_metadata_key="dataset_config_fingerprint")
    semantic = service.query(query="reusable dataset", object_type="dataset", n_results=3)

    assert structured[0]["record"]["record_id"] == "dataset:1"
    assert semantic[0]["record"]["record_id"] == "dataset:1"


def test_find_reusable_checks_status_and_required_blobs(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
    service = MemoryService(document_store=InMemoryDocumentStore(), vector_store=InMemoryVectorStore())
    service.persist_record({
        "record_id": "dataset:abc",
        "domain": "demo",
        "kind": "demo_dataset",
        "object_type": "dataset",
        "object_key": "abc",
        "title": "Dataset",
        "summary": "ready",
        "content": {},
        "metadata": {"dataset_status": "ready", "dataset_config_fingerprint": "abc"},
        "validity": {"reusable": True, "status": "ready"},
        "blob_refs": [{"name": "dataset_artifact", "uri": str(dataset_path)}],
    })

    result = service.find_reusable(
        domain="demo",
        object_type="dataset",
        fingerprint="abc",
        fingerprint_metadata_key="dataset_config_fingerprint",
        status_metadata_key="dataset_status",
        ready_statuses=["ready"],
        required_blob_names=["dataset_artifact"],
    )

    assert result["reusable"] is True
    assert result["record"]["record_id"] == "dataset:abc"
    assert result["resolved_blobs"]["dataset_artifact"] == str(dataset_path)


def test_find_reusable_for_profile_uses_declared_spec(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
    service = MemoryService(document_store=InMemoryDocumentStore(), vector_store=InMemoryVectorStore())
    service.persist_record({
        "record_id": "dataset:abc",
        "domain": "demo",
        "kind": "demo_dataset",
        "object_type": "dataset",
        "object_key": "abc",
        "title": "Dataset",
        "summary": "ready",
        "content": {},
        "metadata": {"dataset_status": "ready", "dataset_config_fingerprint": "abc"},
        "blob_refs": [{"name": "dataset_artifact", "uri": str(dataset_path)}],
    })

    profile = {
        **PROFILE,
        "memory": {
            "objects": [
                {
                    **PROFILE["memory"]["objects"][0],
                    "required_blob_names": ["dataset_artifact"],
                }
            ]
        },
    }

    result = service.find_reusable_for_profile(profile, object_type="dataset", fingerprint="abc")

    assert result["reusable"] is True
    assert result["record"]["record_id"] == "dataset:abc"


def test_repair_projections_reindexes_canonical_records():
    document_store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    service = MemoryService(document_store=document_store, vector_store=vector_store)
    document_store.upsert({
        "record_id": "r1",
        "domain": "demo",
        "kind": "note",
        "object_type": "note",
        "object_key": "r1",
        "title": "Note",
        "summary": "hello",
        "content": {},
        "metadata": {},
    })

    repaired = service.repair_projections()

    assert repaired == 1
    assert vector_store.upsert_count == 1
    assert vector_store.get_by_id("r1") is not None


def test_neo4j_graph_store_upserts_projection_and_decodes_query_metadata():
    driver = FakeNeo4jDriver(query_rows=[
        {
            "item": {
                "node_type": "dataset",
                "node_key": "abc",
                "name": "Dataset ABC",
                "metadata_json": '{"domain": "demo", "rows": 2}',
                "record_ids": ["record-1"],
            }
        }
    ])
    store = Neo4jMemoryGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
        driver=driver,
    )

    store.upsert(
        {
            "record_id": "record-1",
            "domain": "demo",
            "kind": "demo_dataset",
            "object_type": "dataset",
            "object_key": "abc",
            "title": "Dataset ABC",
            "summary": "ready",
            "metadata": {"nested": {"ok": True}},
            "tags": ["demo"],
        },
        nodes=[{"node_type": "dataset", "node_key": "abc", "name": "Dataset ABC", "metadata": {"rows": 2}}],
        edges=[{
            "edge_type": "created_by",
            "source_type": "dataset",
            "source_key": "abc",
            "target_type": "proposal",
            "target_key": "proposal-1",
            "metadata": {"score": 0.5},
        }],
    )
    rows = store.query(node_type="dataset", node_key="abc")

    assert rows == [{
        "node_type": "dataset",
        "node_key": "abc",
        "name": "Dataset ABC",
        "record_ids": ["record-1"],
        "metadata": {"domain": "demo", "rows": 2},
    }]
    assert any("MERGE (record:MemoryRecord" in call["query"] for call in driver.write_calls)
    assert any("MEMORY_RELATION" in call["query"] for call in driver.write_calls)
    assert driver.session_databases == ["neo4j", "neo4j"]


def test_get_memory_graph_store_uses_profile_storage_database_override():
    profile = {
        "name": "demo",
        "storage": {
            "memory_graph_backend": "neo4j",
            "memory_neo4j_database": "profile_memory_db",
        },
    }
    cfg = SimpleNamespace(
        memory_graph_backend="neo4j",
        memory_neo4j_uri="neo4j://hp.lan:7687",
        memory_neo4j_username="neo4j",
        memory_neo4j_password="password123",
        memory_neo4j_database="global_memory_db",
    )

    with patch("core.memory.backends.get_config", return_value=cfg):
        with patch("core.memory.backends.Neo4jMemoryGraphStore") as graph_store_cls:
            graph_store_cls.return_value = "graph-store"
            store = get_memory_graph_store(profile)

    assert store == "graph-store"
    assert graph_store_cls.call_args.kwargs["database"] == "profile_memory_db"


class FakeNeo4jDriver:
    def __init__(self, query_rows: list[dict[str, Any]] | None = None) -> None:
        self.query_rows = query_rows or []
        self.write_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []
        self.session_databases: list[str | None] = []

    def session(self, *, database: str | None = None) -> "FakeNeo4jSession":
        self.session_databases.append(database)
        return FakeNeo4jSession(self)


class FakeNeo4jSession:
    def __init__(self, driver: FakeNeo4jDriver) -> None:
        self.driver = driver

    def __enter__(self) -> "FakeNeo4jSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute_write(self, fn, *args):
        return fn(FakeNeo4jTx(self.driver.write_calls, []), *args)

    def execute_read(self, fn, *args):
        return fn(FakeNeo4jTx(self.driver.read_calls, self.driver.query_rows), *args)


class FakeNeo4jTx:
    def __init__(self, calls: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.rows = rows

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "params": params})
        return self.rows


def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = _get(record, key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


def _get(record: dict[str, Any], dotted: str) -> Any:
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
