"""Tests for the typed memory service APIs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from types import SimpleNamespace
from unittest.mock import patch

from core.memory import MemoryService, Neo4jMemoryGraphStore, build_research_kg_update, default_memory_projection, fingerprint_for_spec, memory_object_specs
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
        self.reset_calls: list[str | None] = []

    def upsert(self, record: dict[str, Any], *, kg_update: dict[str, Any]) -> None:
        self.records.append({"record_id": record["record_id"], "kg_update": kg_update})

    def query(self, *, node_type=None, node_key=None, edge_type=None, limit=50):
        return self.records[:limit]

    def list_candidates(self, *, domain: str, node_type: str, limit: int = 25):
        return []

    def prune_orphan_entities(self) -> None:
        return None

    def reset(self, *, domain: str | None = None) -> None:
        self.reset_calls.append(domain)
        self.records.clear()


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


def test_rebuild_graph_from_documents_resets_and_replays_records():
    document_store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    graph_store = InMemoryGraphStore()
    service = MemoryService(
        document_store=document_store,
        vector_store=vector_store,
        graph_store=graph_store,
        profile={
            "name": "demo",
            "evaluation": {"primary_metric": "test_auc", "thresholds": {"test_auc": 0.65}},
            "knowledge_graph": {"canonicalization": {"llm_enabled": False}},
        },
    )
    document_store.upsert({
        "record_id": "exp-1",
        "domain": "demo",
        "kind": "demo_experiment",
        "object_type": "experiment_result",
        "object_key": "exp-1",
        "title": "activation_sparsity",
        "summary": "Sparse activations reached AUC 0.91 on DemoBench.",
        "content": {
            "proposal_name": "activation_sparsity",
            "proposal": {"dataset": "DemoBench", "description": "Sparse activations improve AUC"},
            "metrics": {"test_auc": 0.91},
            "research_direction": "find useful probes",
        },
        "metadata": {
            "dataset": "DemoBench",
            "feature_set_class_name": "ActivationSparsity",
            "assessment": "strong",
        },
        "tags": ["demo"],
    })

    rebuilt = service.rebuild_graph_from_documents({"domain": "demo"})

    assert rebuilt == 1
    assert graph_store.reset_calls == ["demo"]
    assert len(graph_store.records) == 1
    assert any(node["node_type"] == "Evidence" for node in graph_store.records[0]["kg_update"]["nodes"])


def test_default_memory_projection_flattens_list_metadata_for_vectors():
    projection = default_memory_projection({
        "record_id": "r1",
        "domain": "demo",
        "kind": "demo_note",
        "object_type": "note",
        "object_key": "r1",
        "title": "Note",
        "summary": "hello",
        "content": {},
        "metadata": {
            "lessons": ["first", "second"],
            "feature_importance_keys": ["a", "b"],
        },
        "tags": ["demo", "note"],
    })

    assert projection["vector_metadata"]["tags"] == "demo|note"
    assert projection["vector_metadata"]["lessons"] == "first | second"
    assert projection["vector_metadata"]["feature_importance_keys"] == "a | b"
    assert projection["kg_update"]["nodes"] == []
    assert projection["kg_update"]["relations"] == []


def test_build_research_kg_update_distills_experiment_record():
    profile = {
        "name": "demo",
        "evaluation": {
            "primary_metric": "test_auc",
            "thresholds": {"test_auc": 0.65},
        },
        "knowledge_graph": {
            "canonicalization": {"llm_enabled": False},
            "metric_bands": [
                {
                    "metric_name": "test_auc",
                    "operator": ">=",
                    "threshold": 0.9,
                    "display_name": "test_auc >= 0.90",
                    "band_key": "test_auc_gte_0_90",
                }
            ],
        },
    }

    update = build_research_kg_update(
        {
            "record_id": "exp-1",
            "domain": "demo",
            "kind": "demo_experiment",
            "object_type": "experiment_result",
            "title": "activation_sparsity",
            "summary": "Sparse activations reached AUC 0.91 on DemoBench.",
            "content": {
                "proposal_name": "activation_sparsity",
                "proposal": {"dataset": "DemoBench", "description": "Sparse activations improve AUC"},
                "metrics": {"test_auc": 0.91},
                "research_direction": "find useful probes",
            },
            "metadata": {
                "dataset": "DemoBench",
                "feature_set_class_name": "ActivationSparsity",
                "assessment": "strong",
            },
        },
        profile=profile,
    )

    node_types = {node["node_type"] for node in update["nodes"]}
    relation_types = {relation["relation_type"] for relation in update["relations"]}

    assert {"Question", "Hypothesis", "Method", "Evidence", "Dataset", "Metric", "Finding", "PerformanceBand"} <= node_types
    assert {"HAS_HYPOTHESIS", "TESTED_BY", "SUPPORTED_BY", "USES_METHOD", "ON_DATASET", "MEASURED_BY", "PRODUCED_FINDING", "IN_PERFORMANCE_BAND"} <= relation_types


def test_neo4j_graph_store_upserts_projection_and_decodes_query_metadata():
    driver = FakeNeo4jDriver(query_rows=[
        {
            "item": {
                "node_type": "Dataset",
                "canonical_id": "demo:dataset:abc",
                "display_name": "Dataset ABC",
                "aliases": ["Dataset ABC"],
                "metadata_json": '{"rows": 2}',
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
        kg_update={
            "record_id": "record-1",
            "domain": "demo",
            "nodes": [{
                "node_type": "Dataset",
                "canonical_id": "demo:dataset:abc",
                "display_name": "Dataset ABC",
                "aliases": ["Dataset ABC"],
                "properties": {"rows": 2},
            }],
            "relations": [{
                "relation_type": "ON_DATASET",
                "source_id": "demo:evidence:e1",
                "target_id": "demo:dataset:abc",
                "properties": {"score": 0.5},
            }],
        },
    )
    rows = store.query(node_type="Dataset", node_key="demo:dataset:abc")

    assert rows == [{
        "node_type": "Dataset",
        "canonical_id": "demo:dataset:abc",
        "display_name": "Dataset ABC",
        "aliases": ["Dataset ABC"],
        "record_ids": ["record-1"],
        "metadata": {"rows": 2},
    }]
    assert any("MERGE (record:ResearchKGRecord" in call["query"] for call in driver.write_calls)
    assert any("KG_RELATION" in call["query"] for call in driver.write_calls)
    assert driver.session_databases == ["neo4j", "neo4j"]


def test_neo4j_graph_store_upsert_accepts_relation_metadata_domain_without_duplicate_kwarg():
    driver = FakeNeo4jDriver()
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
            "metadata": {},
            "tags": ["demo"],
        },
        kg_update={
            "record_id": "record-1",
            "domain": "demo",
            "nodes": [],
            "relations": [{
                "relation_type": "ON_DATASET",
                "source_id": "edge-domain:evidence:e1",
                "target_id": "edge-domain:dataset:abc",
                "properties": {"domain": "edge-domain", "score": 0.5},
            }],
        },
    )

    relation_calls = [call for call in driver.write_calls if "MERGE (source)-[rel:KG_RELATION" in call["query"]]
    assert len(relation_calls) == 1
    assert relation_calls[0]["params"]["domain"] == "edge-domain"


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
