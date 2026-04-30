"""Storage backend abstractions for canonical memory records."""
from __future__ import annotations

from dataclasses import dataclass
import json
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

    def delete(self, record_id: str) -> None: ...


class MemoryGraphStore(Protocol):
    """Store for graph projections."""

    def upsert(
        self,
        record: MemoryRecord,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self,
        *,
        node_type: str | None = None,
        node_key: str | None = None,
        edge_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...


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

    def delete(self, record_id: str) -> None:
        delete = getattr(self.store, "delete", None)
        if callable(delete):
            delete(record_id)


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

    def query(
        self,
        *,
        node_type: str | None = None,
        node_key: str | None = None,
        edge_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return []


@dataclass
class Neo4jMemoryGraphStore:
    """Neo4j-backed graph projection store for canonical memory records."""

    uri: str
    username: str
    password: str
    database: str | None = None
    driver: Any | None = None

    def __post_init__(self) -> None:
        if self.driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
                raise RuntimeError(
                    "Neo4j graph memory requires the 'neo4j' package. "
                    "Install project dependencies or add neo4j>=5.0."
                ) from exc
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def upsert(
        self,
        record: MemoryRecord,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            return None
        payload = {
            "record_id": record_id,
            "domain": str(record.get("domain") or ""),
            "kind": str(record.get("kind") or ""),
            "object_type": str(record.get("object_type") or ""),
            "object_key": str(record.get("object_key") or ""),
            "object_role": str(record.get("object_role") or ""),
            "title": str(record.get("title") or ""),
            "summary": str(record.get("summary") or ""),
            "created_at": str(record.get("created_at") or ""),
            "metadata_json": _json_safe_json(record.get("metadata") or {}),
            "tags": [str(item) for item in (record.get("tags") or [])],
        }
        clean_nodes = []
        for node in nodes:
            payload_node = _graph_node_payload(node)
            if payload_node is not None:
                clean_nodes.append(payload_node)
        clean_edges = []
        for edge in edges:
            payload_edge = _graph_edge_payload(edge)
            if payload_edge is not None:
                clean_edges.append(payload_edge)

        with self.driver.session(database=self.database) as session:
            session.execute_write(self._upsert_projection, payload, clean_nodes, clean_edges)

    def query(
        self,
        *,
        node_type: str | None = None,
        node_key: str | None = None,
        edge_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params = {
            "node_type": node_type,
            "node_key": node_key,
            "edge_type": edge_type,
            "limit": int(limit),
        }
        with self.driver.session(database=self.database) as session:
            if edge_type:
                result = session.execute_read(self._query_edges, params)
            else:
                result = session.execute_read(self._query_nodes, params)
        return [_decode_graph_item(dict(item)) for item in result]

    @staticmethod
    def _upsert_projection(tx: Any, record_payload: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        record_id = record_payload["record_id"]
        tx.run(
            """
            MERGE (record:MemoryRecord {record_id: $record_id})
            SET record += $record
            WITH record
            OPTIONAL MATCH (record)-[projection:PROJECTS_ENTITY]->()
            DELETE projection
            WITH record
            OPTIONAL MATCH ()-[relation:MEMORY_RELATION {record_id: $record_id}]->()
            DELETE relation
            """,
            record_id=record_id,
            record=record_payload,
        )
        for node in nodes:
            tx.run(
                """
                MERGE (entity:MemoryEntity {node_type: $node_type, node_key: $node_key})
                SET entity.name = $name,
                    entity.metadata_json = $metadata_json
                WITH entity
                MATCH (record:MemoryRecord {record_id: $record_id})
                MERGE (record)-[:PROJECTS_ENTITY]->(entity)
                """,
                record_id=record_id,
                **node,
            )
        for edge in edges:
            tx.run(
                """
                MERGE (source:MemoryEntity {node_type: $source_type, node_key: $source_key})
                ON CREATE SET source.name = $source_key, source.metadata_json = '{}'
                MERGE (target:MemoryEntity {node_type: $target_type, node_key: $target_key})
                ON CREATE SET target.name = $target_key, target.metadata_json = '{}'
                MERGE (source)-[relation:MEMORY_RELATION {
                    record_id: $record_id,
                    edge_type: $edge_type,
                    source_type: $source_type,
                    source_key: $source_key,
                    target_type: $target_type,
                    target_key: $target_key
                }]->(target)
                SET relation.metadata_json = $metadata_json
                """,
                record_id=record_id,
                **edge,
            )

    @staticmethod
    def _query_nodes(tx: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = tx.run(
            """
            MATCH (node:MemoryEntity)
            WHERE ($node_type IS NULL OR node.node_type = $node_type)
              AND ($node_key IS NULL OR node.node_key = $node_key)
            OPTIONAL MATCH (record:MemoryRecord)-[:PROJECTS_ENTITY]->(node)
            RETURN {
                node_type: node.node_type,
                node_key: node.node_key,
                name: node.name,
                metadata_json: node.metadata_json,
                record_ids: collect(DISTINCT record.record_id)
            } AS item
            LIMIT $limit
            """,
            **params,
        )
        return [row["item"] for row in result]

    @staticmethod
    def _query_edges(tx: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = tx.run(
            """
            MATCH (source:MemoryEntity)-[relation:MEMORY_RELATION]->(target:MemoryEntity)
            WHERE ($edge_type IS NULL OR relation.edge_type = $edge_type)
              AND ($node_type IS NULL OR source.node_type = $node_type OR target.node_type = $node_type)
              AND ($node_key IS NULL OR source.node_key = $node_key OR target.node_key = $node_key)
            RETURN {
                edge_type: relation.edge_type,
                source_type: source.node_type,
                source_key: source.node_key,
                target_type: target.node_type,
                target_key: target.node_key,
                metadata_json: relation.metadata_json,
                record_id: relation.record_id
            } AS item
            LIMIT $limit
            """,
            **params,
        )
        return [row["item"] for row in result]


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
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    backend = str(storage_cfg.get("memory_graph_backend") or getattr(cfg, "memory_graph_backend", "noop") or "noop").lower()
    if backend in {"neo4j", "neo4j_memory"}:
        uri = storage_cfg.get("memory_neo4j_uri") or getattr(cfg, "memory_neo4j_uri", "")
        username = storage_cfg.get("memory_neo4j_username") or getattr(cfg, "memory_neo4j_username", "")
        password = storage_cfg.get("memory_neo4j_password") or getattr(cfg, "memory_neo4j_password", "")
        database = storage_cfg.get("memory_neo4j_database") or getattr(cfg, "memory_neo4j_database", None)
        if not uri or not username:
            raise ValueError("Neo4j memory graph backend requires memory_neo4j_uri and memory_neo4j_username.")
        return Neo4jMemoryGraphStore(
            uri=str(uri),
            username=str(username),
            password=str(password or ""),
            database=str(database) if database else None,
        )
    return NoopMemoryGraphStore()


def _graph_node_payload(node: dict[str, Any]) -> dict[str, Any] | None:
    node_type = str(node.get("node_type") or "")
    node_key = str(node.get("node_key") or "")
    if not node_type or not node_key:
        return None
    return {
        "node_type": node_type,
        "node_key": node_key,
        "name": str(node.get("name") or node_key),
        "metadata_json": _json_safe_json(node.get("metadata") or {}),
    }


def _graph_edge_payload(edge: dict[str, Any]) -> dict[str, Any] | None:
    edge_type = str(edge.get("edge_type") or "")
    source_type = str(edge.get("source_type") or "")
    source_key = str(edge.get("source_key") or "")
    target_type = str(edge.get("target_type") or "")
    target_key = str(edge.get("target_key") or "")
    if not all((edge_type, source_type, source_key, target_type, target_key)):
        return None
    return {
        "edge_type": edge_type,
        "source_type": source_type,
        "source_key": source_key,
        "target_type": target_type,
        "target_key": target_key,
        "metadata_json": _json_safe_json(edge.get("metadata") or {}),
    }


def _json_safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _decode_graph_item(item: dict[str, Any]) -> dict[str, Any]:
    metadata_json = item.pop("metadata_json", "")
    try:
        item["metadata"] = json.loads(metadata_json) if metadata_json else {}
    except (TypeError, json.JSONDecodeError):
        item["metadata"] = {}
    return item
