"""Storage backend abstractions for canonical memory records."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

import pymongo

from configs.config import get_config
from core.memory.models import MemoryRecord
from core.tools.chroma_tool import ChromaStore
from core.utils.logger import get_logger

log = get_logger(__name__)


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

    def list_ids(self, *, domain: str | None = None) -> list[str]: ...


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

    def list_record_ids(self, *, domain: str | None = None) -> list[str]: ...

    def delete_record(self, record_id: str) -> None: ...

    def prune_orphan_entities(self) -> None: ...


@dataclass
class MongoMemoryDocumentStore:
    """Mongo-backed source of truth for canonical memory records."""

    mongo_url: str
    db_name: str
    collection_name: str = "memory_records"
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            log.debug("memory.backends | Creating Mongo memory client db=%r collection=%r", self.db_name, self.collection_name)
            self.client = pymongo.MongoClient(self.mongo_url)

    @property
    def collection(self) -> Any:
        return self.client[self.db_name][self.collection_name]

    def upsert(self, record: MemoryRecord) -> None:
        doc = dict(record)
        log.debug(
            "memory.backends | Mongo upsert memory record id=%r db=%r collection=%r",
            doc.get("record_id"),
            self.db_name,
            self.collection_name,
        )
        self.collection.replace_one({"record_id": doc["record_id"]}, doc, upsert=True)

    def get(self, record_id: str) -> MemoryRecord | None:
        log.debug("memory.backends | Mongo get memory record id=%r", record_id)
        doc = self.collection.find_one({"record_id": record_id})
        if not doc:
            log.debug("memory.backends | Mongo memory record not found id=%r", record_id)
            return None
        doc.pop("_id", None)
        return doc

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[MemoryRecord]:
        log.debug("memory.backends | Mongo find filters=%s limit=%d", filters, limit)
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
            log.debug("memory.backends | Creating Chroma memory vector store collection=%r", self.collection_name)
            self.store = ChromaStore(collection_name=self.collection_name)

    def upsert(self, record_id: str, document: str, metadata: dict[str, Any]) -> None:
        log.debug(
            "memory.backends | Chroma upsert memory vector id=%r collection=%r document_chars=%d",
            record_id,
            self.collection_name,
            len(document),
        )
        self.store.upsert(record_id, document, metadata)

    def query_similar(self, text: str, n_results: int) -> list[dict[str, Any]]:
        log.debug("memory.backends | Chroma query n_results=%d query_chars=%d", n_results, len(text))
        results = self.store.query_similar(text, n_results)
        log.debug("memory.backends | Chroma query returned %d hit(s)", len(results))
        return results

    def get_by_id(self, record_id: str) -> dict[str, Any] | None:
        log.debug("memory.backends | Chroma get vector id=%r", record_id)
        return self.store.get_by_id(record_id)

    def delete(self, record_id: str) -> None:
        delete = getattr(self.store, "delete", None)
        if callable(delete):
            log.debug("memory.backends | Chroma delete vector id=%r", record_id)
            delete(record_id)
        else:
            log.debug("memory.backends | Chroma delete skipped; underlying store has no delete method")

    def list_ids(self, *, domain: str | None = None) -> list[str]:
        list_ids = getattr(self.store, "list_ids", None)
        if callable(list_ids):
            return [str(item) for item in list_ids(domain=domain) if item]
        log.debug("memory.backends | Chroma list_ids skipped; underlying store has no list_ids method")
        return []


class NoopMemoryGraphStore:
    """Placeholder graph store until a concrete backend is configured."""

    def upsert(
        self,
        record: MemoryRecord,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        log.debug("memory.backends | Noop graph upsert ignored id=%r", record.get("record_id"))
        return None

    def query(
        self,
        *,
        node_type: str | None = None,
        node_key: str | None = None,
        edge_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        log.debug("memory.backends | Noop graph query returned no results")
        return []

    def list_record_ids(self, *, domain: str | None = None) -> list[str]:
        return []

    def delete_record(self, record_id: str) -> None:
        _ = record_id
        return None

    def prune_orphan_entities(self) -> None:
        return None


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
                log.error("memory.backends | Neo4j graph backend requested but neo4j package is missing")
                raise RuntimeError(
                    "Neo4j graph memory requires the 'neo4j' package. "
                    "Install project dependencies or add neo4j>=5.0."
                ) from exc
            log.debug("memory.backends | Creating Neo4j memory graph driver uri=%r database=%r", self.uri, self.database)
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
            log.warning("memory.backends | Skipping Neo4j graph upsert for record without id")
            return None
        payload = {
            "record_id": record_id,
            "domain": str(record.get("domain") or ""),
            "profile": str(record.get("domain") or ""),
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
        log.debug(
            "memory.backends | Neo4j graph upserted id=%r database=%r nodes=%d edges=%d",
            record_id,
            self.database,
            len(clean_nodes),
            len(clean_edges),
        )

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
        decoded = [_decode_graph_item(dict(item)) for item in result]
        log.debug("memory.backends | Neo4j graph query returned %d result(s)", len(decoded))
        return decoded

    def list_record_ids(self, *, domain: str | None = None) -> list[str]:
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(self._list_record_ids, str(domain) if domain else None)
        return [str(item) for item in rows if item]

    def delete_record(self, record_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._delete_record_projection, str(record_id))
        log.debug("memory.backends | Neo4j graph deleted record projection id=%r", record_id)

    def prune_orphan_entities(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._prune_orphan_entities)
        log.debug("memory.backends | Neo4j graph pruned orphan entities")

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
                    entity.raw_key = $raw_key,
                    entity.domain = $domain,
                    entity.profile = $profile,
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
                SET source.raw_key = $source_raw_key,
                    source.domain = $domain,
                    source.profile = $profile
                MERGE (target:MemoryEntity {node_type: $target_type, node_key: $target_key})
                ON CREATE SET target.name = $target_key, target.metadata_json = '{}'
                SET target.raw_key = $target_raw_key,
                    target.domain = $domain,
                    target.profile = $profile
                MERGE (source)-[relation:MEMORY_RELATION {
                    record_id: $record_id,
                    edge_type: $edge_type,
                    source_type: $source_type,
                    source_key: $source_key,
                    target_type: $target_type,
                    target_key: $target_key
                }]->(target)
                SET relation.metadata_json = $metadata_json,
                    relation.domain = $domain,
                    relation.profile = $profile,
                    relation.source_raw_key = $source_raw_key,
                    relation.target_raw_key = $target_raw_key
                """,
                domain=record_payload.get("domain", ""),
                profile=record_payload.get("profile", ""),
                record_id=record_id,
                **edge,
            )

    @staticmethod
    def _query_nodes(tx: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = tx.run(
            """
            MATCH (node:MemoryEntity)
            WHERE ($node_type IS NULL OR node.node_type = $node_type)
              AND ($node_key IS NULL OR node.node_key = $node_key OR node.raw_key = $node_key)
            OPTIONAL MATCH (record:MemoryRecord)-[:PROJECTS_ENTITY]->(node)
            RETURN {
                node_type: node.node_type,
                node_key: node.node_key,
                raw_key: node.raw_key,
                domain: node.domain,
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
              AND ($node_key IS NULL
                OR source.node_key = $node_key
                OR target.node_key = $node_key
                OR source.raw_key = $node_key
                OR target.raw_key = $node_key)
            RETURN {
                edge_type: relation.edge_type,
                source_type: source.node_type,
                source_key: source.node_key,
                source_raw_key: source.raw_key,
                target_type: target.node_type,
                target_key: target.node_key,
                target_raw_key: target.raw_key,
                domain: relation.domain,
                metadata_json: relation.metadata_json,
                record_id: relation.record_id
            } AS item
            LIMIT $limit
            """,
            **params,
        )
        return [row["item"] for row in result]

    @staticmethod
    def _list_record_ids(tx: Any, domain: str | None = None) -> list[str]:
        result = tx.run(
            """
            MATCH (record:MemoryRecord)
            WHERE ($domain IS NULL OR record.domain = $domain OR record.profile = $domain)
            RETURN record.record_id AS record_id
            """,
            domain=domain,
        )
        return [row.get("record_id") for row in result]

    @staticmethod
    def _delete_record_projection(tx: Any, record_id: str) -> None:
        tx.run(
            """
            MATCH (record:MemoryRecord {record_id: $record_id})
            OPTIONAL MATCH (record)-[projection:PROJECTS_ENTITY]->()
            DELETE projection
            WITH record
            OPTIONAL MATCH ()-[relation:MEMORY_RELATION {record_id: $record_id}]->()
            DELETE relation
            WITH record
            DETACH DELETE record
            """,
            record_id=record_id,
        )

    @staticmethod
    def _prune_orphan_entities(tx: Any) -> None:
        tx.run(
            """
            MATCH (entity:MemoryEntity)
            WHERE NOT EXISTS { MATCH (:MemoryRecord)-[:PROJECTS_ENTITY]->(entity) }
              AND NOT EXISTS { MATCH (entity)-[:MEMORY_RELATION]-() }
            DELETE entity
            """
        )


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
    log.debug("memory.backends | Configured document store db=%r collection=%r", db_name, collection_name)
    log.debug("memory.backends | Using Mongo document store db=%r collection=%r", db_name, collection_name)
    return MongoMemoryDocumentStore(
        mongo_url=cfg.mongo_url,
        db_name=db_name,
        collection_name=collection_name,
    )


def get_memory_vector_store(profile: dict[str, Any]) -> ChromaMemoryVectorStore:
    """Build the configured vector store for memory records."""
    storage_cfg = profile.get("storage") or {}
    collection_name = storage_cfg.get("memory_chroma_collection") or storage_cfg.get("chroma_collection")
    log.debug("memory.backends | Configured vector store collection=%r", collection_name)
    log.debug("memory.backends | Using Chroma vector store collection=%r", collection_name)
    return ChromaMemoryVectorStore(collection_name=collection_name)


def get_memory_graph_store(profile: dict[str, Any]) -> MemoryGraphStore:
    """Build the graph projection store for memory records."""
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    backend = str(storage_cfg.get("memory_graph_backend") or getattr(cfg, "memory_graph_backend", "noop") or "noop").lower()
    log.debug("memory.backends | Configured graph backend=%r", backend)
    if backend in {"neo4j", "neo4j_memory"}:
        uri = storage_cfg.get("memory_neo4j_uri") or getattr(cfg, "memory_neo4j_uri", "")
        username = storage_cfg.get("memory_neo4j_username") or getattr(cfg, "memory_neo4j_username", "")
        password = storage_cfg.get("memory_neo4j_password") or getattr(cfg, "memory_neo4j_password", "")
        database = storage_cfg.get("memory_neo4j_database") or getattr(cfg, "memory_neo4j_database", None)
        if not uri or not username:
            log.error("memory.backends | Neo4j graph backend missing uri or username")
            raise ValueError("Neo4j memory graph backend requires memory_neo4j_uri and memory_neo4j_username.")
        log.debug(
            "memory.backends | Using Neo4j graph store uri=%r database=%r username=%r",
            uri,
            database,
            username,
        )
        return Neo4jMemoryGraphStore(
            uri=str(uri),
            username=str(username),
            password=str(password or ""),
            database=str(database) if database else None,
        )
    log.debug("memory.backends | Using noop graph store")
    return NoopMemoryGraphStore()


def _graph_node_payload(node: dict[str, Any]) -> dict[str, Any] | None:
    node_type = str(node.get("node_type") or "")
    node_key = str(node.get("node_key") or "")
    if not node_type or not node_key:
        return None
    metadata = dict(node.get("metadata") or {})
    return {
        "node_type": node_type,
        "node_key": node_key,
        "name": str(node.get("name") or node_key),
        "raw_key": str(node.get("raw_key") or metadata.get("raw_key") or node_key),
        "domain": str(metadata.get("domain") or metadata.get("profile") or ""),
        "profile": str(metadata.get("profile") or metadata.get("domain") or ""),
        "metadata_json": _json_safe_json(metadata),
    }


def _graph_edge_payload(edge: dict[str, Any]) -> dict[str, Any] | None:
    edge_type = str(edge.get("edge_type") or "")
    source_type = str(edge.get("source_type") or "")
    source_key = str(edge.get("source_key") or "")
    target_type = str(edge.get("target_type") or "")
    target_key = str(edge.get("target_key") or "")
    if not all((edge_type, source_type, source_key, target_type, target_key)):
        return None
    metadata = dict(edge.get("metadata") or {})
    return {
        "edge_type": edge_type,
        "source_type": source_type,
        "source_key": source_key,
        "target_type": target_type,
        "target_key": target_key,
        "source_raw_key": str(edge.get("source_raw_key") or metadata.get("source_raw_key") or source_key),
        "target_raw_key": str(edge.get("target_raw_key") or metadata.get("target_raw_key") or target_key),
        "domain": str(metadata.get("domain") or metadata.get("profile") or ""),
        "profile": str(metadata.get("profile") or metadata.get("domain") or ""),
        "metadata_json": _json_safe_json(metadata),
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
