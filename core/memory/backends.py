"""Storage backend abstractions for canonical memory records."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

import pymongo

from configs.config import get_config
from core.memory.models import MemoryRecord, ResearchKGUpdate
from core.memory.research_kg import GraphCanonicalCandidate, GraphCanonicalLookup
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

    def upsert(self, record: MemoryRecord, *, kg_update: ResearchKGUpdate) -> None: ...

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

    def list_candidates(self, *, domain: str, node_type: str, limit: int = 25) -> list[GraphCanonicalCandidate]: ...

    def reset(self, *, domain: str | None = None) -> None: ...


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


class NoopMemoryGraphStore(GraphCanonicalLookup):
    """Placeholder graph store until a concrete backend is configured."""

    def upsert(self, record: MemoryRecord, *, kg_update: ResearchKGUpdate) -> None:
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

    def list_candidates(self, *, domain: str, node_type: str, limit: int = 25) -> list[GraphCanonicalCandidate]:
        _ = (domain, node_type, limit)
        return []

    def reset(self, *, domain: str | None = None) -> None:
        _ = domain
        return None


@dataclass
class Neo4jMemoryGraphStore(GraphCanonicalLookup):
    """Neo4j-backed distilled research knowledge graph store."""

    uri: str
    username: str
    password: str
    database: str | None = None
    driver: Any | None = None

    def __post_init__(self) -> None:
        if self.driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:  # pragma: no cover
                log.error("memory.backends | Neo4j graph backend requested but neo4j package is missing")
                raise RuntimeError(
                    "Neo4j memory graph requires the 'neo4j' package. "
                    "Install project dependencies or add neo4j>=5.0."
                ) from exc
            log.debug("memory.backends | Creating Neo4j research KG driver uri=%r database=%r", self.uri, self.database)
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def upsert(self, record: MemoryRecord, *, kg_update: ResearchKGUpdate) -> None:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            log.warning("memory.backends | Skipping Neo4j KG upsert for record without id")
            return None
        payload = {
            "record_id": record_id,
            "domain": str(record.get("domain") or ""),
            "kind": str(record.get("kind") or ""),
            "object_type": str(record.get("object_type") or ""),
            "title": str(record.get("title") or ""),
            "summary": str(record.get("summary") or ""),
            "created_at": str(record.get("created_at") or ""),
            "metadata_json": _json_safe_json(record.get("metadata") or {}),
            "tags": [str(item) for item in (record.get("tags") or [])],
        }
        clean_nodes = [_kg_node_payload(node, payload["domain"]) for node in (kg_update.get("nodes") or [])]
        clean_nodes = [node for node in clean_nodes if node is not None]
        clean_relations = [_kg_relation_payload(relation) for relation in (kg_update.get("relations") or [])]
        clean_relations = [relation for relation in clean_relations if relation is not None]
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._upsert_projection, payload, clean_nodes, clean_relations)
        log.debug(
            "memory.backends | Neo4j KG upserted id=%r database=%r nodes=%d relations=%d",
            record_id,
            self.database,
            len(clean_nodes),
            len(clean_relations),
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
                result = session.execute_read(self._query_relations, params)
            else:
                result = session.execute_read(self._query_nodes, params)
        decoded = [_decode_graph_item(dict(item)) for item in result]
        log.debug("memory.backends | Neo4j KG query returned %d result(s)", len(decoded))
        return decoded

    def list_candidates(self, *, domain: str, node_type: str, limit: int = 25) -> list[GraphCanonicalCandidate]:
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(self._list_candidates, {"domain": domain, "node_type": node_type, "limit": int(limit)})
        candidates: list[GraphCanonicalCandidate] = []
        for item in rows:
            decoded = _decode_graph_item(dict(item))
            candidates.append(GraphCanonicalCandidate(
                canonical_id=str(decoded.get("canonical_id") or ""),
                display_name=str(decoded.get("display_name") or ""),
                aliases=[str(alias) for alias in (decoded.get("aliases") or []) if str(alias).strip()],
                properties=dict(decoded.get("properties") or decoded.get("metadata") or {}),
            ))
        return candidates

    def list_record_ids(self, *, domain: str | None = None) -> list[str]:
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(self._list_record_ids, str(domain) if domain else None)
        return [str(item) for item in rows if item]

    def delete_record(self, record_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._delete_record_projection, str(record_id))
        log.debug("memory.backends | Neo4j KG deleted record projection id=%r", record_id)

    def prune_orphan_entities(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._prune_orphan_entities)
        log.debug("memory.backends | Neo4j KG pruned orphan entities")

    def reset(self, *, domain: str | None = None) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._reset_graph, str(domain) if domain else None)
        log.info("memory.backends | Neo4j KG reset database=%r domain=%r", self.database, domain)

    @staticmethod
    def _upsert_projection(tx: Any, record_payload: dict[str, Any], nodes: list[dict[str, Any]], relations: list[dict[str, Any]]) -> None:
        record_id = record_payload["record_id"]
        tx.run(
            """
            MERGE (record:ResearchKGRecord {record_id: $record_id})
            SET record += $record
            WITH record
            OPTIONAL MATCH (record)-[projection:PROJECTS_NODE]->()
            DELETE projection
            WITH record
            OPTIONAL MATCH ()-[relation:KG_RELATION {record_id: $record_id}]->()
            DELETE relation
            """,
            record_id=record_id,
            record=record_payload,
        )
        for node in nodes:
            tx.run(
                """
                MERGE (entity:ResearchKGNode {canonical_id: $canonical_id})
                SET entity.node_type = $node_type,
                    entity.display_name = $display_name,
                    entity.domain = $domain,
                    entity.aliases = $aliases,
                    entity.properties_json = $properties_json
                WITH entity
                MATCH (record:ResearchKGRecord {record_id: $record_id})
                MERGE (record)-[:PROJECTS_NODE]->(entity)
                """,
                **node,
                record_id=record_id,
            )
        for relation in relations:
            tx.run(
                """
                MERGE (source:ResearchKGNode {canonical_id: $source_id})
                ON CREATE SET source.node_type = $source_type,
                              source.display_name = $source_id,
                              source.aliases = [],
                              source.properties_json = '{}',
                              source.domain = $domain
                MERGE (target:ResearchKGNode {canonical_id: $target_id})
                ON CREATE SET target.node_type = $target_type,
                              target.display_name = $target_id,
                              target.aliases = [],
                              target.properties_json = '{}',
                              target.domain = $domain
                MERGE (source)-[rel:KG_RELATION {
                    record_id: $record_id,
                    relation_type: $relation_type,
                    source_id: $source_id,
                    target_id: $target_id
                }]->(target)
                SET rel.properties_json = $properties_json,
                    rel.domain = $domain
                """,
                **relation,
                record_id=record_id,
            )

    @staticmethod
    def _query_nodes(tx: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = tx.run(
            """
            MATCH (node:ResearchKGNode)
            WHERE ($node_type IS NULL OR node.node_type = $node_type)
              AND ($node_key IS NULL OR node.canonical_id = $node_key OR node.display_name = $node_key)
            OPTIONAL MATCH (record:ResearchKGRecord)-[:PROJECTS_NODE]->(node)
            RETURN {
                node_type: node.node_type,
                canonical_id: node.canonical_id,
                display_name: node.display_name,
                aliases: node.aliases,
                metadata_json: node.properties_json,
                record_ids: collect(DISTINCT record.record_id)
            } AS item
            LIMIT $limit
            """,
            **params,
        )
        return [row["item"] for row in result]

    @staticmethod
    def _query_relations(tx: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = tx.run(
            """
            MATCH (source:ResearchKGNode)-[relation:KG_RELATION]->(target:ResearchKGNode)
            WHERE ($edge_type IS NULL OR relation.relation_type = $edge_type)
              AND ($node_type IS NULL OR source.node_type = $node_type OR target.node_type = $node_type)
              AND ($node_key IS NULL OR source.canonical_id = $node_key OR target.canonical_id = $node_key
                   OR source.display_name = $node_key OR target.display_name = $node_key)
            RETURN {
                relation_type: relation.relation_type,
                source_id: source.canonical_id,
                source_type: source.node_type,
                target_id: target.canonical_id,
                target_type: target.node_type,
                metadata_json: relation.properties_json,
                record_id: relation.record_id
            } AS item
            LIMIT $limit
            """,
            **params,
        )
        return [row["item"] for row in result]

    @staticmethod
    def _list_candidates(tx: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = tx.run(
            """
            MATCH (node:ResearchKGNode)
            WHERE node.domain = $domain AND node.node_type = $node_type
            RETURN {
                canonical_id: node.canonical_id,
                display_name: node.display_name,
                aliases: node.aliases,
                metadata_json: node.properties_json
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
            MATCH (record:ResearchKGRecord)
            WHERE ($domain IS NULL OR record.domain = $domain)
            RETURN record.record_id AS record_id
            """,
            domain=domain,
        )
        return [row.get("record_id") for row in result]

    @staticmethod
    def _delete_record_projection(tx: Any, record_id: str) -> None:
        tx.run(
            """
            MATCH (record:ResearchKGRecord {record_id: $record_id})
            OPTIONAL MATCH (record)-[projection:PROJECTS_NODE]->()
            DELETE projection
            WITH record
            OPTIONAL MATCH ()-[relation:KG_RELATION {record_id: $record_id}]->()
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
            MATCH (entity:ResearchKGNode)
            WHERE NOT EXISTS { MATCH (:ResearchKGRecord)-[:PROJECTS_NODE]->(entity) }
              AND NOT EXISTS { MATCH (entity)-[:KG_RELATION]-() }
            DELETE entity
            """
        )

    @staticmethod
    def _reset_graph(tx: Any, domain: str | None = None) -> None:
        tx.run(
            """
            MATCH (record:ResearchKGRecord)
            WHERE ($domain IS NULL OR record.domain = $domain)
            DETACH DELETE record
            """,
            domain=domain,
        )
        tx.run(
            """
            MATCH ()-[relation:KG_RELATION]->()
            WHERE ($domain IS NULL OR relation.domain = $domain)
            DELETE relation
            """,
            domain=domain,
        )
        tx.run(
            """
            MATCH (node:ResearchKGNode)
            WHERE ($domain IS NULL OR node.domain = $domain)
              AND NOT EXISTS { MATCH (:ResearchKGRecord)-[:PROJECTS_NODE]->(node) }
            DETACH DELETE node
            """,
            domain=domain,
        )
        tx.run(
            """
            MATCH (record:MemoryRecord)
            WHERE ($domain IS NULL OR record.domain = $domain OR record.profile = $domain)
            DETACH DELETE record
            """,
            domain=domain,
        )
        tx.run(
            """
            MATCH ()-[relation:MEMORY_RELATION]->()
            WHERE ($domain IS NULL OR relation.domain = $domain OR relation.profile = $domain)
            DELETE relation
            """,
            domain=domain,
        )
        tx.run(
            """
            MATCH (node:MemoryEntity)
            WHERE ($domain IS NULL OR node.domain = $domain OR node.profile = $domain)
            DETACH DELETE node
            """,
            domain=domain,
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


def _kg_node_payload(node: dict[str, Any], domain: str) -> dict[str, Any] | None:
    node_type = str(node.get("node_type") or "")
    canonical_id = str(node.get("canonical_id") or "")
    if not node_type or not canonical_id:
        return None
    properties = dict(node.get("properties") or {})
    return {
        "node_type": node_type,
        "canonical_id": canonical_id,
        "display_name": str(node.get("display_name") or canonical_id),
        "aliases": [str(item) for item in (node.get("aliases") or []) if str(item).strip()],
        "domain": str(domain or properties.get("domain") or ""),
        "properties_json": _json_safe_json(properties),
    }


def _kg_relation_payload(relation: dict[str, Any]) -> dict[str, Any] | None:
    relation_type = str(relation.get("relation_type") or "")
    source_id = str(relation.get("source_id") or "")
    target_id = str(relation.get("target_id") or "")
    if not all((relation_type, source_id, target_id)):
        return None
    source_type = source_id.split(":")[1] if ":" in source_id else ""
    target_type = target_id.split(":")[1] if ":" in target_id else ""
    properties = dict(relation.get("properties") or {})
    return {
        "relation_type": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "source_type": source_type,
        "target_type": target_type,
        "domain": str(properties.get("domain") or ""),
        "properties_json": _json_safe_json(properties),
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
