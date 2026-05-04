"""Orchestration layer for canonical memory persistence and retrieval."""
from __future__ import annotations

from typing import Any
from pathlib import Path

from core.memory.backends import (
    MemoryDocumentStore,
    MemoryGraphStore,
    MemoryVectorStore,
    get_memory_document_store,
    get_memory_graph_store,
    get_memory_vector_store,
)
from core.memory.defaults import (
    build_memory_record,
    default_memory_projection,
    ensure_memory_record_defaults,
    fingerprint_for_spec,
    memory_object_spec,
    record_from_vector_hit,
)
from core.memory.research_kg import build_research_kg_update
from core.memory.models import (
    MemoryObjectSpec,
    MemoryProjection,
    MemoryQuery,
    MemoryRecord,
    MemoryReuseResult,
    MemorySearchHit,
)
from core.utils.logger import get_logger

log = get_logger(__name__)


class MemoryService:
    """Coordinates document, vector, and graph memory backends."""

    def __init__(
        self,
        *,
        document_store: MemoryDocumentStore,
        vector_store: MemoryVectorStore,
        graph_store: MemoryGraphStore | None = None,
        profile: dict[str, Any] | None = None,
    ) -> None:
        self.document_store = document_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.profile = dict(profile or {})

    @classmethod
    def for_profile(cls, profile: dict[str, Any]) -> "MemoryService":
        storage_cfg = profile.get("storage") or {}
        log.debug(
            "memory.service | Building service for profile=%r mongo_db=%r chroma_collection=%r graph_backend=%r",
            profile.get("name"),
            storage_cfg.get("memory_mongodb_db") or storage_cfg.get("mongodb_results_db") or "researcher_results",
            storage_cfg.get("memory_chroma_collection") or storage_cfg.get("chroma_collection"),
            storage_cfg.get("memory_graph_backend"),
        )
        return cls(
            document_store=get_memory_document_store(profile),
            vector_store=get_memory_vector_store(profile),
            graph_store=get_memory_graph_store(profile),
            profile=profile,
        )

    def persist_record(
        self,
        record: MemoryRecord,
        *,
        projection: MemoryProjection | None = None,
    ) -> None:
        record = ensure_memory_record_defaults(record)
        plan = projection or default_memory_projection(record)
        record_id = str(record.get("record_id") or "")
        log.info(
            "memory.service | Persisting record id=%r domain=%r kind=%r object_type=%r",
            record_id,
            record.get("domain"),
            record.get("kind"),
            record.get("object_type"),
        )
        log.debug(
            "memory.service | Record id=%r title=%r tags=%s blob_refs=%d entities=%d relations=%d",
            record_id,
            record.get("title"),
            list(record.get("tags") or []),
            len(record.get("blob_refs") or []),
            len(record.get("entities") or []),
            len(record.get("relations") or []),
        )
        self.document_store.upsert(record)
        self.vector_store.upsert(
            record_id,
            str(plan.get("embedding_text") or ""),
            dict(plan.get("vector_metadata") or {}),
        )
        log.debug(
            "memory.service | Vector projection upserted id=%r embedding_chars=%d metadata_keys=%s",
            record_id,
            len(str(plan.get("embedding_text") or "")),
            sorted((plan.get("vector_metadata") or {}).keys()),
        )
        if self.graph_store is not None:
            kg_update = dict(plan.get("kg_update") or {})
            if not (kg_update.get("nodes") or kg_update.get("relations")):
                kg_update = build_research_kg_update(
                    record,
                    profile=self.profile,
                    canonical_lookup=self.graph_store,
                )
            self.graph_store.upsert(record, kg_update=kg_update)
            log.debug(
                "memory.service | Graph projection upserted id=%r nodes=%d relations=%d",
                record_id,
                len(kg_update.get("nodes") or []),
                len(kg_update.get("relations") or []),
            )
            log.debug(
                "memory.service | Graph payload id=%r node_types=%s relation_types=%s",
                record_id,
                [str(node.get("node_type") or "") for node in (kg_update.get("nodes") or [])],
                [str(edge.get("relation_type") or "") for edge in (kg_update.get("relations") or [])],
            )

    def persist_records(
        self,
        records: list[MemoryRecord],
        *,
        projections: dict[str, MemoryProjection] | None = None,
    ) -> None:
        log.debug("memory.service | Persisting %d memory record(s)", len(records))
        for record in records:
            record_id = str(record.get("record_id") or "")
            log.debug("memory.service | Persist batch includes id=%r", record_id)
            self.persist_record(record, projection=(projections or {}).get(record_id))

    def emit(
        self,
        *,
        profile: dict[str, Any],
        object_type: str,
        payload: dict[str, Any],
        node: str = "",
        kind: str = "",
        object_key: str = "",
        object_role: str = "artifact",
        title: str = "",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        spec: MemoryObjectSpec | None = None,
        source_state_keys: list[str] | None = None,
        source_record_ids: list[str] | None = None,
        blob_refs: list[dict[str, Any]] | None = None,
        entities: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
    ) -> MemoryRecord:
        """Build and persist one typed memory record from a node emission."""
        log.debug(
            "memory.service | Emitting memory object profile=%r node=%r object_type=%r role=%r",
            profile.get("name"),
            node,
            object_type,
            object_role,
        )
        record = build_memory_record(
            profile=profile,
            object_type=object_type,
            payload=payload,
            node=node,
            kind=kind,
            object_key=object_key,
            object_role=object_role,
            title=title,
            summary=summary,
            metadata=metadata or {},
            tags=tags or [],
            spec=spec,
            source_state_keys=source_state_keys or [],
            source_record_ids=source_record_ids or [],
            blob_refs=blob_refs or [],
            entities=entities or [],
            relations=relations or [],
        )
        log.debug(
            "memory.service | Built emitted record id=%r title=%r metadata_keys=%s",
            record.get("record_id"),
            record.get("title"),
            sorted((record.get("metadata") or {}).keys()),
        )
        self.persist_record(record)
        log.debug(
            "memory.service | Emitted memory record id=%r object_type=%r node=%r",
            record.get("record_id"),
            object_type,
            node,
        )
        return record

    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
        include_blobs: bool = False,
    ) -> list[MemorySearchHit]:
        log.debug(
            "memory.service | Semantic search query=%r n_results=%d filters=%s include_blobs=%s",
            query[:120],
            n_results,
            filters or {},
            include_blobs,
        )
        hits = self.vector_store.query_similar(query, n_results=n_results)
        output: list[MemorySearchHit] = []
        for hit in hits:
            record_id = str(hit.get("id") or "")
            try:
                record = self.document_store.get(record_id) if record_id else None
            except Exception as exc:
                log.warning("memory.service | Failed to hydrate vector hit id=%r: %s", record_id, exc)
                record = None
            if record is None:
                log.debug("memory.service | Using vector fallback record for id=%r", record_id)
                record = record_from_vector_hit(hit)
            if filters and not _matches_filters(record, filters):
                log.debug("memory.service | Search hit id=%r rejected by filters", record_id)
                continue
            if include_blobs:
                record = self.hydrate_blobs(record)
            output.append({
                "record": record,
                "distance": hit.get("distance"),
                "document": str(hit.get("document") or ""),
                "vector_metadata": dict(hit.get("metadata") or {}),
            })
        log.info("memory.service | Semantic search returned %d memory hit(s)", len(output))
        return output

    def query(self, request: MemoryQuery | None = None, **kwargs: Any) -> list[MemorySearchHit]:
        """Retrieve memory through one backend-agnostic request shape."""
        req: dict[str, Any] = {**(request or {}), **kwargs}
        filters = dict(req.get("filters") or {})
        for key in ("domain", "kind", "object_type", "object_key", "object_role"):
            if req.get(key):
                filters[key] = req[key]
        if req.get("tags"):
            filters["tags"] = {"$all": list(req["tags"])}
        fingerprint = req.get("fingerprint")
        if fingerprint:
            metadata_key = req.get("fingerprint_metadata_key") or "fingerprint"
            filters[f"metadata.{metadata_key}"] = fingerprint

        limit = int(req.get("limit") or req.get("n_results") or 50)
        include_blobs = bool(req.get("include_blobs"))
        semantic_query = str(req.get("query") or "")
        log.debug(
            "memory.service | Query request semantic=%s filters=%s limit=%d include_blobs=%s",
            bool(semantic_query),
            filters,
            limit,
            include_blobs,
        )
        if semantic_query:
            return self.search(
                semantic_query,
                n_results=int(req.get("n_results") or limit),
                filters=filters or None,
                include_blobs=include_blobs,
            )

        records = self.find_records(filters, limit=limit)
        log.info("memory.service | Retrieved %d memory record(s) from structured query", len(records))
        output: list[MemorySearchHit] = []
        for record in records:
            if include_blobs:
                record = self.hydrate_blobs(record)
            output.append({
                "record": record,
                "distance": None,
                "document": str(record.get("summary") or ""),
                "vector_metadata": {},
            })
        return output

    def find_reusable(
        self,
        *,
        domain: str,
        object_type: str,
        fingerprint: str,
        fingerprint_metadata_key: str = "fingerprint",
        status_metadata_key: str | None = None,
        ready_statuses: list[str] | None = None,
        required_blob_names: list[str] | None = None,
        extra_filters: dict[str, Any] | None = None,
    ) -> MemoryReuseResult:
        """Find an exact reusable object without exposing backend details."""
        filters = {
            "domain": domain,
            "object_type": object_type,
            f"metadata.{fingerprint_metadata_key}": fingerprint,
            **(extra_filters or {}),
        }
        if status_metadata_key and ready_statuses:
            filters[f"metadata.{status_metadata_key}"] = {"$in": ready_statuses}

        log.debug(
            "memory.service | Reuse lookup domain=%r object_type=%r fingerprint_key=%r fingerprint=%r",
            domain,
            object_type,
            fingerprint_metadata_key,
            fingerprint,
        )
        record = self.find_one_record(filters)
        if not record:
            log.info("memory.service | Reuse miss object_type=%r fingerprint=%r reason=not_found", object_type, fingerprint)
            return {"reusable": False, "record": None, "reason": "not_found", "fingerprint": fingerprint}
        validity = record.get("validity") or {}
        if validity.get("reusable") is False:
            log.info("memory.service | Reuse rejected id=%r reason=marked_not_reusable", record.get("record_id"))
            return {"reusable": False, "record": record, "reason": "marked_not_reusable", "fingerprint": fingerprint}
        if str(validity.get("status") or "").lower() in {"invalid", "stale", "superseded"}:
            log.info(
                "memory.service | Reuse rejected id=%r reason=status:%s",
                record.get("record_id"),
                validity.get("status"),
            )
            return {"reusable": False, "record": record, "reason": f"status:{validity.get('status')}", "fingerprint": fingerprint}

        resolved = self.resolve_blob_refs(record)
        for name in required_blob_names or []:
            if name not in resolved:
                log.info(
                    "memory.service | Reuse rejected id=%r reason=missing_blob:%s",
                    record.get("record_id"),
                    name,
                )
                return {"reusable": False, "record": record, "reason": f"missing_blob:{name}", "fingerprint": fingerprint, "resolved_blobs": resolved}
        log.info("memory.service | Reuse hit id=%r object_type=%r", record.get("record_id"), object_type)
        return {"reusable": True, "record": record, "reason": "matched", "fingerprint": fingerprint, "resolved_blobs": resolved}

    def find_reusable_for_profile(
        self,
        profile: dict[str, Any],
        *,
        object_type: str,
        fingerprint: str,
        extra_filters: dict[str, Any] | None = None,
    ) -> MemoryReuseResult:
        """Find a reusable object using the profile-declared object spec."""
        spec = memory_object_spec(profile, object_type) or {}
        if not spec:
            log.debug(
                "memory.service | No profile memory spec for profile=%r object_type=%r; using defaults",
                profile.get("name"),
                object_type,
            )
        return self.find_reusable(
            domain=str(profile.get("name") or ""),
            object_type=object_type,
            fingerprint=fingerprint,
            fingerprint_metadata_key=str(spec.get("fingerprint_metadata_key") or "fingerprint"),
            status_metadata_key=str(spec.get("status_metadata_key") or "") or None,
            ready_statuses=list(spec.get("ready_statuses") or []),
            required_blob_names=list(spec.get("required_blob_names") or []),
            extra_filters=extra_filters or {},
        )

    def find_records(self, filters: dict[str, Any], *, limit: int = 50) -> list[MemoryRecord]:
        log.debug("memory.service | Document find filters=%s limit=%d", filters, limit)
        records = self.document_store.find(filters, limit=limit)
        log.debug("memory.service | Document find returned %d record(s)", len(records))
        return records

    def find_one_record(self, filters: dict[str, Any]) -> MemoryRecord | None:
        records = self.find_records(filters, limit=1)
        return records[0] if records else None

    def resolve_blob_refs(self, record: MemoryRecord) -> dict[str, str]:
        """Resolve usable local/blob URIs by blob name while staying backend-blind."""
        resolved: dict[str, str] = {}
        for ref in record.get("blob_refs") or []:
            name = str(ref.get("name") or ref.get("blob_id") or "")
            uri = str(ref.get("uri") or "")
            if not name or not uri:
                log.debug("memory.service | Skipping incomplete blob ref on record=%r", record.get("record_id"))
                continue
            path = _local_path_from_uri(uri)
            if path is None or path.exists():
                resolved[name] = uri if path is None else str(path)
            else:
                log.warning(
                    "memory.service | Blob ref missing record=%r name=%r uri=%r",
                    record.get("record_id"),
                    name,
                    uri,
                )
        log.debug("memory.service | Resolved %d blob ref(s) for record=%r", len(resolved), record.get("record_id"))
        return resolved

    def hydrate_blobs(self, record: MemoryRecord) -> MemoryRecord:
        hydrated = dict(record)
        metadata = dict(hydrated.get("metadata") or {})
        metadata["resolved_blobs"] = self.resolve_blob_refs(record)
        hydrated["metadata"] = metadata
        return hydrated

    def repair_projections(self, filters: dict[str, Any] | None = None, *, limit: int = 1000) -> int:
        """Rebuild vector and graph projections from canonical document records."""
        log.info("memory.service | Repairing memory projections filters=%s limit=%d", filters or {}, limit)
        records = self.find_records(filters or {}, limit=limit)
        for record in records:
            self.persist_record(record)
        log.info("memory.service | Repaired %d memory projection(s)", len(records))
        return len(records)

    def rebuild_graph_from_documents(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 1000,
        reset_first: bool = True,
    ) -> int:
        """Reset Neo4j graph content and rebuild it from canonical Mongo records."""
        if self.graph_store is None:
            log.warning("memory.service | Graph rebuild skipped; graph store not configured")
            return 0
        active_filters = dict(filters or {})
        domain = str(active_filters.get("domain") or self.profile.get("name") or "")
        log.info(
            "memory.service | Rebuilding graph from documents filters=%s limit=%d reset_first=%s",
            active_filters,
            limit,
            reset_first,
        )
        if reset_first:
            try:
                self.graph_store.reset(domain=domain or None)
            except Exception as exc:
                log.error("memory.service | Graph reset failed domain=%r: %s", domain, exc)
                raise
        records = self.find_records(active_filters, limit=limit)
        for record in records:
            kg_update = build_research_kg_update(
                record,
                profile=self.profile,
                canonical_lookup=self.graph_store,
            )
            self.graph_store.upsert(record, kg_update=kg_update)
        try:
            self.graph_store.prune_orphan_entities()
        except Exception as exc:
            log.warning("memory.service | Graph prune after rebuild failed: %s", exc)
        log.info("memory.service | Rebuilt graph from %d record(s)", len(records))
        return len(records)

    def graph_query(
        self,
        *,
        node_type: str | None = None,
        node_key: str | None = None,
        edge_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if self.graph_store is None:
            log.debug("memory.service | Graph query skipped; graph store not configured")
            return []
        results = self.graph_store.query(
            node_type=node_type,
            node_key=node_key,
            edge_type=edge_type,
            limit=limit,
        )
        log.info("memory.service | Graph query returned %d result(s)", len(results))
        return results

    def fingerprint_for_spec(self, spec: MemoryObjectSpec, payload: dict[str, Any]) -> str:
        return fingerprint_for_spec(spec, payload)


def _local_path_from_uri(uri: str) -> Path | None:
    if uri.startswith("file://"):
        return Path(uri.removeprefix("file://"))
    if "://" in uri:
        return None
    return Path(uri)


def _matches_filters(record: MemoryRecord, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = _get_nested(record, key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$all" in expected:
                actual_list = actual if isinstance(actual, list) else []
                if not all(item in actual_list for item in expected["$all"]):
                    return False
        elif actual != expected:
            return False
    return True


def _get_nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
