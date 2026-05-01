from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from configs.config import get_config
from core.artifacts.store import MongoArtifactMetadataStore
from core.memory import MemoryService
from core.memory.backends import get_memory_document_store
from core.memory.backends import get_memory_graph_store, get_memory_vector_store
from core.utils.profile_loader import list_profiles, load_profile


@dataclass
class ProfileContext:
    name: str
    profile: dict[str, Any]
    memory_service: MemoryService


def load_profile_contexts() -> list[ProfileContext]:
    contexts: list[ProfileContext] = []
    for profile_name in list_profiles():
        try:
            profile = load_profile(profile_name)
            contexts.append(
                ProfileContext(
                    name=profile_name,
                    profile=profile,
                    memory_service=MemoryService.for_profile(profile),
                )
            )
        except Exception:
            continue
    return contexts


def list_run_summaries(*, profile_name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for ctx in load_profile_contexts():
        if profile_name and ctx.name != profile_name:
            continue
        try:
            records = ctx.memory_service.find_records({"object_type": "experiment_result"}, limit=limit)
        except Exception:
            continue
        for record in records:
            metadata = dict(record.get("metadata") or {})
            runs.append({
                "profile_name": ctx.name,
                "record_id": str(record.get("record_id") or ""),
                "title": str(record.get("title") or ""),
                "created_at": str(record.get("created_at") or ""),
                "experiment_id": str(metadata.get("experiment_id") or ""),
                "proposal_name": str(metadata.get("proposal_name") or ""),
                "dataset": str(metadata.get("dataset") or ""),
                "detector": str(metadata.get("detector") or ""),
                "assessment": str(metadata.get("assessment") or ""),
                "mlflow_run_id": str(metadata.get("mlflow_run_id") or ""),
                "primary_metric_name": _primary_metric_name(ctx.profile, metadata),
                "primary_metric_value": metadata.get(_primary_metric_name(ctx.profile, metadata), None),
            })
    runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return runs[:limit]


def diagnostics() -> dict[str, Any]:
    cfg = get_config()
    profiles: list[dict[str, Any]] = []
    for profile_name in list_profiles():
        item: dict[str, Any] = {"profile_name": profile_name}
        try:
            profile = load_profile(profile_name)
            storage_cfg = profile.get("storage") or {}
            doc_store = get_memory_document_store(profile)
            db_name = storage_cfg.get("memory_mongodb_db") or storage_cfg.get("mongodb_results_db") or "researcher_results"
            collection_name = storage_cfg.get("memory_mongodb_collection", "memory_records")
            item.update({
                "loaded": True,
                "memory_db": db_name,
                "memory_collection": collection_name,
                "chroma_collection": storage_cfg.get("memory_chroma_collection") or storage_cfg.get("chroma_collection"),
                "graph_backend": storage_cfg.get("memory_graph_backend") or getattr(cfg, "memory_graph_backend", "noop"),
                "artifact_db": storage_cfg.get("artifacts_mongodb_db") or getattr(cfg, "artifacts_db_name", "researcher_artifacts"),
                "artifact_collection": storage_cfg.get("artifacts_collection") or getattr(cfg, "artifacts_collection", "artifacts"),
            })
            try:
                item["object_counts"] = _diagnostic_object_counts(doc_store)
                item["experiment_result_count"] = int((item["object_counts"] or {}).get("experiment_result", 0))
                if item["experiment_result_count"] == 0 and item["object_counts"]:
                    item["empty_reason"] = (
                        "No experiment_result records were found. "
                        "The pipeline may have only persisted earlier-stage objects "
                        "such as featuresets or datasets."
                    )
            except Exception as exc:
                item["object_counts"] = {}
                item["experiment_result_count"] = None
                item["count_error"] = str(exc)
        except Exception as exc:
            item.update({
                "loaded": False,
                "error": str(exc),
            })
        profiles.append(item)
    return {
        "mongo_url": getattr(cfg, "mongo_url", ""),
        "profiles": profiles,
    }


def get_run_bundle(profile_name: str, record_id: str) -> dict[str, Any]:
    ctx = _load_context(profile_name)
    record = ctx.memory_service.document_store.get(record_id)
    if not record:
        raise KeyError(f"Run record not found: profile={profile_name} record_id={record_id}")

    metadata = dict(record.get("metadata") or {})
    try:
        vector_hit = get_memory_vector_store(ctx.profile).get_by_id(record_id)
    except Exception as exc:
        vector_hit = {"error": str(exc)}
    related_records = _related_records(ctx, record)
    artifacts = _artifact_records(ctx.profile, record, metadata)
    graph = _graph_bundle(ctx, record)
    mlflow_bundle = _mlflow_bundle(ctx.profile, metadata)

    return {
        "profile_name": profile_name,
        "run": {
            "record": record,
            "summary": {
                "record_id": record_id,
                "experiment_id": str(metadata.get("experiment_id") or ""),
                "proposal_name": str(metadata.get("proposal_name") or ""),
                "dataset": str(metadata.get("dataset") or ""),
                "detector": str(metadata.get("detector") or ""),
                "assessment": str(metadata.get("assessment") or ""),
                "hypothesis_supported": metadata.get("hypothesis_supported"),
                "created_at": str(record.get("created_at") or ""),
                "mlflow_run_id": str(metadata.get("mlflow_run_id") or ""),
                "primary_metric_name": _primary_metric_name(ctx.profile, metadata),
                "primary_metric_value": metadata.get(_primary_metric_name(ctx.profile, metadata), None),
            },
        },
        "backend": {
            "mongo": {
                "memory_record": record,
                "related_records": related_records,
                "artifacts": artifacts,
            },
            "chroma": {
                "record": vector_hit,
            },
            "neo4j": graph,
            "mlflow": mlflow_bundle,
        },
    }


def _load_context(profile_name: str) -> ProfileContext:
    profile = load_profile(profile_name)
    return ProfileContext(
        name=profile_name,
        profile=profile,
        memory_service=MemoryService.for_profile(profile),
    )


def _primary_metric_name(profile: dict[str, Any], metadata: dict[str, Any]) -> str:
    configured = str((profile.get("evaluation") or {}).get("primary_metric") or "").strip()
    if configured:
        return configured
    for fallback in ("test_auc", "best_metric_value"):
        if fallback in metadata:
            return fallback
    return "metric"


def _related_records(ctx: ProfileContext, run_record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = dict(run_record.get("metadata") or {})
    proposal_name = str(metadata.get("proposal_name") or "")
    research_direction = str(metadata.get("research_direction") or "")
    filters: list[dict[str, Any]] = []
    if proposal_name:
        filters.append({"metadata.proposal_name": proposal_name})
    if research_direction:
        filters.append({"metadata.research_direction": research_direction})

    seen: set[str] = {str(run_record.get("record_id") or "")}
    output: list[dict[str, Any]] = []
    for record_filter in filters:
        for record in ctx.memory_service.find_records(record_filter, limit=200):
            record_key = str(record.get("record_id") or "")
            if not record_key or record_key in seen:
                continue
            seen.add(record_key)
            output.append(record)
    output.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return output


def _artifact_records(profile: dict[str, Any], run_record: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    try:
        store = MongoArtifactMetadataStore(
            mongo_url=cfg.mongo_url,
            db_name=storage_cfg.get("artifacts_mongodb_db") or getattr(cfg, "artifacts_db_name", "researcher_artifacts"),
            collection_name=storage_cfg.get("artifacts_collection") or getattr(cfg, "artifacts_collection", "artifacts"),
        )
    except Exception as exc:
        return [{"error": str(exc)}]

    refs = list(run_record.get("blob_refs") or [])
    artifact_ids = [
        str(ref.get("artifact_id") or "")
        for ref in refs
        if isinstance(ref, dict) and ref.get("artifact_id")
    ]

    records: list[dict[str, Any]] = []
    for artifact_id in artifact_ids:
        try:
            record = store.get(artifact_id)
        except Exception as exc:
            records.append({"artifact_id": artifact_id, "error": str(exc)})
            continue
        if record:
            records.append(record)

    proposal_name = str(metadata.get("proposal_name") or "")
    if proposal_name:
        try:
            for record in store.find({"proposal_name": proposal_name}, limit=200):
                artifact_id = str(record.get("artifact_id") or "")
                if artifact_id and artifact_id not in artifact_ids:
                    records.append(record)
                    artifact_ids.append(artifact_id)
        except Exception as exc:
            records.append({"proposal_name": proposal_name, "error": str(exc)})

    return records


def _graph_bundle(ctx: ProfileContext, run_record: dict[str, Any]) -> dict[str, Any]:
    entities = list(run_record.get("entities") or [])
    relations = list(run_record.get("relations") or [])
    node_results: list[dict[str, Any]] = []
    edge_results: list[dict[str, Any]] = []
    try:
        graph_store = get_memory_graph_store(ctx.profile)
    except Exception as exc:
        return {
            "backend_enabled": False,
            "nodes": entities,
            "edges": relations,
            "queried_nodes": [],
            "queried_edges": [],
            "error": str(exc),
        }

    if graph_store.__class__.__name__.lower().startswith("noop"):
        return {
            "backend_enabled": False,
            "nodes": entities,
            "edges": relations,
            "queried_nodes": [],
            "queried_edges": [],
        }

    for entity in entities:
        node_type = str(entity.get("entity_type") or "")
        node_key = str(entity.get("key") or "")
        if not node_type or not node_key:
            continue
        try:
            node_results.extend(graph_store.query(node_type=node_type, node_key=node_key, limit=50))
        except Exception as exc:
            node_results.append({
                "node_type": node_type,
                "node_key": node_key,
                "error": str(exc),
            })

    return {
        "backend_enabled": True,
        "nodes": entities,
        "edges": relations,
        "queried_nodes": _dedupe_dicts(node_results, ("node_type", "node_key")),
        "queried_edges": _dedupe_dicts(edge_results, ("edge_type", "source_type", "source_key", "target_type", "target_key")),
    }


def _mlflow_bundle(profile: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    cfg = get_config()
    run_id = str(metadata.get("mlflow_run_id") or "")
    tracking_uri = str(metadata.get("mlflow_tracking_uri") or getattr(cfg, "mlflow_uri", "") or "")
    experiment_name = str(metadata.get("mlflow_experiment") or (profile.get("storage") or {}).get("mlflow_experiment", ""))
    if not run_id or not tracking_uri:
        return {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "run": None,
            "ui_url": "",
        }

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        experiment_id = str(run.info.experiment_id)
        return {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "run": {
                "info": {
                    "run_id": run.info.run_id,
                    "experiment_id": experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                },
                "data": {
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags),
                },
            },
            "ui_url": f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}",
        }
    except Exception as exc:
        return {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "run": None,
            "ui_url": "",
            "error": str(exc),
        }


def _dedupe_dicts(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        marker = tuple(item.get(key) for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def _diagnostic_object_counts(doc_store: Any, *, limit: int = 2000) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in doc_store.find({}, limit=limit):
        object_type = str(record.get("object_type") or record.get("kind") or "unknown")
        counts[object_type] = counts.get(object_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))
