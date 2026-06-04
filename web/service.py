from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pymongo
from mlflow.tracking import MlflowClient

from configs.config import get_config
from core.artifacts.store import MongoArtifactMetadataStore
from core.handoffs import (
    build_handoff_cli_command,
    build_handoff_draft,
    build_next_step_cli_command,
    build_handoff_prompt_preview,
    build_proposal_seed_cli_command,
    build_proposal_seed_draft,
    build_proposal_seed_preview,
    handoff_record_to_summary,
    list_proposal_seeds,
    list_run_handoffs,
    proposal_seed_record_to_summary,
    save_proposal_seed,
    save_run_handoff,
)
from core.maintenance import delete_profile_orphans, scan_profiles
from core.memory import MemoryService
from core.memory.backends import get_memory_document_store
from core.memory.backends import get_memory_graph_store, get_memory_vector_store
from core.utils.profile_loader import list_profiles, load_profile
from core.brainstorm import (
    BrainstormEngine,
    create_brainstorm_state,
    execute_brainstorm_handoff,
    load_brainstorm_config,
    load_brainstorm_session,
    persist_brainstorm_session,
    resolve_brainstorm_seed,
)


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
        seen_ids: set[str] = set()
        seen_run_keys: set[tuple[str, str]] = set()
        try:
            records = ctx.memory_service.find_records({"object_type": "experiment_result"}, limit=limit)
        except Exception:
            continue
        for record in records:
            metadata = dict(record.get("metadata") or {})
            record_id = str(record.get("record_id") or "")
            experiment_id = str(metadata.get("experiment_id") or "")
            if record_id:
                seen_ids.add(record_id)
            if experiment_id:
                seen_ids.add(experiment_id)
            seen_run_keys.add((
                str(metadata.get("root_run_family_id") or ""),
                str(metadata.get("research_direction") or ""),
            ))
            mlflow_bundle = _mlflow_bundle(ctx.profile, metadata)
            runs.append({
                "profile_name": ctx.name,
                "record_id": record_id,
                "title": str(record.get("title") or ""),
                "created_at": str(record.get("created_at") or ""),
                "experiment_id": experiment_id,
                "proposal_name": str(metadata.get("proposal_name") or ""),
                "dataset": str(metadata.get("dataset") or ""),
                "detector": str(metadata.get("detector") or ""),
                "assessment": str(metadata.get("assessment") or ""),
                "mlflow_run_id": str(metadata.get("mlflow_run_id") or ""),
                "mlflow_ui_url": str(mlflow_bundle.get("ui_url") or ""),
                "root_run_family_id": str(metadata.get("root_run_family_id") or ""),
                "root_research_direction": str(metadata.get("root_research_direction") or metadata.get("research_direction") or ""),
                "source_next_step_title": str(metadata.get("source_next_step_title") or ""),
                "primary_metric_name": _primary_metric_name(ctx.profile, metadata),
                "primary_metric_value": metadata.get(_primary_metric_name(ctx.profile, metadata), None),
                "source": "memory",
            })
        for record in _pipeline_run_summaries(ctx, seen_run_keys, limit=limit):
            record_id = str(record.get("record_id") or "")
            if record_id:
                seen_ids.add(record_id)
            runs.append(record)
        for raw_run in _raw_result_summaries(ctx.profile, limit=limit, seen_ids=seen_ids):
            runs.append({"profile_name": ctx.name, **raw_run})
    runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return runs[:limit]


def _pipeline_run_summaries(ctx: ProfileContext, seen_run_keys: set[tuple[str, str]], *, limit: int) -> list[dict[str, Any]]:
    try:
        records = ctx.memory_service.find_records({"object_type": "pipeline_run"}, limit=limit)
    except Exception:
        return []
    output: list[dict[str, Any]] = []
    for record in records:
        metadata = dict(record.get("metadata") or {})
        family_id = str(metadata.get("root_run_family_id") or "")
        direction = str(metadata.get("research_direction") or "")
        if (family_id, direction) in seen_run_keys:
            continue
        output.append({
            "profile_name": ctx.name,
            "record_id": str(record.get("record_id") or ""),
            "title": str(record.get("title") or metadata.get("research_direction") or ""),
            "created_at": str(record.get("created_at") or ""),
            "experiment_id": "",
            "proposal_name": "pipeline run",
            "dataset": "",
            "detector": "",
            "assessment": "planning",
            "mlflow_run_id": "",
            "mlflow_ui_url": "",
            "root_run_family_id": family_id,
            "root_research_direction": str(metadata.get("root_research_direction") or metadata.get("research_direction") or ""),
            "source_next_step_title": str(metadata.get("source_next_step_title") or ""),
            "primary_metric_name": "",
            "primary_metric_value": None,
            "source": "pipeline_memory",
        })
    return output


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


def scan_orphans(*, profile_name: str | None = None) -> dict[str, Any]:
    return scan_profiles(_load_profiles(profile_name))


def delete_orphans(*, profile_name: str | None = None) -> dict[str, Any]:
    return delete_profile_orphans(_load_profiles(profile_name))


def create_run_handoff(profile_name: str, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _load_context(profile_name)
    record = ctx.memory_service.document_store.get(record_id)
    if not record:
        record = _raw_result_record(ctx.profile, record_id)
    if not record:
        raise KeyError(f"Run record not found: profile={profile_name} record_id={record_id}")

    saved = save_run_handoff(ctx.profile, record, payload)
    saved_records = list_run_handoffs(ctx.profile, source_experiment_record_id=str(record.get("record_id") or ""), limit=20)
    saved_summaries = [handoff_record_to_summary(profile_name, item) for item in saved_records]
    return {
        "handoff": handoff_record_to_summary(profile_name, saved),
        "saved_handoffs": saved_summaries,
    }


def create_proposal_seed(profile_name: str, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _load_context(profile_name)
    record = ctx.memory_service.document_store.get(record_id)
    if not record:
        record = _raw_result_record(ctx.profile, record_id)
    if not record:
        raise KeyError(f"Run record not found: profile={profile_name} record_id={record_id}")

    related_records = _related_records(ctx, record)
    saved = save_proposal_seed(ctx.profile, record, payload)
    saved_records = list_proposal_seeds(ctx.profile, source_experiment_record_id=str(record.get("record_id") or ""), limit=20)
    saved_summaries = [proposal_seed_record_to_summary(profile_name, item) for item in saved_records]
    draft = build_proposal_seed_draft(profile_name, record, related_records, _run_text_panels(record, related_records))
    return {
        "proposal_seed": proposal_seed_record_to_summary(profile_name, saved),
        "saved_proposal_seeds": saved_summaries,
        "draft": {
            **draft,
            "prompt_preview": build_proposal_seed_preview(draft),
        },
    }


def get_run_bundle(profile_name: str, record_id: str) -> dict[str, Any]:
    ctx = _load_context(profile_name)
    record = ctx.memory_service.document_store.get(record_id)
    if not record:
        record = _raw_result_record(ctx.profile, record_id)
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
    family = _family_bundle(ctx, record)
    mlflow_bundle = _mlflow_bundle(ctx.profile, metadata)
    text_panels = _run_text_panels(record, related_records)
    handoff_draft = build_handoff_draft(profile_name, record, text_panels)
    proposal_seed_draft = build_proposal_seed_draft(profile_name, record, related_records, text_panels)
    saved_handoffs = [
        handoff_record_to_summary(profile_name, item)
        for item in list_run_handoffs(ctx.profile, source_experiment_record_id=str(record.get("record_id") or ""), limit=20)
    ]
    saved_proposal_seeds = [
        proposal_seed_record_to_summary(profile_name, item)
        for item in list_proposal_seeds(ctx.profile, source_experiment_record_id=str(record.get("record_id") or ""), limit=20)
    ]

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
                "mlflow_ui_url": str(mlflow_bundle.get("ui_url") or ""),
                "root_run_family_id": str(metadata.get("root_run_family_id") or ""),
                "root_research_direction": str(metadata.get("root_research_direction") or metadata.get("research_direction") or ""),
                "source_next_step_title": str(metadata.get("source_next_step_title") or ""),
                "primary_metric_name": _primary_metric_name(ctx.profile, metadata),
                "primary_metric_value": metadata.get(_primary_metric_name(ctx.profile, metadata), None),
            },
            "text_panels": text_panels,
            "next_steps": _next_step_summaries(profile_name, related_records),
            "family": family,
            "handoff": {
                "draft": {
                    **handoff_draft,
                    "prompt_preview": build_handoff_prompt_preview(handoff_draft),
                    "copy_command": "",
                },
                "saved": saved_handoffs,
                "copy_command_template": build_handoff_cli_command(
                    profile_name,
                    source_experiment_record_id=str(record.get("record_id") or ""),
                    handoff_record_id="<handoff-record-id>",
                ),
            },
            "proposal_seed": {
                "draft": {
                    **proposal_seed_draft,
                    "prompt_preview": build_proposal_seed_preview(proposal_seed_draft),
                    "copy_command": "",
                },
                "saved": saved_proposal_seeds,
                "copy_command_template": build_proposal_seed_cli_command(
                    profile_name,
                    source_experiment_record_id=str(record.get("record_id") or ""),
                    proposal_seed_record_id="<proposal-seed-record-id>",
                ),
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


def create_brainstorm_session(profile_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile(profile_name)
    brainstorm_cfg = load_brainstorm_config(payload.get("brainstorm_config"))
    seed = resolve_brainstorm_seed(
        profile,
        source_experiment_record_id=str(payload.get("source_experiment") or payload.get("source_experiment_record_id") or "").strip(),
        handoff_record_id=str(payload.get("handoff") or payload.get("handoff_record_id") or "").strip(),
        proposal_seed_record_id=str(payload.get("proposal_seed") or payload.get("proposal_seed_record_id") or "").strip(),
        next_step_record_id=str(payload.get("next_step") or payload.get("next_step_record_id") or "").strip(),
    )
    direction = str(payload.get("direction") or seed.get("research_direction") or "").strip()
    if not direction:
        raise ValueError("direction is required when no brainstorm seed source is provided")
    engine = BrainstormEngine(profile, brainstorm_cfg)
    state = create_brainstorm_state(
        profile_name=profile_name,
        direction=direction,
        brainstorm_cfg=brainstorm_cfg,
        session_id=str(payload.get("session_id") or "").strip() or None,
        seed=seed,
    )
    if bool(payload.get("autorun", True)):
        state = engine.run_until_pause(state)
    persist_brainstorm_session(profile, brainstorm_cfg, state)
    return {
        "profile_name": profile_name,
        "session_id": state.get("session_id"),
        "status": state.get("status"),
        "summary": state.get("last_summary", ""),
        "state": state,
    }


def get_brainstorm_session(profile_name: str, session_id: str) -> dict[str, Any]:
    profile = load_profile(profile_name)
    state = load_brainstorm_session(profile, session_id)
    return {
        "profile_name": profile_name,
        "session_id": session_id,
        "status": state.get("status"),
        "summary": state.get("last_summary", ""),
        "state": state,
    }


def command_brainstorm_session(profile_name: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile(profile_name)
    state = load_brainstorm_session(profile, session_id)
    brainstorm_cfg = load_brainstorm_config(payload.get("brainstorm_config") or state.get("brainstorm_config_path"))
    engine = BrainstormEngine(profile, brainstorm_cfg)
    command = payload.get("command") or payload.get("text") or ""
    state = engine.apply_command(state, command)
    persist_brainstorm_session(profile, brainstorm_cfg, state)
    return {
        "profile_name": profile_name,
        "session_id": session_id,
        "status": state.get("status"),
        "summary": state.get("last_summary", ""),
        "state": state,
    }


def execute_brainstorm_session(profile_name: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile(profile_name)
    state = load_brainstorm_session(profile, session_id)
    brainstorm_cfg = load_brainstorm_config(payload.get("brainstorm_config") or state.get("brainstorm_config_path"))
    start_node, result = execute_brainstorm_handoff(
        state,
        brainstorm_cfg,
        build_initial_state_fn=_build_initial_state_for_web,
        run_pipeline_graph_fn=_run_pipeline_graph_for_web,
        profile_name=profile_name,
        profile=profile,
    )
    state["status"] = "completed"
    persist_brainstorm_session(profile, brainstorm_cfg, state)
    return {
        "profile_name": profile_name,
        "session_id": session_id,
        "start_node": start_node,
        "result": result,
    }


def _build_initial_state_for_web(
    profile_name: str,
    direction: str,
    seed: dict[str, Any],
    *,
    continue_loop: bool,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from main import build_initial_state

    return build_initial_state(
        profile_name,
        direction,
        seed,
        continue_loop=continue_loop,
        extra_state=extra_state,
    )


def _run_pipeline_graph_for_web(
    profile_name: str,
    profile: dict[str, Any],
    *,
    initial_state: dict[str, Any],
    start_node: str,
    print_results: bool = False,
) -> dict[str, Any]:
    from main import run_pipeline_graph

    return run_pipeline_graph(
        profile_name,
        profile,
        initial_state=initial_state,
        start_node=start_node,
        print_results=print_results,
    )


def _load_context(profile_name: str) -> ProfileContext:
    profile = load_profile(profile_name)
    return ProfileContext(
        name=profile_name,
        profile=profile,
        memory_service=MemoryService.for_profile(profile),
    )


def _load_profiles(profile_name: str | None = None) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for name in list_profiles():
        if profile_name and name != profile_name:
            continue
        profiles[name] = load_profile(name)
    return profiles


def _results_store(profile: dict[str, Any]) -> tuple[Any, str, str]:
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    client = pymongo.MongoClient(cfg.mongo_url)
    db_name = storage_cfg.get("mongodb_results_db", "researcher_results")
    collection_name = storage_cfg.get("mongodb_results_collection", "experiments")
    return client, str(db_name), str(collection_name)


def _raw_result_summaries(profile: dict[str, Any], *, limit: int, seen_ids: set[str]) -> list[dict[str, Any]]:
    client, db_name, collection_name = _results_store(profile)
    try:
        docs = list(
            client[db_name][collection_name]
            .find({}, {"_id": 0})
            .sort("inserted_at", -1)
            .limit(limit)
        )
    except Exception:
        return []
    finally:
        client.close()

    output: list[dict[str, Any]] = []
    primary_metric_name = _primary_metric_name(profile, {})
    for doc in docs:
        experiment_id = str(doc.get("experiment_id") or "")
        record_id = experiment_id or str(doc.get("proposal_name") or "")
        if not record_id or record_id in seen_ids:
            continue
        metrics = dict(doc.get("metrics") or {})
        model = dict(doc.get("model") or {})
        mlflow_run_id = str(doc.get("mlflow_run_id") or model.get("mlflow_run_id") or "")
        mlflow_bundle = _mlflow_bundle(profile, {"mlflow_run_id": mlflow_run_id, **metrics})
        output.append({
            "record_id": record_id,
            "title": str(doc.get("proposal_name") or record_id),
            "created_at": str(doc.get("inserted_at") or ""),
            "experiment_id": experiment_id,
            "proposal_name": str(doc.get("proposal_name") or ""),
            "dataset": str((doc.get("proposal") or {}).get("dataset") or ""),
            "detector": str((doc.get("proposal") or {}).get("detector") or ""),
            "assessment": str(((doc.get("evaluation_summary") or {}).get("per_proposal_analysis") or {}).get(str(doc.get("proposal_name") or ""), {}).get("assessment") or ""),
            "mlflow_run_id": mlflow_run_id,
            "mlflow_ui_url": str(mlflow_bundle.get("ui_url") or ""),
            "root_run_family_id": str(doc.get("root_run_family_id") or ""),
            "root_research_direction": str(doc.get("root_research_direction") or doc.get("research_direction") or ""),
            "source_next_step_title": str(doc.get("source_next_step_title") or ""),
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": metrics.get(primary_metric_name),
            "source": "results_mongo",
        })
    return output


def _raw_result_record(profile: dict[str, Any], record_id: str) -> dict[str, Any] | None:
    client, db_name, collection_name = _results_store(profile)
    try:
        doc = client[db_name][collection_name].find_one(
            {"$or": [{"experiment_id": record_id}, {"proposal_name": record_id}]},
            {"_id": 0},
        )
    except Exception:
        return None
    finally:
        client.close()
    if not doc:
        return None
    return _synthetic_memory_record_from_result(profile, doc)


def _synthetic_memory_record_from_result(profile: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(doc.get("experiment_id") or "")
    proposal_name = str(doc.get("proposal_name") or "unknown")
    metrics = dict(doc.get("metrics") or {})
    evaluation_summary = dict(doc.get("evaluation_summary") or {})
    proposal_analysis = dict((evaluation_summary.get("per_proposal_analysis") or {}).get(proposal_name) or {})
    artifact_refs = list(((doc.get("model") or {}).get("stored_figure_artifacts") or []))
    return {
        "record_id": experiment_id or proposal_name,
        "domain": str(profile.get("name") or ""),
        "kind": "prior_experiment",
        "object_type": "experiment_result",
        "object_key": experiment_id or proposal_name,
        "object_role": "result",
        "schema_version": "1",
        "title": proposal_name,
        "summary": str(proposal_analysis.get("assessment") or ""),
        "content": {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "metrics": metrics,
            "model": doc.get("model") or {},
            "evaluation_summary": evaluation_summary,
            "research_direction": str(doc.get("research_direction") or ""),
            "campaign_id": str(doc.get("campaign_id") or ""),
            "campaign_title": str(doc.get("campaign_title") or ""),
            "campaign_variant_id": str(doc.get("campaign_variant_id") or ""),
            "campaign_variant_title": str(doc.get("campaign_variant_title") or ""),
        },
        "metadata": {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "profile": str(doc.get("profile") or profile.get("name") or ""),
            "research_direction": str(doc.get("research_direction") or ""),
            "root_run_family_id": str(doc.get("root_run_family_id") or ""),
            "root_research_direction": str(doc.get("root_research_direction") or doc.get("research_direction") or ""),
            "source_next_step_title": str(doc.get("source_next_step_title") or ""),
            "campaign_id": str(doc.get("campaign_id") or ""),
            "campaign_title": str(doc.get("campaign_title") or ""),
            "campaign_variant_id": str(doc.get("campaign_variant_id") or ""),
            "campaign_variant_title": str(doc.get("campaign_variant_title") or ""),
            "campaign_variant_index": doc.get("campaign_variant_index"),
            "assessment": str(proposal_analysis.get("assessment") or ""),
            "hypothesis_supported": proposal_analysis.get("hypothesis_supported"),
            "lessons": proposal_analysis.get("lessons") or [],
            "mlflow_run_id": str(doc.get("mlflow_run_id") or ""),
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        },
        "created_at": str(doc.get("inserted_at") or ""),
        "blob_refs": [
            {
                "artifact_id": ref.get("artifact_id"),
                "name": ref.get("name"),
                "uri": ref.get("uri"),
            }
            for ref in artifact_refs
            if isinstance(ref, dict)
        ],
        "entities": [],
        "relations": [],
    }


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
    family_id = str(metadata.get("root_run_family_id") or "")
    source_next_step_record_id = str(metadata.get("source_next_step_record_id") or "")
    source_proposal_seed_record_id = str(metadata.get("source_proposal_seed_record_id") or "")
    filters: list[dict[str, Any]] = []
    if family_id:
        filters.append({"metadata.root_run_family_id": family_id})
    if source_next_step_record_id:
        filters.append({"metadata.source_next_step_record_id": source_next_step_record_id})
    if source_proposal_seed_record_id:
        filters.append({"metadata.source_proposal_seed_record_id": source_proposal_seed_record_id})
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


def _family_bundle(ctx: ProfileContext, run_record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(run_record.get("metadata") or {})
    family_id = str(metadata.get("root_run_family_id") or "")
    root_direction = str(metadata.get("root_research_direction") or metadata.get("research_direction") or "")
    if not family_id:
        return {
            "family_id": "",
            "root_research_direction": root_direction,
            "runs": [],
            "graph": {
                "nodes": [],
                "edges": [],
            },
        }

    records: list[dict[str, Any]] = []
    try:
        records = list(ctx.memory_service.find_records({"metadata.root_run_family_id": family_id}, limit=500))
    except Exception:
        records = []
    records = [record for record in records if str(record.get("object_type") or "") == "experiment_result"]

    if not records:
        records = _raw_family_records(ctx.profile, family_id)

    records.sort(key=lambda item: str(item.get("created_at") or ""))
    return {
        "family_id": family_id,
        "root_research_direction": root_direction,
        "runs": [_family_run_summary(ctx.profile, record) for record in records],
        "graph": _family_graph_bundle(records),
    }


def _raw_family_records(profile: dict[str, Any], family_id: str) -> list[dict[str, Any]]:
    client, db_name, collection_name = _results_store(profile)
    try:
        docs = list(
            client[db_name][collection_name]
            .find({"root_run_family_id": family_id}, {"_id": 0})
            .sort("inserted_at", 1)
        )
    except Exception:
        return []
    finally:
        client.close()
    return [_synthetic_memory_record_from_result(profile, doc) for doc in docs]


def _family_run_summary(profile: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    mlflow_bundle = _mlflow_bundle(profile, metadata)
    return {
        "record_id": str(record.get("record_id") or ""),
        "title": str(record.get("title") or ""),
        "created_at": str(record.get("created_at") or ""),
        "experiment_id": str(metadata.get("experiment_id") or ""),
        "proposal_name": str(metadata.get("proposal_name") or ""),
        "research_direction": str(metadata.get("research_direction") or ""),
        "source_next_step_title": str(metadata.get("source_next_step_title") or ""),
        "campaign_id": str(metadata.get("campaign_id") or ""),
        "campaign_title": str(metadata.get("campaign_title") or ""),
        "campaign_variant_id": str(metadata.get("campaign_variant_id") or ""),
        "campaign_variant_title": str(metadata.get("campaign_variant_title") or ""),
        "campaign_variant_index": metadata.get("campaign_variant_index"),
        "assessment": str(metadata.get("assessment") or ""),
        "mlflow_run_id": str(metadata.get("mlflow_run_id") or ""),
        "mlflow_ui_url": str(mlflow_bundle.get("ui_url") or ""),
        "primary_metric_name": _primary_metric_name(profile, metadata),
        "primary_metric_value": metadata.get(_primary_metric_name(profile, metadata), None),
    }


def _family_graph_bundle(records: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for record in records:
        nodes.extend(list(record.get("entities") or []))
        edges.extend(list(record.get("relations") or []))
    return {
        "nodes": _dedupe_dicts(nodes, ("entity_type", "key")),
        "edges": _dedupe_dicts(edges, ("relation_type", "source_type", "source_key", "target_type", "target_key")),
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


def _run_text_panels(run_record: dict[str, Any], related_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    metadata = dict(run_record.get("metadata") or {})
    panels: list[dict[str, str]] = []

    def add_panel(title: str, body: str) -> None:
        text = str(body or "").strip()
        if not text:
            return
        if any(item["title"] == title for item in panels):
            return
        panels.append({"title": title, "body": text})

    add_panel("Direction", str(metadata.get("research_direction") or ""))

    research_record = _find_related_record(related_records, "research_summary")
    add_panel(
        "Research Summary",
        _record_content_text(research_record, content_keys=["research_summary"]) or str((research_record or {}).get("summary") or ""),
    )

    add_panel("Ideas", _records_text(_related_records_by_kind(related_records, "idea")))

    refined_idea_record = _find_related_record(related_records, "refined_idea")
    add_panel("Refined Idea", _record_text(refined_idea_record))

    proposal_record = _find_related_record(related_records, "proposal")
    proposal_text = _record_text(proposal_record)
    if not proposal_text:
        proposal = metadata.get("proposal") if isinstance(metadata.get("proposal"), dict) else {}
        proposal_text = _mapping_text(
            proposal,
            [
                ("name", "Name"),
                ("description", "Description"),
                ("dataset", "Dataset"),
                ("detector", "Detector"),
                ("hypothesis", "Hypothesis"),
                ("rationale", "Rationale"),
            ],
        )
    add_panel("Proposal", proposal_text)

    add_panel("Run Goal", _run_goal_text(run_record))

    implementation_plan_record = _find_related_record(related_records, "implementation_plan")
    add_panel("Implementation Plan", _record_text(implementation_plan_record))

    implementation_record = _find_related_record(related_records, "implementation")
    validation_record = _find_related_record(related_records, "validation_result")

    validated_code_path = _record_content_value(validation_record, "script_path")
    implementation_code_path = _record_content_value(implementation_record, "script_path")
    validated_code = _read_text_file(validated_code_path)
    implementation_code = _read_text_file(implementation_code_path)

    if validated_code:
        add_panel("Validated Code", validated_code)
    elif implementation_code:
        add_panel("Code", implementation_code)

    validation_test_path = _record_content_value(validation_record, "test_file")
    add_panel("Validation Test", _read_text_file(validation_test_path))
    add_panel("Validation Output", _record_content_value(validation_record, "test_output"))

    evaluation_record = _find_related_record(related_records, "evaluation_summary")
    add_panel("Evaluation Summary", _record_text(evaluation_record))
    add_panel("Proposed Next Steps", _next_steps_text(related_records))
    return panels


def _find_related_record(records: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for record in records:
        if str(record.get("kind") or record.get("object_type") or "") == kind:
            return record
    return None


def _related_records_by_kind(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [
        record for record in records
        if str(record.get("kind") or record.get("object_type") or "") == kind
    ]


def _records_text(records: list[dict[str, Any]]) -> str:
    blocks = []
    for index, record in enumerate(records, 1):
        title = str(record.get("title") or "").strip()
        body = _record_text(record)
        blocks.append(f"{index}. {title}\n{body}".strip() if title else body)
    return "\n\n".join(block for block in blocks if block).strip()


def _record_text(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    content = record.get("content")
    if isinstance(content, dict):
        preferred = _mapping_text(
            content,
            [
                ("research_summary", "Summary"),
                ("description", "Description"),
                ("hypothesis", "Hypothesis"),
                ("rationale", "Rationale"),
                ("dataset", "Dataset"),
                ("detector", "Detector"),
                ("base_class", "Base Class"),
                ("main_method", "Main Method"),
                ("class_name", "Class"),
                ("proposal_name", "Proposal"),
                ("passed", "Passed"),
                ("attempts", "Attempts"),
                ("test_source", "Test Source"),
                ("test_output", "Test Output"),
            ],
        )
        if preferred:
            return preferred
    return str(record.get("summary") or "")


def _record_content_text(record: dict[str, Any] | None, *, content_keys: list[str]) -> str:
    if not record:
        return ""
    content = record.get("content")
    if not isinstance(content, dict):
        return ""
    for key in content_keys:
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _record_content_value(record: dict[str, Any] | None, key: str) -> str:
    if not record:
        return ""
    content = record.get("content")
    if not isinstance(content, dict):
        return ""
    value = content.get(key)
    return str(value or "")


def _next_steps_text(records: list[dict[str, Any]]) -> str:
    next_step_records = _next_step_records(records)
    if not next_step_records:
        return ""
    next_step_records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    blocks: list[str] = []
    for index, record in enumerate(next_step_records, 1):
        content = record.get("content") if isinstance(record.get("content"), dict) else {}
        title = str(record.get("title") or content.get("title") or content.get("suggested_direction") or f"next_step_{index}").strip()
        priority = content.get("priority")
        suggested_direction = str(content.get("suggested_direction") or "").strip()
        rationale = str(content.get("rationale") or "").strip()
        lines = [f"{index}. {title}"]
        if priority not in (None, ""):
            lines.append(f"Priority: {priority}")
        if suggested_direction:
            lines.append(f"Suggested Direction: {suggested_direction}")
        if rationale:
            lines.append(f"Rationale: {rationale}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()


def _next_step_summaries(profile_name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in _next_step_records(records):
        content = record.get("content") if isinstance(record.get("content"), dict) else {}
        metadata = dict(record.get("metadata") or {})
        record_id = str(record.get("record_id") or "")
        title = str(record.get("title") or content.get("title") or content.get("suggested_direction") or "").strip()
        summaries.append({
            "record_id": record_id,
            "title": title,
            "priority": content.get("priority"),
            "suggested_direction": str(content.get("suggested_direction") or "").strip(),
            "rationale": str(content.get("rationale") or "").strip(),
            "created_at": str(record.get("created_at") or ""),
            "research_direction": str(metadata.get("research_direction") or ""),
            "copy_command": build_next_step_cli_command(profile_name, next_step_record_id=record_id) if record_id else "",
        })
    summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return summaries


def _next_step_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record for record in records
        if str(record.get("kind") or record.get("object_type") or "") == "next_step"
    ]


def _mapping_text(value: dict[str, Any], fields: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for key, label in fields:
        item = value.get(key)
        if item is None or item == "":
            continue
        lines.append(f"{label}: {item}")
    return "\n".join(lines).strip()


def _run_goal_text(run_record: dict[str, Any]) -> str:
    metadata = dict(run_record.get("metadata") or {})
    proposal = metadata.get("proposal") if isinstance(metadata.get("proposal"), dict) else {}
    parts = [
        f"Direction: {metadata.get('research_direction')}" if metadata.get("research_direction") else "",
        f"Proposal: {metadata.get('proposal_name')}" if metadata.get("proposal_name") else "",
        f"Dataset: {proposal.get('dataset') or metadata.get('dataset')}" if (proposal.get("dataset") or metadata.get("dataset")) else "",
        f"Detector: {proposal.get('detector') or metadata.get('detector')}" if (proposal.get("detector") or metadata.get("detector")) else "",
        f"What it is testing: {proposal.get('description')}" if proposal.get("description") else "",
        f"Hypothesis: {proposal.get('hypothesis')}" if proposal.get("hypothesis") else "",
        f"Rationale: {proposal.get('rationale')}" if proposal.get("rationale") else "",
        f"Assessment: {metadata.get('assessment')}" if metadata.get("assessment") else "",
        f"Hypothesis supported: {metadata.get('hypothesis_supported')}" if metadata.get("hypothesis_supported") is not None else "",
    ]
    return "\n".join(part for part in parts if part).strip()


def _read_text_file(path_value: str, *, max_chars: int = 50000) -> str:
    path_str = str(path_value or "").strip()
    if not path_str:
        return ""
    try:
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n... [truncated]"


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
