from __future__ import annotations

from typing import Any

import pymongo

from configs.config import get_config
from core.memory import MemoryService

KNOWN_ARTIFACT_EXCLUSIONS = {
    "experiment_result",
    "evaluation_summary",
    "idea",
    "implementation",
    "implementation_plan",
    "next_step",
    "proposal",
    "proposal_seed",
    "refined_idea",
    "research_artifact",
    "research_summary",
    "run_handoff",
    "validation_result",
}

START_NODE_REQUIREMENTS: dict[str, list[str]] = {
    "ideate": ["research_summary"],
    "refine": ["ideas"],
    "propose_experiments": ["refined_ideas"],
    "plan_implementation": ["proposals"],
    "implement": ["implementation_plans"],
    "validate": ["implementations"],
    "prepare_experiment": ["proposals"],
    "create_dataset": ["proposals"],
    "execute_experiment": ["proposals"],
    "submit_experiment_jobs": ["proposals"],
    "run_experiment": ["proposals"],
    "check_experiment_jobs": ["experiment_jobs"],
    "create_model": ["experiment_results"],
    "evaluate": ["experiment_results"],
    "store_results": ["experiment_results"],
    "propose_next_steps": ["experiment_results"],
    "rank_next_steps": ["next_steps"],
}


def build_resume_state(profile: dict[str, Any], source_record_id: str) -> dict[str, Any]:
    service = MemoryService.for_profile(profile)
    source_record = service.document_store.get(source_record_id)
    if not source_record:
        source_record = _raw_result_record(profile, source_record_id)
    if not source_record:
        raise ValueError(f"Resume source record not found: {source_record_id}")

    records = _resume_records(service, source_record)
    metadata = dict(source_record.get("metadata") or {})
    content = source_record.get("content") if isinstance(source_record.get("content"), dict) else {}

    state: dict[str, Any] = {
        "research_direction": str(
            metadata.get("research_direction")
            or content.get("research_direction")
            or ""
        ).strip(),
        "root_run_family_id": str(metadata.get("root_run_family_id") or content.get("root_run_family_id") or ""),
        "root_research_direction": str(
            metadata.get("root_research_direction")
            or content.get("root_research_direction")
            or metadata.get("research_direction")
            or content.get("research_direction")
            or ""
        ).strip(),
        "errors": [],
    }
    for key in (
        "source_next_step_record_id",
        "source_next_step_title",
        "source_proposal_seed_record_id",
        "source_proposal_seed_title",
    ):
        value = metadata.get(key) or content.get(key) or ""
        if value:
            state[key] = str(value)

    latest_research = _latest_record(records, "research_summary")
    if latest_research:
        research_content = latest_research.get("content") if isinstance(latest_research.get("content"), dict) else {}
        state["research_summary"] = str(research_content.get("research_summary") or latest_research.get("summary") or "")
        state["research_artifacts"] = list(research_content.get("research_artifacts") or [])
        state["research_papers"] = list(research_content.get("research_papers") or [])
        state["paper_digests"] = list(research_content.get("paper_digests") or [])

    state["ideas"] = _contents_by_type(records, "idea")
    state["refined_ideas"] = _contents_by_type(records, "refined_idea")
    state["proposals"] = _contents_by_type(records, "proposal")
    if not state["proposals"] and isinstance(content.get("proposal"), dict):
        state["proposals"] = [dict(content.get("proposal") or {})]
    state["implementation_plans"] = _contents_by_type(records, "implementation_plan")
    state["implementations"] = _contents_by_type(records, "implementation")
    state["validation_results"] = _contents_by_type(records, "validation_result")
    state["next_steps"] = _contents_by_type(records, "next_step")
    if "proposal_seed_planning_notes" in content and content.get("proposal_seed_planning_notes"):
        state["proposal_seed_planning_notes"] = str(content.get("proposal_seed_planning_notes") or "").strip()

    experiment_artifacts = _experiment_artifact_contents(records)
    if experiment_artifacts:
        state["experiment_artifacts"] = experiment_artifacts
        state["datasets"] = list(experiment_artifacts)

    experiment_results = _experiment_results_from_records(records)
    if experiment_results:
        state["experiment_results"] = experiment_results
        models = [result.get("model") for result in experiment_results if isinstance(result.get("model"), dict) and result.get("model")]
        if models:
            state["models"] = models
    elif str(source_record.get("object_type") or "") == "experiment_result":
        result = _experiment_result_from_record(source_record)
        if result:
            state["experiment_results"] = [result]
            if isinstance(result.get("model"), dict) and result.get("model"):
                state["models"] = [result["model"]]

    latest_evaluation = _latest_record(records, "evaluation_summary")
    if latest_evaluation:
        evaluation_content = latest_evaluation.get("content") if isinstance(latest_evaluation.get("content"), dict) else {}
        if evaluation_content:
            state["evaluation_summary"] = dict(evaluation_content)
    elif isinstance(content.get("evaluation_summary"), dict):
        state["evaluation_summary"] = dict(content.get("evaluation_summary") or {})

    return {key: value for key, value in state.items() if value not in (None, "", [], {}) or key == "errors"}


def ensure_resume_state_for_node(start_node: str, state: dict[str, Any]) -> None:
    required = START_NODE_REQUIREMENTS.get(start_node) or []
    missing = [key for key in required if not state.get(key)]
    if missing:
        raise ValueError(
            f"Cannot resume at node {start_node!r}; missing required state keys: {', '.join(missing)}"
        )


def _resume_records(service: MemoryService, source_record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = dict(source_record.get("metadata") or {})
    proposal_name = str(metadata.get("proposal_name") or "")
    research_direction = str(metadata.get("research_direction") or "")
    filters: list[dict[str, Any]] = []
    if proposal_name:
        filters.append({"metadata.proposal_name": proposal_name})
    if research_direction:
        filters.append({"metadata.research_direction": research_direction})

    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in [source_record]:
        record_id = str(record.get("record_id") or "")
        if record_id and record_id not in seen:
            seen.add(record_id)
            output.append(record)
    for record_filter in filters:
        for record in service.find_records(record_filter, limit=500):
            record_id = str(record.get("record_id") or "")
            if not record_id or record_id in seen:
                continue
            seen.add(record_id)
            output.append(record)
    output.sort(key=lambda item: str(item.get("created_at") or ""))
    return output


def _contents_by_type(records: list[dict[str, Any]], object_type: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("object_type") or record.get("kind") or "") != object_type:
            continue
        content = record.get("content")
        if isinstance(content, dict) and content:
            items.append(dict(content))
    return items


def _experiment_artifact_contents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        object_type = str(record.get("object_type") or record.get("kind") or "")
        object_role = str(record.get("object_role") or "")
        if object_role != "artifact" or object_type in KNOWN_ARTIFACT_EXCLUSIONS:
            continue
        content = record.get("content")
        if isinstance(content, dict) and content:
            items.append(dict(content))
    return items


def _experiment_results_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("object_type") or "") != "experiment_result":
            continue
        normalized = _experiment_result_from_record(record)
        if normalized:
            results.append(normalized)
    return results


def _experiment_result_from_record(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content") if isinstance(record.get("content"), dict) else {}
    metadata = dict(record.get("metadata") or {})
    experiment_id = str(content.get("experiment_id") or metadata.get("experiment_id") or record.get("record_id") or "")
    proposal_name = str(content.get("proposal_name") or metadata.get("proposal_name") or record.get("title") or "")
    proposal = content.get("proposal") if isinstance(content.get("proposal"), dict) else {}
    metrics = content.get("metrics") if isinstance(content.get("metrics"), dict) else {}
    model = content.get("model") if isinstance(content.get("model"), dict) else {}
    if not experiment_id and not proposal_name:
        return {}
    return {
        "experiment_id": experiment_id,
        "proposal_name": proposal_name,
        "proposal": proposal,
        "metrics": metrics,
        "model": model,
        "evaluation_summary": content.get("evaluation_summary") if isinstance(content.get("evaluation_summary"), dict) else {},
        "research_direction": str(content.get("research_direction") or metadata.get("research_direction") or ""),
        "mlflow_run_id": str(metadata.get("mlflow_run_id") or model.get("mlflow_run_id") or ""),
        "artifacts": content.get("artifacts") if isinstance(content.get("artifacts"), dict) else {},
        "execution_config": content.get("execution_config") if isinstance(content.get("execution_config"), dict) else {},
        "variant_results": content.get("variant_results") if isinstance(content.get("variant_results"), list) else [],
        "report": str(content.get("report") or ""),
    }


def _latest_record(records: list[dict[str, Any]], object_type: str) -> dict[str, Any] | None:
    matches = [
        record for record in records
        if str(record.get("object_type") or record.get("kind") or "") == object_type
    ]
    return matches[-1] if matches else None


def _raw_result_record(profile: dict[str, Any], record_id: str) -> dict[str, Any] | None:
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    client = pymongo.MongoClient(cfg.mongo_url)
    db_name = storage_cfg.get("mongodb_results_db", "researcher_results")
    collection_name = storage_cfg.get("mongodb_results_collection", "experiments")
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
    model = dict(doc.get("model") or {})
    return {
        "record_id": experiment_id or proposal_name,
        "domain": str(profile.get("name") or ""),
        "kind": "prior_experiment",
        "object_type": "experiment_result",
        "object_key": experiment_id or proposal_name,
        "object_role": "result",
        "schema_version": "1",
        "title": proposal_name,
        "summary": "",
        "content": {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "proposal": dict(doc.get("proposal") or {}),
            "metrics": metrics,
            "model": model,
            "evaluation_summary": evaluation_summary,
            "research_direction": str(doc.get("research_direction") or ""),
            "root_run_family_id": str(doc.get("root_run_family_id") or ""),
            "root_research_direction": str(doc.get("root_research_direction") or doc.get("research_direction") or ""),
            "artifacts": dict(doc.get("artifacts") or {}),
            "execution_config": dict(doc.get("execution_config") or {}),
            "variant_results": list(doc.get("variant_results") or []),
            "report": str(doc.get("report") or ""),
        },
        "metadata": {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "profile": str(profile.get("name") or ""),
            "research_direction": str(doc.get("research_direction") or ""),
            "root_run_family_id": str(doc.get("root_run_family_id") or ""),
            "root_research_direction": str(doc.get("root_research_direction") or doc.get("research_direction") or ""),
            "mlflow_run_id": str(doc.get("mlflow_run_id") or model.get("mlflow_run_id") or ""),
        },
        "created_at": str(doc.get("inserted_at") or ""),
    }
