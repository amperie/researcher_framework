"""Store results step - persist experiment records plus canonical memory.

Reads:
    state['experiment_results']
    state['models']
    state['evaluation_summary']
    state['proposals']
    state['research_direction']

Writes:
    state['stored_result_ids']
"""
from __future__ import annotations

from datetime import datetime, timezone

import mlflow
import pymongo

from configs.config import get_config
from core.graph.state import ResearchState
from core.graph.nodes.memory import build_memory_records_for_state
from core.memory import MemoryService
from core.utils.logger import get_logger

log = get_logger(__name__)


def store_results_node(state: ResearchState, profile: dict) -> dict:
    experiment_results = state.get("experiment_results") or []
    models = state.get("models") or []
    evaluation_summary = state.get("evaluation_summary") or {}
    direction = state.get("research_direction", "")
    root_run_family_id = str(state.get("root_run_family_id") or "")
    root_research_direction = str(state.get("root_research_direction") or direction or "")

    if not experiment_results:
        log.warning("store_results_node | Nothing to store")
        return {"stored_result_ids": []}

    storage_cfg = profile.get("storage") or {}
    mlflow_experiment = storage_cfg.get("mlflow_experiment", "researcher_experiments")
    mongo_db = storage_cfg.get("mongodb_results_db", "researcher_results")
    mongo_collection = storage_cfg.get("mongodb_results_collection", "experiments")
    cfg = get_config()
    stored_ids: list[str] = []
    errors = list(state.get("errors") or [])

    model_by_exp = {m.get("experiment_id"): m for m in models}

    for result in experiment_results:
        experiment_id = result.get("experiment_id", "")
        proposal_name = result.get("proposal_name", "unknown")
        metrics = result.get("metrics") or {}

        inserted_at = datetime.now(timezone.utc).isoformat()

        # --- MLflow ---
        mlflow_run_id = result.get("mlflow_run_id") or ""
        if mlflow_run_id:
            log.info("store_results_node | Reusing existing MLflow run - %s", mlflow_run_id)
        else:
            try:
                mlflow.set_tracking_uri(cfg.mlflow_uri)
                mlflow.set_experiment(mlflow_experiment)
                with mlflow.start_run(run_name=f"{proposal_name}_{experiment_id[:8]}") as run:
                    mlflow.log_params({
                        "proposal_name": proposal_name,
                        "research_direction": direction[:250],
                    })
                    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    if numeric:
                        mlflow.log_metrics(numeric)
                    model = model_by_exp.get(experiment_id)
                    if model and model.get("metrics"):
                        mlflow.log_metrics(
                            {f"model_{k}": v for k, v in model["metrics"].items() if isinstance(v, (int, float))}
                        )
                    mlflow.set_tags({
                        "experiment_id": experiment_id,
                        "profile": profile.get("name", ""),
                        "source": "researcher_pipeline",
                    })
                    mlflow_run_id = run.info.run_id
                log.info("store_results_node | MLflow run logged - %s", mlflow_run_id)
            except Exception as exc:
                log.warning("store_results_node | MLflow failed: %s", exc)
                errors.append(f"store_results: MLflow failed for {proposal_name}: {exc}")

        # --- MongoDB ---
        try:
            client = pymongo.MongoClient(cfg.mongo_url)
            doc = {
                "experiment_id": experiment_id,
                "proposal_name": proposal_name,
                "profile": profile.get("name", ""),
                "research_direction": direction,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "metrics": metrics,
                "model": model_by_exp.get(experiment_id),
                "evaluation_summary": evaluation_summary,
                "mlflow_run_id": mlflow_run_id,
                "inserted_at": inserted_at,
            }
            client[mongo_db][mongo_collection].insert_one(doc)
            client.close()
            log.info("store_results_node | MongoDB insert - %s", experiment_id)
        except Exception as exc:
            log.warning("store_results_node | MongoDB failed: %s", exc)
            errors.append(f"store_results: MongoDB failed for {proposal_name}: {exc}")

        stored_ids.append(experiment_id)

    try:
        memory_service = MemoryService.for_profile(profile)
        records = build_memory_records_for_state(profile, state)
        memory_service.persist_records(records)
    except Exception as exc:
        log.warning("store_results_node | Memory persistence failed: %s", exc)
        errors.append(f"store_results: memory persistence failed: {exc}")

    return {"stored_result_ids": stored_ids, "errors": errors}
