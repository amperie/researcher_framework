"""NS Experiment Runner Node — invokes neuralsignal automation directly.

For each feature proposal this node:
  1. Resolves the dataset (DatasetManager cache hit, or automation create_dataset)
  2. Trains an XGBoost S1 model via S1Trainer, optimising for AUC
  3. Logs results to MLflow (experiment from config.yaml experiment_runner section)
  4. Upserts a text embedding to ChromaDB for semantic experiment comparison
  5. Inserts the full result record into MongoDB (agent_experiments database)

Configuration keys (all read from config.yaml `experiment_runner` section):
  mlflow_experiment, agent_experiments_db, modeling_row_limit, max_evals,
  optimization_metric, default_detector, default_dataset, dataset_row_limit
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import pandas as pd
import pymongo

from configs.config import get_config
from graph.state import ResearchState
from tools.chroma_tool import ChromaStore
from utils.dataset_manager import DatasetManager
from utils.logger import get_logger
from utils.utils import load_yaml_section

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Neuralsignal SDK — optional import so unit tests run without the full SDK
# ---------------------------------------------------------------------------
try:
    from neuralsignal.automation.dataset_automation_core import (  # type: ignore
        create_dataset as ns_create_dataset,
        create_s1_model as ns_create_s1_model,
    )
    NS_AVAILABLE = True
except ImportError:
    NS_AVAILABLE = False
    log.warning(
        "ns_experiment_runner | neuralsignal SDK not importable — "
        "dataset creation and model training will be unavailable"
    )


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _runner_cfg() -> dict:
    return load_yaml_section("experiment_runner") or {}


# ---------------------------------------------------------------------------
# Feature-set config mapping
# ---------------------------------------------------------------------------

def _build_feature_set_config(proposal: dict) -> dict:
    """Map a feature_proposal dict to a neuralsignal feature_set_config dict.

    Uses:
      - proposal['feature_set_name']  → config['name']
      - proposal['zone_config']       → target_zone_size, layer_names_to_include
      - proposal['scan_fields_used']  → field_to_process
    """
    zone_config = proposal.get("zone_config") or {}
    scan_fields = proposal.get("scan_fields_used") or []

    # Determine field_to_process from scan_fields_used, default "outputs"
    field_to_process = "outputs"
    for f in scan_fields:
        if f in ("outputs", "inputs"):
            field_to_process = f
            break

    # Zone size: prefer zone_config["zone_size"], then "default", then 1024
    if isinstance(zone_config, dict):
        zone_size = (
            zone_config.get("zone_size")
            or zone_config.get("default")
            or 1024
        )
        layer_patterns = (
            zone_config.get("layer_names_to_include")
            or zone_config.get("layers")
            or []
        )
    else:
        zone_size = 1024
        layer_patterns = []

    config: dict = {
        "name": proposal.get("feature_set_name") or "zones",
        "target_zone_size": {"default": int(zone_size)},
        "field_to_process": field_to_process,
    }
    if layer_patterns:
        config["layer_names_to_include"] = list(layer_patterns)

    return config


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

def _resolve_datasets(
    proposal: dict,
    dm: DatasetManager,
    cfg,
    rcfg: dict,
) -> tuple[pd.DataFrame | None, list[dict], int]:
    """Return (df, dataset_entries, cache_hit_count).

    Checks DatasetManager first; falls back to neuralsignal automation if needed.
    The returned DataFrame always includes a 'target' column (binary label).
    """
    fs_config = _build_feature_set_config(proposal)
    feature_set_name: str = fs_config["name"]
    detector_name: str = (
        proposal.get("detector_name") or rcfg.get("default_detector", "hallucination")
    )
    dataset_name: str = (
        proposal.get("dataset") or rcfg.get("default_dataset", "HaluBench")
    )

    # Stable cache identity (volatile keys excluded by DatasetManager.config_hash)
    exp_config = {
        "feature_set_name": feature_set_name,
        "detector_name": detector_name,
        "dataset": dataset_name,
        "zone_config": proposal.get("zone_config") or {},
        "field_to_process": fs_config.get("field_to_process", "outputs"),
    }

    # --- Cache hit ---
    cached = dm.find_cached(feature_set_name, exp_config)
    if cached:
        log.info(
            "ns_experiment_runner | Dataset cache hit — feature_set=%r, id=%s",
            feature_set_name, cached["dataset_id"],
        )
        df = dm.load_dataset(cached)
        return df, [cached], 1

    # --- Cache miss: create via automation ---
    if not NS_AVAILABLE:
        log.error(
            "ns_experiment_runner | neuralsignal SDK unavailable and no cached dataset "
            "for feature_set=%r", feature_set_name,
        )
        return None, [], 0

    experiments_dir = Path(cfg.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    automation_cfg = {
        "run_data_collection": False,
        "create_dataset": True,
        "create_s1_model": False,
        "detector_names": [detector_name],
        "feature_set_configs": [fs_config],
        "dataset": dataset_name,
        "application_name": "ns_researcher",
        "sub_application_name": feature_set_name,
        "backend_config": {
            "backend_type": "neuralsignal_v1",
            "mongo_url": cfg.mongo_url,
        },
        "zone_size": fs_config["target_zone_size"].get("default", 1024),
        "row_limit": rcfg.get("dataset_row_limit", 0),
        "write_to_file": True,
        "build_in_memory": False,
        "use_gt_as_target": True,
        "file_out": str(experiments_dir / f"{feature_set_name}_{detector_name}"),
    }

    log.info(
        "ns_experiment_runner | Creating dataset — feature_set=%r, detector=%r, dataset=%r",
        feature_set_name, detector_name, dataset_name,
    )

    try:
        file_paths = ns_create_dataset(automation_cfg, create_dataset=True)
    except Exception as exc:
        log.error("ns_experiment_runner | create_dataset failed: %s", exc)
        return None, [], 0

    if not file_paths:
        log.error("ns_experiment_runner | create_dataset returned no output files")
        return None, [], 0

    try:
        df = pd.read_csv(file_paths[0])
    except Exception as exc:
        log.error("ns_experiment_runner | Failed to read dataset CSV %s: %s", file_paths[0], exc)
        return None, [], 0

    log.info(
        "ns_experiment_runner | Dataset created — shape=%s, file=%s",
        df.shape, file_paths[0],
    )

    # Save feature columns (not target) to DatasetManager for future reuse
    feature_cols = [c for c in df.columns if c != "target"]
    rows = df[feature_cols].values.tolist()
    scan_ids = df.index.astype(str).tolist()
    new_exp_id = str(uuid4())

    try:
        entry = dm.save_dataset(
            new_exp_id, feature_set_name, exp_config, feature_cols, rows, scan_ids,
        )
        log.info("ns_experiment_runner | Dataset saved to registry — id=%s", entry["dataset_id"])
    except Exception as exc:
        log.warning("ns_experiment_runner | DatasetManager save failed: %s", exc)
        entry = {"dataset_id": new_exp_id}

    return df, [entry], 0


# ---------------------------------------------------------------------------
# S1 model training
# ---------------------------------------------------------------------------

def _train_s1_model(
    df: pd.DataFrame,
    proposal: dict,
    experiment_id: str,
    cfg,
    rcfg: dict,
):
    """Train XGBoost S1 model on df, return best S1Model or None on failure."""
    if not NS_AVAILABLE:
        log.error("ns_experiment_runner | neuralsignal SDK unavailable — cannot train")
        return None

    feature_set_name: str = proposal.get("feature_set_name") or "zones"
    detector_name: str = (
        proposal.get("detector_name") or rcfg.get("default_detector", "hallucination")
    )
    proposal_name: str = proposal.get("name") or feature_set_name

    s1_cfg = {
        # Required by S1Trainer.__init__
        "application_name": "ns_researcher",
        "sub_application_name": feature_set_name,
        "model_name": f"{feature_set_name}_{experiment_id[:8]}",
        "dataset_path": "n/a",  # bypassed when 'dataframe' key is present
        # DataFrame shortcut — picked up by create_s1_model (cfg['dataframe'])
        "dataframe": df,
        # Automation iteration keys
        "detector_names": [detector_name],
        "modeling_row_limits": [rcfg.get("modeling_row_limit", 10000)],
        # Training hyperparameters
        "optimization_metric": rcfg.get("optimization_metric", "auc"),
        "max_evals": rcfg.get("max_evals", 20),
        "test_set_size": 0.33,
        "seed": 42,
        "run_cross_validation": True,
        "cv_folds": 3,
        "create_reduced_feature_model": False,
        # Skip backend persistence — we handle it
        "save_to_backend": False,
        "description": proposal.get("description", ""),
        "tags": {
            "proposal_name": proposal_name,
            "experiment_id": experiment_id,
            "source": "ns_researcher",
        },
    }

    feature_cols = [c for c in df.columns if c != "target"]
    log.info(
        "ns_experiment_runner | Training S1 model — proposal=%r, rows=%d, features=%d",
        proposal_name, len(df), len(feature_cols),
    )

    try:
        models = ns_create_s1_model(s1_cfg)
    except Exception as exc:
        log.error(
            "ns_experiment_runner | S1 training failed (proposal=%r): %s", proposal_name, exc
        )
        return None

    if not models:
        log.error("ns_experiment_runner | No models returned for proposal=%r", proposal_name)
        return None

    best = max(models, key=lambda m: m.metrics.get("test_auc", 0.0))
    log.info(
        "ns_experiment_runner | Best model — test_auc=%.4f, train_auc=%.4f",
        best.metrics.get("test_auc", 0.0), best.metrics.get("train_auc", 0.0),
    )
    return best


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def _log_mlflow(
    result: dict,
    state: ResearchState,
    cfg,
    rcfg: dict,
) -> str:
    """Log result to MLflow. Returns run_id or '' on failure."""
    experiment_name: str = rcfg.get("mlflow_experiment", "neuralsignal_agent_experiments")
    proposal = result["proposal"]
    metrics = result["metrics"]
    experiment_id = result["experiment_id"]
    proposal_name = result["proposal_name"]

    try:
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(experiment_name)

        run_name = f"{proposal_name}_{experiment_id[:8]}"
        with mlflow.start_run(run_name=run_name) as run:
            # Parameters
            mlflow.log_params({
                "proposal_name": proposal_name,
                "feature_set_name": proposal.get("feature_set_name", ""),
                "detector_name": result.get("detector_name", ""),
                "dataset": result.get("dataset_name", ""),
                "n_features": result.get("n_features", 0),
                "rows_dataset": metrics.get("rows_dataset", 0),
                "optimization_metric": rcfg.get("optimization_metric", "auc"),
            })

            # Metrics — all numeric values from S1Model.metrics
            numeric_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)

            # Tags
            research_direction = state.get("research_direction", "")
            mlflow.set_tags({
                "research_direction": research_direction[:250],
                "experiment_id": experiment_id,
                "source": "ns_experiment_runner",
                "cache_hits": str(result.get("cache_hits", 0)),
            })

            # Artifacts
            if result.get("feature_importance"):
                mlflow.log_dict(result["feature_importance"], "feature_importance.json")
            if result.get("s1_params"):
                mlflow.log_dict(result["s1_params"], "model_params.json")

            proposal_json = json.dumps(
                {k: v for k, v in proposal.items() if not callable(v)},
                indent=2, default=str,
            )
            mlflow.log_text(proposal_json, "proposal.json")

            run_id = run.info.run_id

        log.info(
            "ns_experiment_runner | MLflow logged — experiment=%r, run_id=%s, test_auc=%.4f",
            experiment_name, run_id, metrics.get("test_auc", 0.0),
        )
        return run_id

    except Exception as exc:
        log.warning("ns_experiment_runner | MLflow logging failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# ChromaDB logging
# ---------------------------------------------------------------------------

def _build_chroma_document(result: dict, state: ResearchState) -> tuple[str, dict]:
    """Return (document_text, metadata) for ChromaDB upsert."""
    proposal = result["proposal"]
    metrics = result["metrics"]
    research_direction = state.get("research_direction", "")

    document = (
        f"Research direction: {research_direction}\n"
        f"Proposal: {proposal.get('name', '')}\n"
        f"Description: {proposal.get('description', '')}\n"
        f"Feature set: {proposal.get('feature_set_name', '')}\n"
        f"Hypothesis: {proposal.get('hypothesis', '')}\n"
        f"Target behavior: {proposal.get('target_behavior', '')}\n"
        f"Metrics: test_auc={metrics.get('test_auc', 0.0):.4f}, "
        f"test_f1={metrics.get('test_f1', 0.0):.4f}, "
        f"test_accuracy={metrics.get('test_accuracy', 0.0):.4f}"
    )

    metadata = {
        "experiment_id": result["experiment_id"],
        "proposal_name": result["proposal_name"],
        "feature_set_name": proposal.get("feature_set_name", ""),
        "test_auc": float(metrics.get("test_auc", 0.0)),
        "train_auc": float(metrics.get("train_auc", 0.0)),
        "test_f1": float(metrics.get("test_f1", 0.0)),
        "test_accuracy": float(metrics.get("test_accuracy", 0.0)),
        "mlflow_run_id": result.get("mlflow_run_id", ""),
        "detector_name": result.get("detector_name", ""),
        "dataset": result.get("dataset_name", ""),
        "inserted_at": result.get("inserted_at", ""),
    }
    return document, metadata


def _log_chromadb(result: dict, state: ResearchState, rcfg: dict) -> str:
    """Upsert result into ChromaDB. Returns experiment_id (used as record ID)."""
    experiment_id = result["experiment_id"]
    try:
        collection_name = rcfg.get("chroma_collection", "agent_experiments")
        store = ChromaStore(collection_name=collection_name)
        document, metadata = _build_chroma_document(result, state)
        store.upsert(experiment_id, document, metadata)
        log.info("ns_experiment_runner | ChromaDB upsert complete — id=%s", experiment_id)
    except Exception as exc:
        log.warning("ns_experiment_runner | ChromaDB upsert failed: %s", exc)
    return experiment_id


# ---------------------------------------------------------------------------
# MongoDB logging
# ---------------------------------------------------------------------------

def _log_mongo(result: dict, cfg, rcfg: dict) -> None:
    """Insert full result into MongoDB agent_experiments.experiments. Never raises."""
    db_name: str = rcfg.get("agent_experiments_db", "agent_experiments")
    try:
        client = pymongo.MongoClient(cfg.mongo_url)
        db = client[db_name]
        # Strip any non-serialisable runtime objects before inserting
        doc = {k: v for k, v in result.items() if k != "proposal" or isinstance(v, dict)}
        db["experiments"].insert_one(doc)
        client.close()
        log.info(
            "ns_experiment_runner | MongoDB logged — db=%r, experiment_id=%s",
            db_name, result["experiment_id"],
        )
    except Exception as exc:
        log.warning("ns_experiment_runner | MongoDB logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def ns_experiment_runner_node(state: ResearchState) -> dict:
    """Run neuralsignal automation experiments for every feature proposal.

    Reads:
        state['feature_proposals']   — proposals from feature_proposal_node
        state['research_direction']  — for MLflow tags and ChromaDB embedding

    Writes:
        state['ns_experiment_results']    — one result dict per successful proposal
        state['ns_best_auc']              — highest test_auc across proposals
        state['ns_mlflow_run_ids']        — MLflow run IDs (one per proposal)
        state['ns_chroma_record_ids']     — ChromaDB record IDs (one per proposal)
        state['execution_success']        — True if ≥1 proposal produced results
        state['dataset_ids']              — all dataset UUIDs touched in this run
        state['dataset_cache_hits']       — total DatasetManager cache hits
    """
    cfg = get_config()
    rcfg = _runner_cfg()

    proposals = state.get("feature_proposals") or []
    if not proposals:
        log.error("ns_experiment_runner_node | No feature_proposals in state")
        return {"execution_success": False, "errors": ["No feature_proposals in state"]}

    dm = DatasetManager(cfg.mongo_url, cfg.datasets_db_name)

    all_results: list[dict] = []
    all_dataset_ids: list[str] = []
    total_cache_hits = 0
    mlflow_run_ids: list[str] = []
    chroma_ids: list[str] = []

    log.info("ns_experiment_runner_node | Processing %d proposal(s)", len(proposals))

    for idx, proposal in enumerate(proposals):
        proposal_name: str = proposal.get("name") or proposal.get("feature_set_name", f"proposal_{idx}")
        log.info(
            "ns_experiment_runner_node | Proposal %d/%d — %r",
            idx + 1, len(proposals), proposal_name,
        )

        experiment_id = str(uuid4())

        # 1. Resolve dataset (cache or create via automation)
        df, entries, cache_hits = _resolve_datasets(proposal, dm, cfg, rcfg)
        total_cache_hits += cache_hits
        all_dataset_ids.extend(e.get("dataset_id", "") for e in entries)

        if df is None or df.empty:
            log.warning(
                "ns_experiment_runner_node | No dataset for proposal=%r — skipping",
                proposal_name,
            )
            continue

        # 2. Train S1 model (XGBoost, AUC-optimised)
        s1_model = _train_s1_model(df, proposal, experiment_id, cfg, rcfg)
        if s1_model is None:
            log.warning(
                "ns_experiment_runner_node | Training failed for proposal=%r — skipping",
                proposal_name,
            )
            continue

        # 3. Assemble result record
        feature_cols = [c for c in df.columns if c != "target"]
        detector_name: str = (
            proposal.get("detector_name") or rcfg.get("default_detector", "hallucination")
        )
        dataset_name: str = (
            proposal.get("dataset") or rcfg.get("default_dataset", "HaluBench")
        )

        result: dict = {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "proposal": {k: v for k, v in proposal.items() if not callable(v)},
            "metrics": dict(s1_model.metrics),
            "s1_params": dict(s1_model.params) if s1_model.params else {},
            "feature_importance": (
                s1_model.artifacts.get("feature_importance", {})
                if s1_model.artifacts else {}
            ),
            "dataset_ids": [e.get("dataset_id", "") for e in entries],
            "cache_hits": cache_hits,
            "n_features": len(feature_cols),
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            "inserted_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": "",
            "chroma_record_id": "",
        }

        # 4. Log to MLflow
        run_id = _log_mlflow(result, state, cfg, rcfg)
        result["mlflow_run_id"] = run_id
        if run_id:
            mlflow_run_ids.append(run_id)

        # 5. Log to ChromaDB
        chroma_id = _log_chromadb(result, state, rcfg)
        result["chroma_record_id"] = chroma_id
        chroma_ids.append(chroma_id)

        # 6. Log to MongoDB
        _log_mongo(result, cfg, rcfg)

        all_results.append(result)
        log.info(
            "ns_experiment_runner_node | Proposal %r done — test_auc=%.4f, mlflow=%s",
            proposal_name, result["metrics"].get("test_auc", 0.0), run_id,
        )

    # Summary
    n_ok = len(all_results)
    best_auc = max(
        (r["metrics"].get("test_auc", 0.0) for r in all_results),
        default=0.0,
    )
    log.info(
        "ns_experiment_runner_node | Complete — %d/%d proposals succeeded, best_auc=%.4f",
        n_ok, len(proposals), best_auc,
    )

    return {
        "ns_experiment_results": all_results,
        "ns_best_auc": best_auc,
        "ns_mlflow_run_ids": mlflow_run_ids,
        "ns_chroma_record_ids": chroma_ids,
        "execution_success": n_ok > 0,
        "dataset_ids": [did for did in all_dataset_ids if did],
        "dataset_cache_hits": total_cache_hits,
    }
