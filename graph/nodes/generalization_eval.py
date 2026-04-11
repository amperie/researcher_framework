"""Generalization evaluation node — test experiment on an alternative dataset or model."""
from __future__ import annotations

from graph.state import ResearchState
from logger import get_logger

log = get_logger(__name__)


def generalization_eval_node(state: ResearchState) -> dict:
    """Run a generalization evaluation and log it as a child MLflow run.

    Reads:
        state['generated_code']
        state['experiment_config']
        state['mlflow_run_id']       — parent run for nesting

    Writes:
        state['generalization_results']          — metrics from generalization run
        state['mlflow_generalization_run_id']    — MLflow child run ID

    Steps:
        1. Ask LLM to produce a modified experiment_config targeting an alternative
           dataset (e.g. swap HaluBench → TruthfulQA) or probe model variant.
        2. Regenerate or patch the experiment script for the new config.
        3. Run via subprocess (same pattern as experiment_runner_node).
        4. Open MLflow child run nested under state['mlflow_run_id']:
               with mlflow.start_run(run_id=parent_id):
                   with mlflow.start_run(nested=True,
                                         tags={'eval_type': 'generalization'}):
                       mlflow.log_metrics(generalization_results)
        5. Store child run ID and metrics in state.

    TODO: implement steps 1–5.
    TODO: gracefully handle subprocess failure — set generalization_results to
          {'error': ...} and continue (do not abort the pipeline).
    """
    parent_run = state.get("mlflow_run_id", "")
    base_config = state.get("experiment_config") or {}
    log.info("generalization_eval_node | Starting generalization eval — parent_run=%r", parent_run)
    log.debug("generalization_eval_node | Base config keys: %s", list(base_config.keys()))

    # TODO: ask LLM for alt config
    # log.info("generalization_eval_node | Alt config — dataset=%r, model=%r",
    #          alt_config.get("dataset"), alt_config.get("model"))

    # TODO: run subprocess
    # log.info("generalization_eval_node | Generalization subprocess started")
    # if not success:
    #     log.warning("generalization_eval_node | Subprocess failed — recording error and continuing")
    #     return {"generalization_results": {"error": stderr[-200:]}, ...}

    # TODO: log to MLflow child run
    # log.info("generalization_eval_node | Child run created — run_id=%r", child_run_id)
    # log.debug("generalization_eval_node | Generalization metrics: %s", gen_results)

    raise NotImplementedError("generalization_eval_node is not yet implemented")
