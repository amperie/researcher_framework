"""MLflow logger node — log the experiment to the MLflow tracking server."""
from __future__ import annotations

from graph.state import ResearchState
from utils.logger import get_logger

log = get_logger(__name__)


def mlflow_logger_node(state: ResearchState) -> dict:
    """Log the experiment run to MLflow.

    Reads:
        state['experiment_id']
        state['experiment_config']
        state['raw_results']
        state['feature_proposals']
        state['arxiv_papers']
        state['research_summary']
        state['generated_code']

    Writes:
        state['mlflow_run_id']
        state['mlflow_experiment_name']

    MLflow logging pattern (mirrors neuralsignal.backend.backend_util.save_to_mlflow):
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name=state['experiment_id']) as run:
            mlflow.log_params(experiment_config)
            mlflow.log_metrics({k: v for k, v in raw_results.items()
                                 if isinstance(v, (int, float))})
            mlflow.set_tags({...})
            mlflow.log_text(generated_code, 'experiment_script.py')
            mlflow.log_text(research_summary, 'research_summary.md')
            mlflow.log_dict(arxiv_papers, 'arxiv_papers.json')

    TODO: implement the above pattern.
    TODO: filter raw_results to only log numeric values as metrics.
    """
    exp_id = state.get("experiment_id", "")
    raw = state.get("raw_results") or {}
    numeric_keys = [k for k, v in raw.items() if isinstance(v, (int, float))]
    log.info("mlflow_logger_node | Starting MLflow run — experiment_id=%r", exp_id)
    log.debug("mlflow_logger_node | Metrics to log: %s", numeric_keys)
    log.debug("mlflow_logger_node | Params to log: %s",
              list((state.get("experiment_config") or {}).keys()))

    # TODO: mlflow.set_tracking_uri(cfg.mlflow_uri)
    # log.debug("mlflow_logger_node | Tracking URI: %r", cfg.mlflow_uri)

    # TODO: start run and log everything
    # log.info("mlflow_logger_node | Run created — run_id=%r", run.info.run_id)
    # log.debug("mlflow_logger_node | Logged %d metrics, %d params, %d artifacts",
    #           len(numeric_keys), len(experiment_config), 3)

    raise NotImplementedError("mlflow_logger_node is not yet implemented")
