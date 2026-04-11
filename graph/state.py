"""ResearchState — the shared state flowing through the LangGraph pipeline.

Every node receives the full state and returns a dict of fields to update.
Fields not returned by a node are left unchanged.
"""
from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------
    research_direction: str
    """The research direction supplied by the user at the start of the session
    (or promoted from a follow-up proposal in subsequent loop iterations)."""

    # -------------------------------------------------------------------------
    # Research stage  (research_node)
    # -------------------------------------------------------------------------
    arxiv_papers: list[dict]
    """Arxiv papers retrieved and scored for relevance.
    Each dict: {title, abstract, url, arxiv_id, published, relevance_score}."""

    research_summary: str
    """LLM-synthesised summary of the key themes and techniques found in the papers."""

    # -------------------------------------------------------------------------
    # Feature proposal stage  (feature_proposal_node)
    # -------------------------------------------------------------------------
    feature_proposals: list[dict]
    """Concrete experiment proposals grounded in neuralsignal classes.
    Each dict: {name, description, neuralsignal_classes, feature_sets,
                zone_config, dataset, rationale}."""

    # -------------------------------------------------------------------------
    # Code generation stage  (code_generation_node)
    # -------------------------------------------------------------------------
    generated_code: str
    """Complete Python script that runs the experiment and prints JSON results."""

    experiment_config: dict
    """Structured parameters describing the experiment (mirrors what the code uses)."""

    # -------------------------------------------------------------------------
    # Execution stage  (experiment_runner_node)
    # -------------------------------------------------------------------------
    experiment_id: str
    """UUID assigned to this experiment run (also used as script filename and MLflow run name)."""

    execution_stdout: str
    """Full stdout captured from the experiment subprocess."""

    execution_stderr: str
    """Full stderr captured from the experiment subprocess."""

    execution_success: bool
    """True if the subprocess exited with code 0 and stdout contained valid JSON."""

    raw_results: dict
    """Parsed JSON results emitted by the experiment script (metrics, artefact paths, etc.)."""

    # -------------------------------------------------------------------------
    # Result analysis stage  (result_analysis_node)
    # -------------------------------------------------------------------------
    similar_experiments: list[dict]
    """Past experiments retrieved from ChromaDB by semantic similarity."""

    analysis_summary: str
    """LLM analysis comparing current results against similar past experiments."""

    # -------------------------------------------------------------------------
    # MLflow logging stage  (mlflow_logger_node)
    # -------------------------------------------------------------------------
    mlflow_run_id: str
    """MLflow run ID for the primary experiment run."""

    mlflow_experiment_name: str
    """Name of the MLflow experiment under which the run was logged."""

    # -------------------------------------------------------------------------
    # Generalization evaluation stage  (generalization_eval_node)
    # -------------------------------------------------------------------------
    generalization_results: dict
    """Metrics from the generalization evaluation run."""

    mlflow_generalization_run_id: str
    """MLflow run ID for the generalization evaluation (child run of mlflow_run_id)."""

    # -------------------------------------------------------------------------
    # Database logging stage  (db_logger_node)
    # -------------------------------------------------------------------------
    chroma_record_id: str
    """ChromaDB document ID for the persisted experiment record."""

    # -------------------------------------------------------------------------
    # Follow-up proposal stage  (followup_proposal_node)
    # -------------------------------------------------------------------------
    followup_proposals: list[dict]
    """Proposed follow-up experiments.
    Each dict: {title, rationale, suggested_direction, priority}."""

    # -------------------------------------------------------------------------
    # Control / loop
    # -------------------------------------------------------------------------
    continue_loop: bool
    """When True, the graph loops back to feature_proposal_node using the top
    follow-up proposal as the new research direction instead of terminating."""

    errors: list[str]
    """Accumulated non-fatal error messages from any node."""
