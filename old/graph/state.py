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

    paper_digests: list[dict]
    """Structured digests extracted from the full text of the top-scored papers.
    Each dict: {arxiv_id, title, published, abstract, digest}
    where digest is an LLM-produced structured extraction covering methods,
    findings, applicable techniques, and open problems (~400–700 words).
    Cached to dev/papers/<arxiv_id>.digest to avoid redundant LLM calls."""

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
    generated_scripts: list[dict]
    """One generated script per feature proposal.
    Each dict: {experiment_id, code, experiment_config, proposal_name, script_path}."""

    # Legacy single-value fields — kept for downstream compat, populated from
    # generated_scripts[0] (first entry) by code_generation_node.
    generated_code: str
    """Alias for generated_scripts[0]['code']. Kept for backward compat."""

    experiment_config: dict
    """Alias for generated_scripts[0]['experiment_config']. Kept for backward compat."""

    # -------------------------------------------------------------------------
    # Execution stage  (experiment_runner_node)
    # -------------------------------------------------------------------------
    experiment_results: list[dict]
    """One result per generated script.
    Each dict: {experiment_id, proposal_name, stdout, stderr, success, raw_results}."""

    # Legacy single-value fields — populated from the first successful result
    # (or first overall) by experiment_runner_node for downstream compat.
    experiment_id: str
    """UUID of the best/first successful experiment. Kept for backward compat."""

    execution_stdout: str
    """stdout from the best/first successful experiment. Kept for backward compat."""

    execution_stderr: str
    """stderr from the best/first successful experiment. Kept for backward compat."""

    execution_success: bool
    """True if at least one experiment script succeeded."""

    raw_results: dict
    """Parsed JSON results from the best/first successful experiment."""

    # -------------------------------------------------------------------------
    # Dataset registry stage  (experiment_runner_node)
    # -------------------------------------------------------------------------
    dataset_ids: list[str]
    """Dataset IDs registered or reused in this pipeline run."""

    dataset_cache_hits: int
    """Number of experiments skipped because a cached dataset was found."""

    # -------------------------------------------------------------------------
    # NS Experiment runner stage  (ns_experiment_runner_node)
    # -------------------------------------------------------------------------
    ns_experiment_results: list[dict]
    """One result dict per proposal run through ns_experiment_runner_node.
    Each: {experiment_id, proposal_name, proposal, metrics, s1_params,
           feature_importance, dataset_ids, cache_hits, n_features,
           detector_name, dataset_name, inserted_at, mlflow_run_id, chroma_record_id}"""

    ns_best_auc: float
    """Highest test_auc across all proposals in this pipeline run."""

    ns_mlflow_run_ids: list[str]
    """MLflow run IDs from ns_experiment_runner_node (one per proposal)."""

    ns_chroma_record_ids: list[str]
    """ChromaDB record IDs from ns_experiment_runner_node (one per proposal)."""

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
