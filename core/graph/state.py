"""ResearchState — the shared state flowing through the LangGraph pipeline.

Every node receives the full state and returns a partial dict of keys to update.
Fields not returned by a node are left unchanged.

All keys are domain-generic. No neuralsignal-specific or trading-specific fields
belong here — those details live in the profile YAML and plugin adapters.
"""
from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    # -------------------------------------------------------------------------
    # Input / control
    # -------------------------------------------------------------------------
    profile_name: str
    """Name of the active research profile (e.g. 'neuralsignal', 'trading')."""

    research_direction: str
    """The research question supplied by the user (or promoted from next_steps)."""

    continue_loop: bool
    """When True, promote the top next_step as the new direction and loop."""

    source_next_step_record_id: str
    """Optional lineage pointer to the next_step record that initiated this run."""

    source_next_step_title: str
    """Human-readable next_step title carried forward for graph lineage."""

    source_proposal_seed_record_id: str
    """Optional lineage pointer to a saved proposal-stage seed that initiated this run."""

    source_proposal_seed_title: str
    """Human-readable proposal-seed title carried forward for lineage and UI."""

    root_run_family_id: str
    """Stable lineage id shared by the initial direction and all descendant runs."""

    root_research_direction: str
    """The initial research direction that seeded the current run family."""

    # -------------------------------------------------------------------------
    # research step
    # -------------------------------------------------------------------------
    research_papers: list[dict]
    """Papers retrieved and scored for relevance.
    Each dict: {title, abstract, url, arxiv_id, published, relevance_score}."""

    research_artifacts: list[dict]
    """All selected, scored research inputs from profile-configured tools.
    Examples: papers, prior experiments, platform inventories, profile context."""

    research_summary: str
    """LLM-synthesised summary of key themes and findings."""

    paper_digests: list[dict]
    """Structured digests from full-text of top-scored papers.
    Each dict: {arxiv_id, title, published, abstract, digest}.
    Cached under the configured dev root at papers/<arxiv_id>.digest."""

    # -------------------------------------------------------------------------
    # ideate step
    # -------------------------------------------------------------------------
    ideas: list[dict]
    """Raw brainstorm from ideate step.
    Each dict: {name, description, hypothesis, rationale}."""

    # -------------------------------------------------------------------------
    # refine step
    # -------------------------------------------------------------------------
    refined_ideas: list[dict]
    """Feasibility-filtered and improved ideas from refine step."""

    # -------------------------------------------------------------------------
    # propose_experiments step
    # -------------------------------------------------------------------------
    proposals: list[dict]
    """Fully-specified experiment proposals including hyperparameters.
    Each dict: {name, description, dataset, detector, hyperparameters,
                expected_outputs, success_criterion, ...}"""

    proposal_seed_planning_notes: str
    """Optional operator-authored notes attached to a saved proposal seed."""

    # -------------------------------------------------------------------------
    # plan_implementation step
    # -------------------------------------------------------------------------
    implementation_plans: list[dict]
    """Detailed implementation plan per proposal (structured JSON, no code).
    Each dict: {proposal_name, base_class, init_logic, main_method_steps,
                output_keys, ...}"""

    # -------------------------------------------------------------------------
    # implement step
    # -------------------------------------------------------------------------
    implementations: list[dict]
    """Generated implementation per plan.
    Each dict: {script_path, class_name, proposal_name, proposal, plan}."""

    # -------------------------------------------------------------------------
    # validate step
    # -------------------------------------------------------------------------
    validation_results: list[dict]
    """Validation outcome per implementation.
    Each dict: {script_path, class_name, passed, test_file, test_output, attempts}."""

    # -------------------------------------------------------------------------
    # prepare_experiment step
    # -------------------------------------------------------------------------
    experiment_artifacts: list[dict]
    """Domain-specific prepared artifacts for experiment execution.
    Examples: NeuralSignal feature datasets, trading market-data bundles,
    parameter grids, cached backtest inputs."""

    # -------------------------------------------------------------------------
    # create_dataset step (legacy compatibility alias)
    # -------------------------------------------------------------------------
    datasets: list[dict]
    """Created or cached datasets.
    Each dict: {dataset_id, feature_set_name, file_path, rows, columns, ...}."""

    # -------------------------------------------------------------------------
    # run_experiment step
    # -------------------------------------------------------------------------
    experiment_results: list[dict]
    """Raw experiment results per proposal.
    Each dict: {experiment_id, proposal_name, proposal, metrics, feature_importance, ...}."""

    experiment_jobs: list[dict]
    """Long-running submitted experiment jobs.
    Each dict: {job_id, runner, stage, proposal_name, status, job_dir, result_path, ...}."""

    # -------------------------------------------------------------------------
    # create_model step
    # -------------------------------------------------------------------------
    models: list[dict]
    """Optional trained models per experiment.
    Each dict: {model_id, experiment_id, metrics, params, feature_importance}."""

    # -------------------------------------------------------------------------
    # evaluate step
    # -------------------------------------------------------------------------
    evaluation_summary: dict
    """Analysis of results across all proposals.
    Keys: {best_metric_value, best_proposal, per_proposal_analysis, conclusion}."""

    # -------------------------------------------------------------------------
    # store_results step
    # -------------------------------------------------------------------------
    stored_result_ids: list[str]
    """IDs of records persisted (MLflow run IDs, ChromaDB IDs, MongoDB IDs, etc.)."""

    # -------------------------------------------------------------------------
    # propose_next_steps step
    # -------------------------------------------------------------------------
    next_steps: list[dict]
    """Proposed follow-up research directions.
    After ranking, this is the selected ordered subset used for loop execution.
    Each dict: {title, rationale, suggested_direction, priority}."""

    next_step_selection: dict
    """Diagnostics for next-step ranking and pruning.
    Shape: {selected: [...], dropped: [...], ranking_mode: str}."""

    # -------------------------------------------------------------------------
    # Accumulated non-fatal errors
    # -------------------------------------------------------------------------
    errors: list[str]
    """Non-fatal error messages from any step. Fatal errors raise and abort."""
