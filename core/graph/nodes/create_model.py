"""Create model step — train a model from experiment results (optional step).

Delegates to the profile's experiment adapter. Profiles that don't need model
training can omit this step from pipeline.steps entirely.

Reads:
    state['experiment_results']
    state['datasets']

Writes:
    state['models']
"""
from __future__ import annotations

import importlib

from core.graph.state import ResearchState
from core.utils.logger import get_logger

log = get_logger(__name__)


def create_model_node(state: ResearchState, profile: dict) -> dict:
    experiment_results = state.get("experiment_results") or []
    datasets = state.get("datasets") or []

    if not experiment_results:
        log.warning("create_model_node | No experiment results — skipping model training")
        return {"models": []}

    adapter_path = profile.get("experiment_adapter")
    if not adapter_path:
        return {
            "models": [],
            "errors": (state.get("errors") or []) + ["create_model: no experiment_adapter configured"],
        }

    try:
        adapter = importlib.import_module(adapter_path)
    except ImportError as exc:
        return {
            "models": [],
            "errors": (state.get("errors") or []) + [f"create_model: adapter import failed: {exc}"],
        }

    if not hasattr(adapter, "train_model"):
        log.info("create_model_node | Adapter has no train_model() — skipping")
        return {"models": []}

    dataset_by_proposal = {d.get("proposal_name", ""): d for d in datasets}

    models: list[dict] = []
    errors = list(state.get("errors") or [])

    for result in experiment_results:
        proposal_name = result.get("proposal_name", "unknown")
        dataset = dataset_by_proposal.get(proposal_name)

        log.info("create_model_node | Training model for proposal=%r", proposal_name)
        try:
            model_entry = adapter.train_model(
                profile=profile,
                experiment_result=result,
                dataset=dataset,
            )
            if model_entry:
                model_entry.setdefault("experiment_id", result.get("experiment_id"))
                model_entry.setdefault("proposal_name", proposal_name)
                models.append(model_entry)
                log.info(
                    "create_model_node | Model trained — id=%s metrics=%s",
                    model_entry.get("model_id"), model_entry.get("metrics"),
                )
        except Exception as exc:
            log.error(
                "create_model_node | Training failed for proposal=%r: %s", proposal_name, exc, exc_info=True,
            )
            errors.append(f"create_model: {proposal_name} failed: {exc}")

    return {"models": models, "errors": errors}
