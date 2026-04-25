"""Run experiment step — execute experiments against created datasets.

Delegates to the profile's experiment adapter.

Reads:
    state['datasets']
    state['proposals']
    state['implementations']

Writes:
    state['experiment_results']
"""
from __future__ import annotations

import importlib
from uuid import uuid4

from core.graph.state import ResearchState
from core.utils.logger import get_logger

log = get_logger(__name__)


def run_experiment_node(state: ResearchState, profile: dict) -> dict:
    datasets = state.get("datasets") or []
    proposals = state.get("proposals") or []
    implementations = state.get("implementations") or []

    if not datasets:
        log.warning("run_experiment_node | No datasets available — skipping")
        return {"experiment_results": []}

    adapter_path = profile.get("experiment_adapter")
    if not adapter_path:
        return {
            "experiment_results": [],
            "errors": (state.get("errors") or []) + ["run_experiment: no experiment_adapter configured"],
        }

    try:
        adapter = importlib.import_module(adapter_path)
    except ImportError as exc:
        return {
            "experiment_results": [],
            "errors": (state.get("errors") or []) + [f"run_experiment: adapter import failed: {exc}"],
        }

    if not hasattr(adapter, "run_experiment"):
        return {
            "experiment_results": [],
            "errors": (state.get("errors") or []) + ["run_experiment: adapter missing run_experiment()"],
        }

    # Build lookups
    proposal_by_name = {p.get("name", ""): p for p in proposals}
    impl_by_name = {i.get("proposal_name", ""): i for i in implementations}
    dataset_by_proposal = {d.get("proposal_name", ""): d for d in datasets}

    results: list[dict] = []
    errors = list(state.get("errors") or [])

    for dataset in datasets:
        proposal_name = dataset.get("proposal_name", "unknown")
        proposal = proposal_by_name.get(proposal_name, {})
        impl = impl_by_name.get(proposal_name)
        experiment_id = str(uuid4())

        log.info("run_experiment_node | Running experiment for proposal=%r", proposal_name)
        try:
            result = adapter.run_experiment(
                profile=profile,
                proposal=proposal,
                implementation=impl,
                dataset=dataset,
                experiment_id=experiment_id,
            )
            if result:
                result.setdefault("experiment_id", experiment_id)
                result.setdefault("proposal_name", proposal_name)
                results.append(result)
                log.info(
                    "run_experiment_node | Experiment complete — id=%s metrics=%s",
                    experiment_id, result.get("metrics"),
                )
        except Exception as exc:
            log.error(
                "run_experiment_node | Failed for proposal=%r: %s", proposal_name, exc, exc_info=True,
            )
            errors.append(f"run_experiment: {proposal_name} failed: {exc}")

    return {"experiment_results": results, "errors": errors}
