"""Prepare experiment artifacts using the active domain adapter.

This node is intentionally generic. For NeuralSignal, the prepared artifact is a
feature dataset. For trading, it might be a data slice, parameter grid, or cached
market data bundle.
"""
from __future__ import annotations

from graph.state import ResearchState
from plugins.loader import adapter_has, load_adapter
from utils.logger import get_logger

log = get_logger(__name__)


def prepare_experiment_node(state: ResearchState, profile: dict) -> dict:
    proposals = state.get("proposals") or []
    implementations = state.get("implementations") or []

    if not proposals:
        log.warning("prepare_experiment_node | No proposals - skipping")
        return {"experiment_artifacts": []}

    try:
        adapter = load_adapter(profile)
    except Exception as exc:
        return {
            "experiment_artifacts": [],
            "errors": (state.get("errors") or []) + [f"prepare_experiment: adapter load failed: {exc}"],
        }

    impl_by_name = {
        impl.get("proposal_name", ""): impl
        for impl in implementations
        if impl.get("proposal_name")
    }

    artifacts: list[dict] = []
    errors = list(state.get("errors") or [])

    for proposal in proposals:
        proposal_name = proposal.get("name", "unknown")
        impl = impl_by_name.get(proposal_name)
        log.info("prepare_experiment_node | Preparing proposal=%r", proposal_name)

        try:
            if adapter_has(adapter, "prepare_experiment"):
                artifact = adapter.prepare_experiment(profile, proposal, impl, state)
            elif adapter_has(adapter, "create_dataset"):
                artifact = adapter.create_dataset(profile, proposal, impl)
            else:
                artifact = {
                    "artifact_id": proposal_name,
                    "proposal_name": proposal_name,
                    "artifact_type": "none",
                }

            if artifact:
                artifact.setdefault("proposal_name", proposal_name)
                artifact.setdefault("artifact_type", artifact.get("type", "prepared_artifact"))
                artifacts.append(artifact)
            else:
                errors.append(f"prepare_experiment: no artifact returned for {proposal_name}")
        except Exception as exc:
            log.error("prepare_experiment_node | Failed for %r: %s", proposal_name, exc, exc_info=True)
            errors.append(f"prepare_experiment: {proposal_name} failed: {exc}")

    delta: dict = {"experiment_artifacts": artifacts, "errors": errors}

    # Compatibility alias for existing NeuralSignal/debug tooling.
    datasets = [a for a in artifacts if a.get("artifact_type") == "dataset" or a.get("dataset_id")]
    if datasets:
        delta["datasets"] = datasets

    return delta

