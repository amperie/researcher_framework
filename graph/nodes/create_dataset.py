"""Create dataset step — build a feature dataset for each experiment proposal.

Delegates to the experiment adapter declared in the profile. The adapter handles
all domain-specific logic (neuralsignal scan extraction, trading bar assembly, etc.)

Reads:
    state['proposals']
    state['implementations']

Writes:
    state['datasets']
"""
from __future__ import annotations

import importlib

from graph.state import ResearchState
from utils.logger import get_logger

log = get_logger(__name__)


def create_dataset_node(state: ResearchState, profile: dict) -> dict:
    proposals = state.get("proposals") or []
    implementations = state.get("implementations") or []

    if not proposals:
        log.warning("create_dataset_node | No proposals — skipping dataset creation")
        return {"datasets": []}

    adapter_path = profile.get("experiment_adapter")
    if not adapter_path:
        log.error("create_dataset_node | No experiment_adapter in profile")
        return {
            "datasets": [],
            "errors": (state.get("errors") or []) + ["create_dataset: no experiment_adapter configured"],
        }

    try:
        adapter = importlib.import_module(adapter_path)
    except ImportError as exc:
        log.error("create_dataset_node | Failed to import adapter %r: %s", adapter_path, exc)
        return {
            "datasets": [],
            "errors": (state.get("errors") or []) + [f"create_dataset: adapter import failed: {exc}"],
        }

    if not hasattr(adapter, "create_dataset"):
        log.error("create_dataset_node | Adapter %r has no create_dataset()", adapter_path)
        return {
            "datasets": [],
            "errors": (state.get("errors") or []) + [f"create_dataset: adapter missing create_dataset()"],
        }

    # Build an impl lookup by proposal_name for quick access
    impl_by_name: dict[str, dict] = {
        impl.get("proposal_name", ""): impl
        for impl in implementations
        if impl.get("proposal_name")
    }

    datasets: list[dict] = []
    errors = list(state.get("errors") or [])

    for proposal in proposals:
        proposal_name = proposal.get("name", "unknown")
        impl = impl_by_name.get(proposal_name)
        log.info("create_dataset_node | Creating dataset for proposal=%r", proposal_name)

        try:
            dataset_entry = adapter.create_dataset(
                profile=profile,
                proposal=proposal,
                implementation=impl,
            )
            if dataset_entry:
                datasets.append(dataset_entry)
                log.info(
                    "create_dataset_node | Dataset created — id=%s, rows=%s",
                    dataset_entry.get("dataset_id"), dataset_entry.get("rows"),
                )
            else:
                log.warning("create_dataset_node | Adapter returned None for %r", proposal_name)
                errors.append(f"create_dataset: no dataset returned for {proposal_name}")
        except Exception as exc:
            log.error(
                "create_dataset_node | Failed for proposal=%r: %s", proposal_name, exc, exc_info=True,
            )
            errors.append(f"create_dataset: {proposal_name} failed: {exc}")

    return {"datasets": datasets, "errors": errors}
