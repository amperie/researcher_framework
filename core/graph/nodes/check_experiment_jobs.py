"""Check long-running experiment jobs through the active domain adapter."""
from __future__ import annotations

from core.graph.state import ResearchState
from core.plugins.loader import adapter_has, load_adapter
from core.utils.logger import get_logger

log = get_logger(__name__)


def check_experiment_jobs_node(state: ResearchState, profile: dict) -> dict:
    try:
        adapter = load_adapter(profile)
    except Exception as exc:
        return {
            "experiment_jobs": list(state.get("experiment_jobs") or []),
            "errors": (state.get("errors") or []) + [f"check_experiment_jobs: adapter load failed: {exc}"],
        }

    if not adapter_has(adapter, "check_experiment_jobs"):
        return {
            "experiment_jobs": list(state.get("experiment_jobs") or []),
            "errors": (state.get("errors") or []) + [
                "check_experiment_jobs: adapter does not implement check_experiment_jobs"
            ],
        }

    log.info("check_experiment_jobs_node | Delegating full state to adapter")
    try:
        delta = adapter.check_experiment_jobs(profile, state)
    except Exception as exc:
        log.error("check_experiment_jobs_node | Adapter failed: %s", exc, exc_info=True)
        return {
            "experiment_jobs": list(state.get("experiment_jobs") or []),
            "errors": (state.get("errors") or []) + [f"check_experiment_jobs: adapter failed: {exc}"],
        }
    return _normalize_delta(delta, state)


def _normalize_delta(delta: dict | None, state: ResearchState) -> dict:
    if not delta:
        return {
            "experiment_jobs": list(state.get("experiment_jobs") or []),
            "errors": list(state.get("errors") or []),
        }
    normalized = dict(delta)
    normalized.setdefault("experiment_jobs", list(state.get("experiment_jobs") or []))
    normalized.setdefault("errors", list(state.get("errors") or []))
    return normalized
