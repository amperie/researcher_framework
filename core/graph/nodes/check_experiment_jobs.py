"""Check long-running experiment jobs through the active domain adapter."""
from __future__ import annotations

import time

from core.graph.state import ResearchState
from core.plugins.loader import adapter_has, load_adapter
from core.plugins.job_runner import TERMINAL_STATUSES
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

    log.info("check_experiment_jobs_node | Draining experiment jobs through adapter")
    poll_seconds = int((profile.get("execution") or {}).get("poll_interval_seconds", 30) or 30)
    merged_state: dict = dict(state)
    iteration = 0

    while True:
        iteration += 1
        try:
            delta = adapter.check_experiment_jobs(profile, merged_state)
        except Exception as exc:
            log.error("check_experiment_jobs_node | Adapter failed: %s", exc, exc_info=True)
            return {
                "experiment_jobs": list(merged_state.get("experiment_jobs") or []),
                "errors": (merged_state.get("errors") or []) + [f"check_experiment_jobs: adapter failed: {exc}"],
            }

        normalized = _normalize_delta(delta, merged_state)
        merged_state = {**merged_state, **normalized}
        jobs = list(merged_state.get("experiment_jobs") or [])
        active_jobs = [job for job in jobs if job.get("status") not in TERMINAL_STATUSES]
        log.info(
            "check_experiment_jobs_node | Iteration %d complete - jobs=%d active=%d results=%d",
            iteration,
            len(jobs),
            len(active_jobs),
            len(merged_state.get("experiment_results") or []),
        )
        if not active_jobs:
            return normalized
        time.sleep(poll_seconds)


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
