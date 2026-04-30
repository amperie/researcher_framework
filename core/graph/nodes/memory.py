"""Helpers for incremental memory persistence from graph nodes."""
from __future__ import annotations

from core.memory import MemoryService, build_core_memory_records, dedupe_memory_records
from core.plugins.loader import adapter_has, load_adapter
from core.utils.logger import get_logger

log = get_logger(__name__)


def build_memory_records_for_state(profile: dict, state: dict) -> list[dict]:
    """Build the full current memory projection for a merged graph state."""
    records = list(build_core_memory_records(profile, state))
    try:
        adapter = load_adapter(profile)
    except Exception:
        adapter = None
    if adapter is not None and adapter_has(adapter, "build_memory_records"):
        try:
            records.extend(adapter.build_memory_records(profile, state) or [])
        except Exception as exc:
            log.warning("memory | adapter build_memory_records failed: %s", exc)
    return dedupe_memory_records(records)


def persist_memory_records_for_state(profile: dict, state: dict) -> None:
    """Persist the current memory records for the given merged graph state."""
    records = build_memory_records_for_state(profile, state)
    if not records:
        return
    MemoryService.for_profile(profile).persist_records(records)
