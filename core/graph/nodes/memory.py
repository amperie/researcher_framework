"""Helpers for incremental memory persistence from graph nodes."""
from __future__ import annotations

from core.memory import MemoryService, build_core_memory_records, dedupe_memory_records
from core.plugins.loader import adapter_has, load_adapter
from core.utils.logger import get_logger

log = get_logger(__name__)


def build_memory_records_for_state(profile: dict, state: dict) -> list[dict]:
    """Build the full current memory projection for a merged graph state."""
    log.debug("memory.node | Building memory records for profile=%r", profile.get("name"))
    records = list(build_core_memory_records(profile, state))
    try:
        adapter = load_adapter(profile)
    except Exception as exc:
        log.debug("memory.node | No adapter memory records available: %s", exc)
        adapter = None
    if adapter is not None and adapter_has(adapter, "build_memory_records"):
        try:
            adapter_records = adapter.build_memory_records(profile, state) or []
            log.debug("memory.node | Adapter built %d memory record(s)", len(adapter_records))
            records.extend(adapter_records)
        except Exception as exc:
            log.warning("memory | adapter build_memory_records failed: %s", exc)
    deduped = dedupe_memory_records(records)
    log.info("memory.node | Built %d memory record(s) for profile=%r", len(deduped), profile.get("name"))
    return deduped


def persist_memory_records_for_state(profile: dict, state: dict) -> None:
    """Persist the current memory records for the given merged graph state."""
    records = build_memory_records_for_state(profile, state)
    if not records:
        log.debug("memory.node | No memory records to persist for profile=%r", profile.get("name"))
        return
    log.info("memory.node | Persisting %d memory record(s) for profile=%r", len(records), profile.get("name"))
    MemoryService.for_profile(profile).persist_records(records)


def emit_memory_record(
    profile: dict,
    *,
    object_type: str,
    payload: dict,
    node: str,
    kind: str = "",
    object_key: str = "",
    object_role: str = "artifact",
    title: str = "",
    summary: str = "",
    metadata: dict | None = None,
    tags: list[str] | None = None,
    source_state_keys: list[str] | None = None,
    source_record_ids: list[str] | None = None,
    blob_refs: list[dict] | None = None,
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> dict:
    """Emit and persist a single typed memory object from a graph node."""
    log.info(
        "memory.node | Emitting memory record profile=%r node=%r object_type=%r",
        profile.get("name"),
        node,
        object_type,
    )
    return MemoryService.for_profile(profile).emit(
        profile=profile,
        object_type=object_type,
        payload=payload,
        node=node,
        kind=kind,
        object_key=object_key,
        object_role=object_role,
        title=title,
        summary=summary,
        metadata=metadata or {},
        tags=tags or [],
        source_state_keys=source_state_keys or [],
        source_record_ids=source_record_ids or [],
        blob_refs=blob_refs or [],
        entities=entities or [],
        relations=relations or [],
    )
