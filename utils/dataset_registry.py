"""Dataset registry — loads and caches configs/datasets.yaml.

Each entry in the registry maps a logical dataset name to its MongoDB source
coordinates (application_name = database, sub_application_name = collection)
and metadata used by the experiment runner and feature-proposal LLM.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from utils.logger import get_logger

log = get_logger(__name__)

_DATASETS_FILE = Path("configs/datasets.yaml")


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    try:
        data = yaml.safe_load(_DATASETS_FILE.read_text(encoding="utf-8"))
        entries = data.get("datasets") or []
        log.debug("dataset_registry | Loaded %d dataset(s) from %s", len(entries), _DATASETS_FILE)
        return entries
    except Exception as exc:
        log.warning("dataset_registry | Failed to load %s: %s", _DATASETS_FILE, exc)
        return []


def get_all_datasets() -> list[dict]:
    """Return all registered dataset entries."""
    return _load()


def get_dataset(name: str) -> dict | None:
    """Look up a dataset entry by name. Returns None if not found."""
    for entry in _load():
        if entry.get("name") == name:
            return entry
    return None


def get_dataset_or_first(name: str | None) -> dict | None:
    """Look up by name; fall back to first entry if not found or name is None.

    Returns None only if the registry is empty.
    """
    if name:
        entry = get_dataset(name)
        if entry:
            return entry
        log.warning(
            "dataset_registry | Dataset %r not in registry — falling back to first entry", name,
        )
    entries = _load()
    return entries[0] if entries else None
