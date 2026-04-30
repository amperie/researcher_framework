"""Deterministic fingerprint helpers for canonical memory objects."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint_json(value: Any) -> str:
    """Return a stable sha256 fingerprint for a JSON-serializable value."""
    normalized = _normalize(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    return value
