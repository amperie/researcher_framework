"""Experiment adapter loading helpers."""
from __future__ import annotations

import importlib
from typing import Any


def load_adapter(profile: dict[str, Any]) -> Any:
    """Load the adapter declared by ``profile['experiment_adapter']``.

    If the module exposes ``get_adapter()``, return that object. Otherwise return
    the module itself so legacy module-level adapter functions continue to work.
    """
    adapter_path = profile.get("experiment_adapter")
    if not adapter_path:
        raise ValueError("profile is missing experiment_adapter")

    module = importlib.import_module(adapter_path)
    if hasattr(module, "get_adapter"):
        return module.get_adapter()
    return module


def adapter_has(adapter: Any, method_name: str) -> bool:
    return callable(getattr(adapter, method_name, None))

