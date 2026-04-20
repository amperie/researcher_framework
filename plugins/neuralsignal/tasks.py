"""NeuralSignal task callables for the generic subprocess runner.

These functions run inside the NeuralSignal-capable Python process. Keep heavy
imports here instead of in ``adapter.py``.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from typing import Any

log = logging.getLogger(__name__)


def create_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a NeuralSignal feature dataset and return output file paths."""
    cfg = _automation_config(payload)
    _inject_feature_processor(cfg)

    from neuralsignal.automation import create_dataset as ns_create_dataset  # type: ignore

    balanced_cfg = cfg.get("balanced_target") or {}
    if balanced_cfg.get("enabled"):
        return _create_balanced_dataset(cfg, ns_create_dataset)

    file_paths = ns_create_dataset(cfg, create_dataset=True) or []
    return {"file_paths": [str(path) for path in file_paths]}


def create_s1_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a NeuralSignal S1 model and return best-model metrics."""
    cfg = _automation_config(payload)
    _inject_feature_processor(cfg)

    from neuralsignal.automation import create_s1_model as ns_create_s1_model  # type: ignore

    models = ns_create_s1_model(cfg)
    if not models:
        return {"error": "No models returned"}

    best = max(models, key=lambda model: model.metrics.get("test_auc", 0.0))
    return {
        "metrics": dict(best.metrics),
        "params": dict(best.params) if best.params else {},
        "feature_importance": best.artifacts.get("feature_importance", {}) if best.artifacts else {},
    }


def _automation_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge task payload over NeuralSignal's packaged automation defaults."""
    from neuralsignal.automation import get_config  # type: ignore

    cfg = get_config()
    cfg.update(payload)
    return cfg


def _create_balanced_dataset(cfg: dict[str, Any], ns_create_dataset: Any) -> dict[str, Any]:
    """Create a class-balanced dataset by running one query per target value."""
    balanced_cfg = cfg.get("balanced_target") or {}
    field = balanced_cfg.get("field", "ground_truth")
    values = list(balanced_cfg.get("values") or [0, 1])
    if not values:
        raise ValueError("balanced_target.values must contain at least one target value")

    total_limit = int(cfg.get("dataset_row_limit", cfg.get("row_limit", 0)) or 0)
    if total_limit <= 0:
        raise ValueError("balanced_target requires dataset_row_limit > 0")

    per_value_limits = _split_limit(total_limit, len(values))
    base_query = dict(cfg.get("query") or {})
    all_paths: list[str] = []
    pulls: list[dict[str, Any]] = []

    for idx, (value, limit) in enumerate(zip(values, per_value_limits)):
        if limit <= 0:
            continue
        class_cfg = dict(cfg)
        class_cfg["query"] = {**base_query, field: value}
        class_cfg["dataset_row_limit"] = limit
        class_cfg["row_limit"] = limit
        class_cfg["overwrite_dataset_file"] = idx == 0
        class_cfg["write_header"] = idx == 0

        file_paths = ns_create_dataset(class_cfg, create_dataset=True) or []
        str_paths = [str(path) for path in file_paths]
        all_paths.extend(str_paths)
        pulls.append({
            "field": field,
            "value": value,
            "row_limit": limit,
            "query": class_cfg["query"],
            "file_paths": str_paths,
        })

    return {
        "file_paths": _dedupe_preserve_order(all_paths),
        "balanced_target": {
            "enabled": True,
            "field": field,
            "values": values,
            "total_row_limit": total_limit,
            "pulls": pulls,
        },
    }


def _split_limit(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _inject_feature_processor(cfg: dict[str, Any]) -> None:
    """Load a generated FeatureSetBase subclass and inject FeatureProcessor."""
    if not cfg.get("feature_set_class_path"):
        return

    from neuralsignal.core.modules.feature_sets.feature_processor import FeatureProcessor  # type: ignore
    from neuralsignal.core.modules.feature_sets import feature_set_base as real_base_module  # type: ignore

    FeatureSetBase = real_base_module.FeatureSetBase

    class_path = cfg["feature_set_class_path"]
    class_name = cfg.get("feature_set_class_name", "")

    spec = importlib.util.spec_from_file_location("_dyn_feature_set", class_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature set module from {class_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    finally:
        # Generated files may install offline validation stubs into sys.modules.
        # Restore the real runtime base class before NeuralSignal imports use it.
        sys.modules[
            "neuralsignal.core.modules.feature_sets.feature_set_base"
        ].FeatureSetBase = FeatureSetBase

    cls = _find_feature_set_class(module, class_name, FeatureSetBase)
    if cls is None:
        raise RuntimeError(f"No FeatureSetBase subclass found in {class_path}")
    if not issubclass(cls, FeatureSetBase):
        cls = _wrap_feature_set_class(cls, FeatureSetBase)

    fs_cfg: dict[str, Any] = {
        "name": class_name or cls.__name__,
        "output_format": "name_and_value_columns",
    }
    if cfg.get("ffn_layer_patterns"):
        fs_cfg["ffn_layer_patterns"] = cfg["ffn_layer_patterns"]
    if cfg.get("attn_layer_patterns"):
        fs_cfg["attn_layer_patterns"] = cfg["attn_layer_patterns"]

    cfg["feature_processor"] = FeatureProcessor(feature_sets=[cls(fs_cfg)])
    cfg["feature_set_configs"] = None
    log.info("Injected FeatureProcessor for %s", cls.__name__)


def _find_feature_set_class(module: Any, class_name: str, feature_set_base: type) -> type | None:
    """Find a generated feature set by nominal or structural contract."""
    if class_name:
        cls = getattr(module, class_name, None)
        if isinstance(cls, type) and _looks_like_feature_set_class(cls):
            return cls

    strict = [
        obj for obj in vars(module).values()
        if isinstance(obj, type)
        and obj is not feature_set_base
        and issubclass(obj, feature_set_base)
        and _looks_like_feature_set_class(obj)
    ]
    if strict:
        return strict[0]

    structural = [
        obj for obj in vars(module).values()
        if isinstance(obj, type)
        and obj.__name__ != "FeatureSetBase"
        and _looks_like_feature_set_class(obj)
    ]
    return structural[0] if structural else None


def _looks_like_feature_set_class(cls: type) -> bool:
    return callable(getattr(cls, "get_feature_set_name", None)) and callable(
        getattr(cls, "process_feature_set", None)
    )


def _wrap_feature_set_class(cls: type, feature_set_base: type) -> type:
    """Make structurally valid generated classes nominally inherit the real base."""
    return type(cls.__name__, (cls, feature_set_base), {"__module__": cls.__module__})

