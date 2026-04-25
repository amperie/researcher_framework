"""NeuralSignal task callables for the generic subprocess runner.

These functions run inside the NeuralSignal-capable Python process. Keep heavy
imports here instead of in ``adapter.py``.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def create_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a NeuralSignal feature dataset and return output file paths."""
    cfg = _automation_config(payload)
    _enable_mongo_no_cursor_timeout()
    _inject_feature_processor(cfg)

    from neuralsignal.automation import create_dataset as ns_create_dataset  # type: ignore

    balanced_cfg = cfg.get("balanced_target") or {}
    if balanced_cfg.get("enabled"):
        result = _create_balanced_dataset(cfg, ns_create_dataset)
        result["file_paths"] = _move_dataset_files(result.get("file_paths") or [], cfg)
        return result

    file_paths = ns_create_dataset(cfg, create_dataset=True) or []
    return {"file_paths": _move_dataset_files(file_paths, cfg)}


def create_s1_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a NeuralSignal S1 model and return best-model metrics."""
    cfg = _automation_config(payload)
    _inject_feature_processor(cfg)

    from neuralsignal.automation import create_s1_model as ns_create_s1_model  # type: ignore

    models = ns_create_s1_model(cfg)
    if not models:
        return {"error": "No models returned"}

    best = max(models, key=lambda model: _model_section(model, "metrics").get("test_auc", 0.0))
    return {
        "metrics": dict(_model_section(best, "metrics")),
        "params": dict(_model_section(best, "params")),
        "feature_importance": dict(_model_section(best, "artifacts").get("feature_importance", {}) or {}),
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


def _move_dataset_files(file_paths: list[Any], cfg: dict[str, Any]) -> list[str]:
    output_dir = cfg.get("dataset_output_dir")
    if not output_dir:
        return [str(path) for path in file_paths]

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for file_path in file_paths:
        src = Path(str(file_path))
        if not src.is_absolute():
            src = Path.cwd() / src
        dest = dest_dir / src.name
        if src.resolve() != dest.resolve() and src.exists():
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
        moved.append(str(dest))
    return moved


def _enable_mongo_no_cursor_timeout() -> None:
    """Patch the NeuralSignal Mongo backend for long-running dataset scans.

    Dataset creation can spend many seconds per scan deserializing tensors and
    featurizing activations. The SDK's default Mongo cursor uses the server's
    normal idle timeout, which can expire during long runs and raise
    CursorNotFound. Using no_cursor_timeout keeps the cursor alive for the
    lifetime of the task process.
    """
    from neuralsignal.backend.mongo_backend import MongoBackend  # type: ignore

    if getattr(MongoBackend, "_nsr_no_cursor_timeout_enabled", False):
        return

    def _query_no_cursor_timeout(self: Any, query: dict) -> Any:
        return self.col.find(query, no_cursor_timeout=True)

    MongoBackend.query = _query_no_cursor_timeout
    MongoBackend._nsr_no_cursor_timeout_enabled = True


def _model_section(model: Any, name: str) -> dict[str, Any]:
    value = getattr(model, name, None)
    if isinstance(value, dict):
        return value

    config = getattr(model, "config", None)
    if isinstance(config, dict):
        section = config.get(name)
        if isinstance(section, dict):
            return section

    try:
        section = model[name]
    except Exception:
        section = None
    return section if isinstance(section, dict) else {}


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

    cfg["feature_processor"] = FeatureProcessor(
        feature_sets=[_ScanShapeCompatibleFeatureSet(cls(fs_cfg))]
    )
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


class _ScanShapeCompatibleFeatureSet:
    """Retry generated feature sets with common NeuralSignal scan shapes.

    Generated code is validated on synthetic scans, but real NeuralSignal scans
    currently store activations as a flat dict keyed by string layer ids:
    ``outputs[layer_id] -> Tensor``. Some generated implementations assume a
    pass/batch wrapper: ``outputs[0][layer_id]``. This adapter keeps generated
    code usable while making empty feature outputs fail loudly.
    """

    def __init__(self, feature_set: Any):
        self.feature_set = feature_set
        self.config = feature_set.config

    def get_feature_set_name(self) -> str:
        return self.feature_set.get_feature_set_name()

    def get_config(self) -> dict[str, Any]:
        if hasattr(self.feature_set, "get_config"):
            return self.feature_set.get_config()
        return self.config

    def make_column_name(self, column_name: str) -> str:
        if hasattr(self.feature_set, "make_column_name"):
            return self.feature_set.make_column_name(column_name)
        return f"{self.get_feature_set_name()}__{column_name}"

    def process_feature_set(self, scan: dict[str, Any]) -> Any:
        output_format = self.config.get("output_format", "name_and_value_columns")
        first_result: Any = None
        errors: list[str] = []

        for candidate in _scan_shape_variants(scan):
            try:
                result = self.feature_set.process_feature_set(candidate)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue

            if first_result is None:
                first_result = result
            if output_format == "tensor_dict":
                return result
            if _has_feature_columns(result):
                return result

        if first_result is not None:
            raise RuntimeError(
                f"{self.get_feature_set_name()} returned no feature columns for real scan shape"
            )
        raise RuntimeError(
            f"{self.get_feature_set_name()} failed for all supported scan shapes: {'; '.join(errors[-3:])}"
        )


def _scan_shape_variants(scan: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [scan]
    outputs = scan.get("outputs")
    inputs = scan.get("inputs")

    if _is_flat_layer_tensor_dict(outputs):
        variants.append({**scan, "outputs": [outputs]})
        variants.append({**scan, "outputs": {0: outputs}})
    elif _is_nested_single_batch_dict(outputs):
        flat_outputs = next(iter(outputs.values()))
        variants.append({**scan, "outputs": flat_outputs})
        variants.append({**scan, "outputs": [flat_outputs]})

    if _is_flat_layer_tensor_dict(inputs):
        base_variants = list(variants)
        for variant in base_variants:
            variants.append({**variant, "inputs": [inputs]})
            variants.append({**variant, "inputs": {0: inputs}})

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for variant in variants:
        key = (str(type(variant.get("outputs"))), str(type(variant.get("inputs"))))
        if key not in seen:
            unique.append(variant)
            seen.add(key)
    return unique


def _is_flat_layer_tensor_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and not isinstance(next(iter(value.values())), dict)


def _is_nested_single_batch_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and isinstance(next(iter(value.values())), dict)


def _has_feature_columns(result: Any) -> bool:
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0])
    if isinstance(result, dict):
        return bool(result)
    columns = getattr(result, "columns", None)
    return columns is not None and len(columns) > 0

