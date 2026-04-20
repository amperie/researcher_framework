"""NeuralSignal task callables for the generic subprocess runner.

These functions run inside the NeuralSignal-capable Python process. Keep heavy
imports here instead of in ``adapter.py``.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any

log = logging.getLogger(__name__)


def create_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a NeuralSignal feature dataset and return output file paths."""
    cfg = dict(payload)
    _inject_feature_processor(cfg)

    from neuralsignal.automation.dataset_automation_core import create_dataset as ns_create_dataset  # type: ignore

    file_paths = ns_create_dataset(cfg, create_dataset=True) or []
    return {"file_paths": [str(path) for path in file_paths]}


def create_s1_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a NeuralSignal S1 model and return best-model metrics."""
    cfg = dict(payload)
    _inject_feature_processor(cfg)

    from neuralsignal.automation.dataset_automation_core import create_s1_model as ns_create_s1_model  # type: ignore

    models = ns_create_s1_model(cfg)
    if not models:
        return {"error": "No models returned"}

    best = max(models, key=lambda model: model.metrics.get("test_auc", 0.0))
    return {
        "metrics": dict(best.metrics),
        "params": dict(best.params) if best.params else {},
        "feature_importance": best.artifacts.get("feature_importance", {}) if best.artifacts else {},
    }


def _inject_feature_processor(cfg: dict[str, Any]) -> None:
    """Load a generated FeatureSetBase subclass and inject FeatureProcessor."""
    if "feature_set_class_path" not in cfg:
        return

    from neuralsignal.core.modules.feature_sets.feature_processor import FeatureProcessor  # type: ignore
    from neuralsignal.core.modules.feature_sets.feature_set_base import FeatureSetBase  # type: ignore

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

    cls = next(
        (
            obj for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, FeatureSetBase) and obj is not FeatureSetBase
        ),
        None,
    )
    if cls is None:
        raise RuntimeError(f"No FeatureSetBase subclass found in {class_path}")

    fs_cfg: dict[str, Any] = {
        "name": class_name or cls.__name__,
        "output_format": "name_and_value_columns",
    }
    if cfg.get("ffn_layer_patterns"):
        fs_cfg["ffn_layer_patterns"] = cfg["ffn_layer_patterns"]
    if cfg.get("attn_layer_patterns"):
        fs_cfg["attn_layer_patterns"] = cfg["attn_layer_patterns"]

    cfg["feature_processor"] = FeatureProcessor(feature_sets=[cls(fs_cfg)])
    log.info("Injected FeatureProcessor for %s", cls.__name__)

