"""Subprocess bridge for neuralsignal automation calls.

Invoked by the neuralsignal dataset_adapter when the SDK cannot be imported
into the main process (heavy ML deps not in this venv).

Usage (called by _call_bridge in dataset_adapter.py):
    <neuralsignal_python> plugins/neuralsignal/bridge.py create_dataset
    <neuralsignal_python> plugins/neuralsignal/bridge.py create_s1_model

JSON config is read from stdin; result is written as JSON to stdout.
PYTHONPATH must include the neuralsignal source root (dataset_adapter handles this).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _inject_feature_processor(cfg: dict) -> None:
    """If cfg has 'feature_set_class_path', load the class and inject a FeatureProcessor."""
    if "feature_set_class_path" not in cfg:
        return

    from neuralsignal.core.modules.feature_sets.feature_set_base import FeatureSetBase  # type: ignore
    from neuralsignal.core.modules.feature_sets.feature_processor import FeatureProcessor  # type: ignore

    class_path = cfg["feature_set_class_path"]
    class_name = cfg.get("feature_set_class_name", "")

    spec = importlib.util.spec_from_file_location("_dyn_fs", class_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass

    cls = next(
        (obj for obj in vars(mod).values()
         if isinstance(obj, type) and issubclass(obj, FeatureSetBase) and obj is not FeatureSetBase),
        None,
    )
    if cls is None:
        raise RuntimeError(f"No FeatureSetBase subclass found in {class_path}")

    fs_cfg: dict = {"name": class_name or cls.__name__, "output_format": "name_and_value_columns"}
    if cfg.get("ffn_layer_patterns"):
        fs_cfg["ffn_layer_patterns"] = cfg["ffn_layer_patterns"]
    if cfg.get("attn_layer_patterns"):
        fs_cfg["attn_layer_patterns"] = cfg["attn_layer_patterns"]

    instance = cls(fs_cfg)
    cfg["feature_processor"] = FeatureProcessor(feature_sets=[instance])
    log.info("bridge | Injected FeatureProcessor for %s", cls.__name__)


def _create_dataset(cfg: dict) -> dict:
    _inject_feature_processor(cfg)
    from neuralsignal.automation.dataset_automation_core import create_dataset  # type: ignore
    file_paths = create_dataset(cfg, create_dataset=True) or []
    return {"file_paths": [str(p) for p in file_paths]}


def _create_s1_model(cfg: dict) -> dict:
    _inject_feature_processor(cfg)
    from neuralsignal.automation.dataset_automation_core import create_s1_model  # type: ignore
    models = create_s1_model(cfg)
    if not models:
        return {"error": "No models returned"}
    best = max(models, key=lambda m: m.metrics.get("test_auc", 0.0))
    return {
        "metrics": dict(best.metrics),
        "params": dict(best.params) if best.params else {},
        "feature_importance": best.artifacts.get("feature_importance", {}) if best.artifacts else {},
    }


_ACTIONS = {
    "create_dataset": _create_dataset,
    "create_s1_model": _create_s1_model,
}


def main() -> None:
    log.info("bridge started — action=%s", sys.argv[1] if len(sys.argv) > 1 else "?")
    if len(sys.argv) < 2 or sys.argv[1] not in _ACTIONS:
        print(json.dumps({"error": f"Usage: bridge.py <action>  known: {list(_ACTIONS)}"}))
        sys.exit(1)

    action = sys.argv[1]
    try:
        cfg = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON on stdin: {exc}"}))
        sys.exit(1)

    try:
        result = _ACTIONS[action](cfg)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
