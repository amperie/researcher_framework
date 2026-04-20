"""Subprocess bridge for neuralsignal automation calls.

This script is invoked by ns_experiment_runner_node when the neuralsignal SDK
cannot be imported directly into the NeuralSignalResearcher process (e.g. because
neuralsignal's heavy ML deps — torch, transformers — are not installed in this venv).

Usage (called by _call_bridge in ns_experiment_runner.py):
    <neuralsignal_python> utils/ns_automation_bridge.py create_dataset
    <neuralsignal_python> utils/ns_automation_bridge.py create_s1_model

JSON config is read from stdin; result is written as JSON to stdout.

Configure via .env:
    NEURALSIGNAL_PYTHON=../neuralsignal/.venv/Scripts/python.exe   (Windows)
    NEURALSIGNAL_PYTHON=../neuralsignal/.venv/bin/python            (Linux/Mac)

PYTHONPATH must include the neuralsignal source root (ns_experiment_runner handles this).
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


def _inject_feature_processor(cfg: dict) -> None:
    """If cfg contains 'feature_set_class_path', load the class and inject a
    FeatureProcessor into cfg['feature_processor'].  No-op otherwise."""
    if "feature_set_class_path" not in cfg:
        return

    from neuralsignal.core.modules.feature_sets.feature_set_base import (  # type: ignore
        FeatureSetBase,
    )
    from neuralsignal.core.modules.feature_sets.feature_processor import (  # type: ignore
        FeatureProcessor,
    )

    class_path = cfg["feature_set_class_path"]
    feature_set_name = cfg.get("feature_set_class_name", "")

    spec = importlib.util.spec_from_file_location("_dyn_fs", class_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass

    cls = next(
        (
            obj for obj in vars(mod).values()
            if isinstance(obj, type)
            and issubclass(obj, FeatureSetBase)
            and obj is not FeatureSetBase
        ),
        None,
    )
    if cls is None:
        raise RuntimeError(
            f"No FeatureSetBase subclass found in {class_path}"
        )

    fs_cfg: dict = {"name": feature_set_name, "output_format": "name_and_value_columns"}
    if cfg.get("ffn_layer_patterns"):
        fs_cfg["ffn_layer_patterns"] = cfg["ffn_layer_patterns"]
    if cfg.get("attn_layer_patterns"):
        fs_cfg["attn_layer_patterns"] = cfg["attn_layer_patterns"]
    instance = cls(fs_cfg)
    cfg["feature_processor"] = FeatureProcessor(feature_sets=[instance])
    logging.getLogger(__name__).info(
        "ns_automation_bridge | Injected FeatureProcessor for class %s", cls.__name__,
    )


def _create_dataset(cfg: dict) -> dict:
    _inject_feature_processor(cfg)
    from neuralsignal.automation.dataset_automation_core import (  # type: ignore
        create_dataset,
    )
    file_paths = create_dataset(cfg, create_dataset=True) or []
    return {"file_paths": [str(p) for p in file_paths]}


def _create_s1_model(cfg: dict) -> dict:
    _inject_feature_processor(cfg)
    from neuralsignal.automation.dataset_automation_core import (  # type: ignore
        create_s1_model,
    )
    models = create_s1_model(cfg)
    if not models:
        return {"error": "No models returned from create_s1_model"}

    best = max(models, key=lambda m: m.metrics.get("test_auc", 0.0))
    return {
        "metrics": dict(best.metrics),
        "params": dict(best.params) if best.params else {},
        "feature_importance": (
            best.artifacts.get("feature_importance", {})
            if best.artifacts else {}
        ),
    }


_ACTIONS = {
    "create_dataset": _create_dataset,
    "create_s1_model": _create_s1_model,
}


def main() -> None:
    print(f"[ns_automation_bridge] started — action={sys.argv[1] if len(sys.argv) > 1 else '?'}", file=sys.stderr, flush=True)
    if len(sys.argv) < 2 or sys.argv[1] not in _ACTIONS:
        known = ", ".join(_ACTIONS)
        print(json.dumps({"error": f"Usage: bridge.py <action>  (known: {known})"}))
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
