"""neuralsignal experiment adapter.

Implements the three adapter interface functions required by the pipeline:
  - create_dataset(profile, proposal, implementation) -> dict
  - run_experiment(profile, proposal, implementation, dataset, experiment_id) -> dict
  - train_model(profile, experiment_result, dataset) -> dict

When neuralsignal's heavy ML deps (torch, transformers) are available in-process,
automation is called directly. Otherwise it falls back to the subprocess bridge.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
from pathlib import Path
from uuid import uuid4

import pandas as pd

from configs.config import get_config
from plugins.base import ResearchAdapter
from utils.logger import get_logger

log = get_logger(__name__)

_BRIDGE_SCRIPT = Path(__file__).parent / "bridge.py"

# ---------------------------------------------------------------------------
# Neuralsignal optional import
# ---------------------------------------------------------------------------
try:
    from neuralsignal.automation.dataset_automation_core import (  # type: ignore
        create_dataset as _ns_create_dataset,
        create_s1_model as _ns_create_s1_model,
    )
    from neuralsignal.core.modules.feature_sets.feature_set_base import (  # type: ignore
        FeatureSetBase as _FeatureSetBase,
    )
    from neuralsignal.core.modules.feature_sets.feature_processor import (  # type: ignore
        FeatureProcessor as _FeatureProcessor,
    )
    NS_AVAILABLE = True
except ImportError:
    _FeatureSetBase = None
    _FeatureProcessor = None
    NS_AVAILABLE = False
    log.warning("neuralsignal not importable in this process — will use subprocess bridge")


# ---------------------------------------------------------------------------
# Dynamic class loading
# ---------------------------------------------------------------------------

def _load_feature_set_class(script_path: str) -> type:
    """Dynamically import a FeatureSetBase subclass from a script file."""
    spec = importlib.util.spec_from_file_location("_dyn_fs", script_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass

    for obj in vars(mod).values():
        if (
            isinstance(obj, type)
            and _FeatureSetBase is not None
            and issubclass(obj, _FeatureSetBase)
            and obj is not _FeatureSetBase
        ):
            log.info("_load_feature_set_class | Loaded %s", obj.__name__)
            return obj

    raise RuntimeError(f"No FeatureSetBase subclass found in {script_path}")


def _build_feature_processor(proposal: dict, profile: dict, script_path: str):
    """Build a FeatureProcessor from the generated class, injecting layer patterns."""
    datasets = profile.get("datasets") or []
    ds = next((d for d in datasets if d.get("name") == proposal.get("dataset")), None) or (datasets[0] if datasets else None)

    class_name = Path(script_path).stem
    fs_cfg: dict = {"name": class_name, "output_format": "name_and_value_columns"}

    if ds:
        patterns = ds.get("layer_name_patterns") or {}
        if patterns.get("ffn"):
            fs_cfg["ffn_layer_patterns"] = patterns["ffn"]
        if patterns.get("attn"):
            fs_cfg["attn_layer_patterns"] = patterns["attn"]

    cls = _load_feature_set_class(script_path)
    instance = cls(fs_cfg)
    return _FeatureProcessor(feature_sets=[instance])


# ---------------------------------------------------------------------------
# Bridge subprocess helper
# ---------------------------------------------------------------------------

def _call_bridge(action: str, bridge_cfg: dict, timeout: int, cwd: str | None = None) -> dict:
    cfg = get_config()
    ns_python_cmd: str = cfg.neuralsignal_python
    bridge_path = str(_BRIDGE_SCRIPT.resolve())

    env = os.environ.copy()
    ns_src = str(Path(cfg.neuralsignal_src_path).resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ns_src}{os.pathsep}{existing}" if existing else ns_src

    serialisable = {k: v for k, v in bridge_cfg.items() if k != "dataframe"}
    cfg_json = json.dumps(serialisable, default=str)
    cmd = ns_python_cmd.split() + ["-u", bridge_path, action]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=cwd,
    )

    stdout_chunks: list[str] = []

    def _read_stdout():
        stdout_chunks.append(proc.stdout.read())

    def _relay_stderr():
        for line in iter(proc.stderr.readline, ""):
            stripped = line.rstrip("\n")
            if stripped:
                log.info("[bridge] %s", stripped)

    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err = threading.Thread(target=_relay_stderr, daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.stdin.write(cfg_json)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    t_out.join(timeout=timeout)
    t_err.join(timeout=10)
    proc.wait(timeout=10)

    stdout_data = "".join(stdout_chunks)

    if proc.returncode != 0:
        raise RuntimeError(f"Bridge {action!r} exited {proc.returncode}: {stdout_data[:500]}")

    # Find last JSON line (neuralsignal may append ANSI codes)
    json_line = next(
        (line for line in reversed(stdout_data.splitlines()) if line.lstrip().startswith(("{", "["))),
        None,
    )
    if json_line is None:
        raise RuntimeError(f"Bridge {action!r} produced no JSON output.\nstdout={stdout_data[:500]!r}")

    result = json.loads(json_line)
    if "error" in result:
        raise RuntimeError(f"Bridge {action!r} error: {result['error']}")
    return result


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class NeuralSignalAdapter(ResearchAdapter):
    """ResearchAdapter implementation for NeuralSignal feature experiments."""

    def validate_environment(self, profile: dict) -> dict:
        cfg = get_config()
        return {
            "available_in_process": NS_AVAILABLE,
            "bridge_script": str(_BRIDGE_SCRIPT),
            "neuralsignal_python": cfg.neuralsignal_python,
            "neuralsignal_src_path": cfg.neuralsignal_src_path,
        }

    def build_context(self, profile: dict, state: dict) -> dict:
        return {
            "datasets": profile.get("datasets") or [],
            "base_classes": profile.get("base_classes") or [],
            "evaluation": profile.get("evaluation") or {},
        }

    def prepare_experiment(
        self,
        profile: dict,
        proposal: dict,
        implementation: dict | None,
        state: dict,
    ) -> dict | None:
        dataset = create_dataset(profile, proposal, implementation)
        if dataset:
            dataset.setdefault("artifact_type", "dataset")
        return dataset

    def execute_experiment(
        self,
        profile: dict,
        proposal: dict,
        implementation: dict | None,
        artifact: dict | None,
        experiment_id: str,
        state: dict,
    ) -> dict | None:
        if artifact is None:
            log.error("NeuralSignalAdapter.execute_experiment | No dataset artifact")
            return None

        result = run_experiment(profile, proposal, implementation, artifact, experiment_id)
        if result is None:
            return None

        model = train_model(profile, result, artifact)
        if model:
            model.setdefault("experiment_id", experiment_id)
            model.setdefault("proposal_name", result.get("proposal_name"))
            result["model"] = model
            result["metrics"] = {**(result.get("metrics") or {}), **(model.get("metrics") or {})}
            result["feature_importance"] = model.get("feature_importance") or result.get("feature_importance") or {}

        result.setdefault("artifact", _serialisable_artifact_summary(artifact))
        return result

    def summarize_result(self, profile: dict, result: dict) -> dict:
        metrics = result.get("metrics") or {}
        primary = (profile.get("evaluation") or {}).get("primary_metric")
        return {
            "proposal_name": result.get("proposal_name"),
            "primary_metric": primary,
            "primary_metric_value": metrics.get(primary) if primary else None,
            "metrics": metrics,
        }


def get_adapter() -> NeuralSignalAdapter:
    return NeuralSignalAdapter()


def _serialisable_artifact_summary(artifact: dict) -> dict:
    return {
        k: v
        for k, v in artifact.items()
        if not k.startswith("_") and k != "columns"
    }


def create_dataset(profile: dict, proposal: dict, implementation: dict | None) -> dict | None:
    """Create a feature dataset for the given proposal.

    Returns a dataset entry dict, or None on failure.
    """
    cfg = get_config()
    timeout = cfg.experiment_timeout_seconds

    # Resolve dataset metadata from profile
    datasets = profile.get("datasets") or []
    dataset_name = proposal.get("dataset") or (datasets[0]["name"] if datasets else "")
    ds_meta = next((d for d in datasets if d.get("name") == dataset_name), datasets[0] if datasets else {})

    detector_name = proposal.get("detector") or (ds_meta.get("available_detectors") or ["hallucination"])[0]
    app_name = ds_meta.get("storage", {}).get("application_name", "")
    sub_app_name = ds_meta.get("storage", {}).get("sub_application_name", "")

    feature_set_name = proposal.get("name", "unknown")
    experiments_dir = Path(cfg.experiments_dir) / (profile.get("name", "default")) / "datasets"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    automation_cfg = {
        "run_data_collection": False,
        "indirect_config": {"indirect_model": "", "indirect_batch_size": 1, "quantization": "int8", "device": "cuda:0"},
        "indirect_instrumentation_config": {},
        "create_dataset": True,
        "create_s1_model": False,
        "detector_names": [detector_name],
        "dataset": dataset_name,
        "application_name": app_name,
        "sub_application_name": sub_app_name,
        "backend_config": {
            "backend_type": "neuralsignal_v1",
            "mongo_url": cfg.mongo_url,
            "mlflow_uri": "http://hp.lan:8899/",
            "mlflow_register_model": False,
        },
        "zone_size": proposal.get("hyperparameters", {}).get("zone_size", 1024),
        "row_limit": 0,
        "write_to_file": True,
        "build_in_memory": False,
        "use_gt_as_target": True,
        "query": proposal.get("mongo_query") or {},
        "file_out": f"{feature_set_name}_{detector_name}",
    }

    script_path = (implementation or {}).get("script_path", "")

    if NS_AVAILABLE and script_path and Path(script_path).exists():
        fp = _build_feature_processor(proposal, profile, script_path)
        automation_cfg["feature_processor"] = fp
        try:
            file_paths = _ns_create_dataset(automation_cfg, create_dataset=True)
        except Exception as exc:
            log.error("neuralsignal.create_dataset failed: %s", exc, exc_info=True)
            return None
    else:
        if script_path and Path(script_path).exists():
            automation_cfg["feature_set_class_path"] = str(Path(script_path).resolve())
            automation_cfg["feature_set_class_name"] = Path(script_path).stem
            # Pass layer patterns for bridge to inject
            layer_patterns = ds_meta.get("layer_name_patterns") or {}
            if layer_patterns.get("ffn"):
                automation_cfg["ffn_layer_patterns"] = layer_patterns["ffn"]
            if layer_patterns.get("attn"):
                automation_cfg["attn_layer_patterns"] = layer_patterns["attn"]

        try:
            result = _call_bridge("create_dataset", automation_cfg, timeout, cwd=str(experiments_dir.resolve()))
            file_paths = [str(experiments_dir / p) for p in result.get("file_paths", [])]
        except Exception as exc:
            log.error("bridge.create_dataset failed: %s", exc, exc_info=True)
            return None

    if not file_paths:
        log.error("create_dataset | No output files returned")
        return None

    try:
        df = pd.read_csv(file_paths[0])
    except Exception as exc:
        log.error("create_dataset | Failed to read CSV %s: %s", file_paths[0], exc)
        return None

    log.info("create_dataset | Dataset ready — shape=%s, file=%s", df.shape, file_paths[0])
    return {
        "dataset_id": str(uuid4()),
        "proposal_name": feature_set_name,
        "feature_set_name": feature_set_name,
        "file_path": file_paths[0],
        "rows": len(df),
        "columns": list(df.columns),
        "detector": detector_name,
        "dataset_name": dataset_name,
        "_dataframe": df,  # in-memory shortcut for train_model
    }


def run_experiment(
    profile: dict,
    proposal: dict,
    implementation: dict | None,
    dataset: dict,
    experiment_id: str,
) -> dict | None:
    """Run the experiment — for neuralsignal this is dataset creation (already done).

    Returns a result dict with raw metrics placeholder.
    """
    # For neuralsignal, the heavy lifting is in create_dataset + train_model.
    # run_experiment just records that the dataset was created successfully.
    return {
        "experiment_id": experiment_id,
        "proposal_name": dataset.get("proposal_name"),
        "dataset_id": dataset.get("dataset_id"),
        "metrics": {},  # populated by train_model
        "feature_importance": {},
    }


def train_model(
    profile: dict,
    experiment_result: dict,
    dataset: dict | None,
) -> dict | None:
    """Train an XGBoost S1 model and return metrics."""
    if dataset is None:
        log.error("train_model | No dataset provided")
        return None

    cfg = get_config()
    timeout = cfg.experiment_timeout_seconds
    proposal_name = dataset.get("proposal_name", "unknown")
    detector_name = dataset.get("detector", "hallucination")
    dataset_name = dataset.get("dataset_name", "")
    experiment_id = experiment_result.get("experiment_id", str(uuid4()))

    # Resolve dataset metadata
    datasets = profile.get("datasets") or []
    ds_meta = next((d for d in datasets if d.get("name") == dataset_name), datasets[0] if datasets else {})
    app_name = ds_meta.get("storage", {}).get("application_name", "")
    sub_app_name = ds_meta.get("storage", {}).get("sub_application_name", "")

    df: pd.DataFrame | None = dataset.get("_dataframe")
    if df is None:
        file_path = dataset.get("file_path", "")
        if not file_path or not Path(file_path).exists():
            log.error("train_model | No dataframe or file_path available")
            return None
        df = pd.read_csv(file_path)

    s1_cfg = {
        "application_name": app_name,
        "sub_application_name": sub_app_name,
        "indirect_config": {"indirect_model": "", "quantization": "int8"},
        "indirect_instrumentation_config": {},
        "model_name": f"{proposal_name}_{experiment_id[:8]}",
        "dataset": dataset_name,
        "zone_size": 1024,
        "use_full_zone_names": False,
        "file_out": f"{proposal_name}_{detector_name}",
        "detector_names": [detector_name],
        "modeling_row_limits": [0],
        "optimization_metric": (profile.get("evaluation") or {}).get("primary_metric", "auc"),
        "max_evals": 20,
        "test_set_size": 0.33,
        "seed": 42,
        "run_cross_validation": True,
        "cv_folds": 3,
        "create_reduced_feature_model": False,
        "save_to_backend": False,
        "backend_config": {
            "backend_type": "neuralsignal_v1",
            "mongo_url": cfg.mongo_url,
            "mlflow_uri": "http://hp.lan:8899/",
            "mlflow_register_model": False,
        },
    }

    if NS_AVAILABLE:
        s1_cfg["dataframe"] = df
        s1_cfg["dataset_path"] = "n/a"
        try:
            models = _ns_create_s1_model(s1_cfg)
        except Exception as exc:
            log.error("train_model | ns_create_s1_model failed: %s", exc, exc_info=True)
            return None
        if not models:
            return None
        best = max(models, key=lambda m: m.metrics.get("test_auc", 0.0))
        return {
            "model_id": str(uuid4()),
            "metrics": dict(best.metrics),
            "params": dict(best.params) if best.params else {},
            "feature_importance": best.artifacts.get("feature_importance", {}) if best.artifacts else {},
        }

    else:
        import tempfile
        tmp_csv = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as fh:
                tmp_csv = fh.name
                df.to_csv(fh, index=False)
            s1_cfg["dataset_path"] = tmp_csv
            result = _call_bridge("create_s1_model", s1_cfg, timeout)
        except Exception as exc:
            log.error("train_model | bridge failed: %s", exc, exc_info=True)
            return None
        finally:
            if tmp_csv:
                Path(tmp_csv).unlink(missing_ok=True)

        return {
            "model_id": str(uuid4()),
            "metrics": result.get("metrics", {}),
            "params": result.get("params", {}),
            "feature_importance": result.get("feature_importance", {}),
        }
