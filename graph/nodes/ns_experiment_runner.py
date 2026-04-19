"""NS Experiment Runner Node — invokes neuralsignal automation directly.

For each feature proposal this node:
  1. Resolves the dataset (DatasetManager cache hit, or automation create_dataset)
  2. Trains an XGBoost S1 model via S1Trainer, optimising for AUC
  3. Logs results to MLflow (experiment from config.yaml experiment_runner section)
  4. Upserts a text embedding to ChromaDB for semantic experiment comparison
  5. Inserts the full result record into MongoDB (agent_experiments database)

Configuration keys (all read from config.yaml `experiment_runner` section):
  mlflow_experiment, agent_experiments_db, modeling_row_limit, max_evals,
  optimization_metric, default_detector, default_dataset, dataset_row_limit
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import pandas as pd
import pymongo

from configs.config import get_config
from graph.state import ResearchState
from tools.chroma_tool import ChromaStore
from utils.dataset_manager import DatasetManager
from utils.dataset_registry import get_dataset_or_first
from utils.logger import get_logger
from utils.utils import load_yaml_section

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Neuralsignal SDK — optional import so unit tests run without the full SDK
# ---------------------------------------------------------------------------
try:
    from neuralsignal.automation.dataset_automation_core import (  # type: ignore
        create_dataset as ns_create_dataset,
        create_s1_model as ns_create_s1_model,
    )
    from neuralsignal.core.modules.feature_sets.feature_set_base import (  # type: ignore
        FeatureSetBase as _FeatureSetBase,
    )
    from neuralsignal.core.modules.feature_sets.feature_processor import (  # type: ignore
        FeatureProcessor as _FeatureProcessor,
    )
    NS_AVAILABLE = True
except ImportError:
    _FeatureSetBase = None  # type: ignore[assignment,misc]
    _FeatureProcessor = None  # type: ignore[assignment,misc]
    NS_AVAILABLE = False
    log.warning(
        "ns_experiment_runner | neuralsignal SDK not importable in this process — "
        "will delegate to ns_automation_bridge subprocess"
    )

# ---------------------------------------------------------------------------
# Feature-set constants
# ---------------------------------------------------------------------------

_REGISTERED_FS: frozenset[str] = frozenset({"zones", "logit-lens", "T-F-diff", "layer_distribution"})
_FS_CACHE_DIR = Path("dev/experiments/featuresets")


# ---------------------------------------------------------------------------
# Subprocess bridge helpers
# ---------------------------------------------------------------------------

_BRIDGE_SCRIPT = Path(__file__).parent.parent.parent / "utils" / "ns_automation_bridge.py"


class _S1Result:
    """Minimal S1Model-compatible result returned from the subprocess bridge."""

    def __init__(self, metrics: dict, params: dict, feature_importance: dict) -> None:
        self.metrics = metrics
        self.params = params
        self.artifacts = {"feature_importance": feature_importance}


def _call_bridge(action: str, bridge_cfg: dict, cfg, cwd: str | None = None) -> dict:
    """Invoke ns_automation_bridge.py as a subprocess.

    Injects neuralsignal_src_path into PYTHONPATH so the bridge can import the SDK.
    Returns the parsed JSON result dict.  Raises RuntimeError on non-zero exit.

    Args:
        cwd: Working directory for the subprocess.  When set, relative file paths
             in bridge_cfg (e.g. file_out) are resolved against this directory.
    """
    ns_python_cmd: str = cfg.neuralsignal_python  # e.g. "python" or "uv run python"
    bridge_path = str(_BRIDGE_SCRIPT.resolve())

    env = os.environ.copy()
    ns_src = str(Path(cfg.neuralsignal_src_path).resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ns_src}{os.pathsep}{existing}" if existing else ns_src

    # Strip non-serialisable values (DataFrames, etc.) before sending to subprocess
    serialisable_cfg = {k: v for k, v in bridge_cfg.items() if k != "dataframe"}
    cfg_json = json.dumps(serialisable_cfg, default=str)

    # -u: unbuffered so log lines arrive immediately rather than in blocks
    cmd = ns_python_cmd.split() + ["-u", bridge_path, action]
    log.info("ns_experiment_runner | bridge cmd: %s  cwd=%s", cmd, cwd)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )

    # Use separate threads for stdout and stderr so they don't block each other
    # and communicate() doesn't compete with the relay for the stderr pipe.
    stdout_chunks: list[str] = []

    def _read_stdout() -> None:
        assert proc.stdout is not None
        stdout_chunks.append(proc.stdout.read())

    def _relay_stderr() -> None:
        assert proc.stderr is not None
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

    timeout = cfg.experiment_timeout_seconds
    t_out.join(timeout=timeout)
    t_err.join(timeout=10)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(
            f"Bridge action={action!r} timed out after {timeout}s"
        )

    stdout_data = "".join(stdout_chunks)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Bridge action={action!r} exited {proc.returncode}: {stdout_data.strip()[:1000]}"
        )

    # The bridge may emit ANSI escape codes or other non-JSON trailing output
    # (e.g. a colorised logger writing \x1b[0m after the result line).
    # Find the last line that starts with '{' or '[' — that's the JSON result.
    json_line = next(
        (
            line
            for line in reversed(stdout_data.splitlines())
            if line.lstrip().startswith(("{", "["))
        ),
        None,
    )
    if json_line is None:
        raise RuntimeError(
            f"Bridge action={action!r} produced no JSON line in stdout.\n"
            f"stdout={stdout_data[:500]!r}"
        )

    try:
        result = json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Bridge action={action!r} returned invalid JSON: {exc}\n"
            f"line={json_line[:500]!r}"
        ) from exc

    if "error" in result:
        raise RuntimeError(f"Bridge action={action!r} reported error: {result['error']}")

    return result


# ---------------------------------------------------------------------------
# Dynamic feature-set class helpers
# ---------------------------------------------------------------------------

def _get_or_generate_feature_set_script(
    proposal: dict,
    state: dict,
    cfg,
) -> Path:
    """Return path to a FeatureSetBase subclass script for the given proposal.

    Resolution order:
      1. Disk cache at dev/experiments/featuresets/<feature_set_name>.py
      2. Matching entry in state['generated_scripts']
      3. Generate via code_generation helpers (Claude Code CLI)
    """
    feature_set_name: str = proposal.get("feature_set_name") or "unknown_fs"

    # 1. Disk cache
    cache_path = _FS_CACHE_DIR / f"{feature_set_name}.py"
    if cache_path.exists():
        log.info("_get_or_generate_feature_set_script | Cache hit: %s", cache_path)
        return cache_path

    # 2. state['generated_scripts'] (populated by code_generation_node)
    for script in (state.get("generated_scripts") or []):
        ec = script.get("experiment_config") or {}
        if ec.get("feature_set_name") == feature_set_name:
            sp = Path(script["script_path"])
            if sp.exists():
                log.info(
                    "_get_or_generate_feature_set_script | Found in generated_scripts: %s", sp,
                )
                return sp

    # 3. Generate via Claude Code CLI (reuse code_generation_node helpers)
    log.info(
        "_get_or_generate_feature_set_script | Generating class for %r via Claude Code CLI",
        feature_set_name,
    )
    from graph.nodes.code_generation import (  # noqa: PLC0415
        _run_streaming,
        _parse_claude_output,
        _strip_fences,
    )
    from configs.prompts import CODE_GENERATION  # noqa: PLC0415

    direction = state.get("research_direction", "")
    prompt = (
        f"{CODE_GENERATION}\n\n"
        f"{'=' * 63}\n"
        f"TASK\n"
        f"{'=' * 63}\n\n"
        f"Research direction: {direction}\n\n"
        f"Experiment ID (embed in results dict): {feature_set_name}\n\n"
        f"FeatureSet proposal:\n{json.dumps(proposal, indent=2)}"
    )
    command = cfg.claude_command.split() + ["--print", "--output-format", "json"]

    try:
        result = _run_streaming(command, prompt, cfg.code_generation_timeout_seconds)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Claude Code CLI not found ({cfg.claude_command!r}). "
            "Cannot generate feature set class."
        ) from exc

    raw = _parse_claude_output(result.stdout)
    code = _strip_fences(raw)
    if not code:
        raise RuntimeError(
            f"Code generation returned empty output for feature set {feature_set_name!r}"
        )

    _FS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(code, encoding="utf-8")
    log.info(
        "_get_or_generate_feature_set_script | Saved %d lines → %s",
        len(code.splitlines()), cache_path,
    )
    return cache_path


def _load_feature_set_class(script_path: Path) -> type:
    """Dynamically import a FeatureSetBase subclass from a script file.

    Works with full standalone test scripts (the if __name__ == '__main__' block
    is not executed on import) or class-only files.
    """
    spec = importlib.util.spec_from_file_location("_dyn_fs", script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass  # standalone scripts may call sys.exit() — safe to ignore on import

    for obj in vars(mod).values():
        if (
            isinstance(obj, type)
            and _FeatureSetBase is not None
            and issubclass(obj, _FeatureSetBase)
            and obj is not _FeatureSetBase
        ):
            log.info(
                "_load_feature_set_class | Loaded %s from %s", obj.__name__, script_path,
            )
            return obj

    raise RuntimeError(
        f"No FeatureSetBase subclass found in {script_path}. "
        "The generated script may be malformed."
    )


def _build_feature_processor(proposal: dict, state: dict, cfg) -> "_FeatureProcessor":
    """Generate/load a FeatureSetBase subclass and wrap it in a FeatureProcessor.

    Only call when NS_AVAILABLE=True.
    """
    if not NS_AVAILABLE or _FeatureProcessor is None:
        raise RuntimeError(
            "Cannot build FeatureProcessor: neuralsignal not importable in this process"
        )

    feature_set_name: str = proposal.get("feature_set_name") or "unknown_fs"
    rcfg = _runner_cfg()
    ds_entry = get_dataset_or_first(
        proposal.get("dataset") or rcfg.get("default_dataset")
    )

    fs_cfg: dict = {"name": feature_set_name, "output_format": "name_and_value_columns"}
    if ds_entry:
        patterns = ds_entry.get("layer_name_patterns") or {}
        if patterns.get("ffn"):
            fs_cfg["ffn_layer_patterns"] = patterns["ffn"]
        if patterns.get("attn"):
            fs_cfg["attn_layer_patterns"] = patterns["attn"]
        log.debug(
            "_build_feature_processor | Injected layer patterns from dataset %r — "
            "ffn=%d, attn=%d",
            ds_entry.get("name"),
            len(fs_cfg.get("ffn_layer_patterns", [])),
            len(fs_cfg.get("attn_layer_patterns", [])),
        )

    script_path = _get_or_generate_feature_set_script(proposal, state, cfg)
    cls = _load_feature_set_class(script_path)

    try:
        instance = cls(fs_cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate {cls.__name__}: {exc}"
        ) from exc

    return _FeatureProcessor(feature_sets=[instance])


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _runner_cfg() -> dict:
    return load_yaml_section("experiment_runner") or {}


# ---------------------------------------------------------------------------
# Scan-source resolution
# ---------------------------------------------------------------------------

def _get_scan_source(proposal: dict, rcfg: dict) -> tuple[str, str]:
    """Return (application_name, sub_application_name) for this proposal.

    Looks up proposal['dataset'] (or rcfg['default_dataset']) in the dataset
    registry (configs/datasets.yaml).  Falls back to the first registry entry
    if the name is not found, or to a hard-coded sentinel if the registry is empty.
    """
    dataset_name: str | None = proposal.get("dataset") or rcfg.get("default_dataset")
    entry = get_dataset_or_first(dataset_name)

    if entry:
        return entry["application_name"], entry["sub_application_name"]

    # Registry is empty — hard fallback
    log.warning("_get_scan_source | Dataset registry is empty; using hard fallback")
    return "ns_researcher", proposal.get("feature_set_name") or "default"


# ---------------------------------------------------------------------------
# Feature-set config mapping
# ---------------------------------------------------------------------------

def _build_feature_set_config(proposal: dict) -> dict:
    """Map a feature_proposal dict to a neuralsignal feature_set_config dict.

    Uses:
      - proposal['feature_set_name']  → config['name']
      - proposal['zone_config']       → target_zone_size, layer_names_to_include
      - proposal['scan_fields_used']  → field_to_process
    """
    zone_config = proposal.get("zone_config") or {}
    scan_fields = proposal.get("scan_fields_used") or []

    # Determine field_to_process from scan_fields_used, default "outputs"
    field_to_process = "outputs"
    for f in scan_fields:
        if f in ("outputs", "inputs"):
            field_to_process = f
            break

    # Zone size: prefer zone_config["zone_size"], then "default", then 1024
    if isinstance(zone_config, dict):
        zone_size = (
            zone_config.get("zone_size")
            or zone_config.get("default")
            or 1024
        )
        layer_patterns = (
            zone_config.get("layer_names_to_include")
            or zone_config.get("layers")
            or []
        )
    else:
        zone_size = 1024
        layer_patterns = []

    # Map the proposal's feature_set_name to a neuralsignal registered name where
    # possible.  Unregistered names are returned as-is — the caller checks
    # _REGISTERED_FS membership and generates/loads the class dynamically.
    proposed_name = proposal.get("feature_set_name") or ""
    if proposed_name in _REGISTERED_FS:
        registered_name = proposed_name
    elif "logit" in proposed_name.lower():
        registered_name = "logit-lens"
    elif "diff" in proposed_name.lower() or "true" in proposed_name.lower():
        registered_name = "T-F-diff"
    elif "distribution" in proposed_name.lower() or "layer" in proposed_name.lower():
        registered_name = "layer_distribution"
    else:
        registered_name = proposed_name  # novel feature set — will be generated dynamically

    config: dict = {
        "name": registered_name,
        "target_zone_size": {"default": int(zone_size)},
        "field_to_process": field_to_process,
    }
    if layer_patterns:
        config["layer_names_to_include"] = list(layer_patterns)

    return config


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------

def _balance_dataframe(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Return a class-balanced DataFrame (equal rows per target class).

    Args:
        df:          DataFrame with a binary 'target' column (0 / 1).
        sample_size: Total desired rows (0 = keep all available balanced rows).
                     Each class gets at most floor(sample_size / 2) rows.

    Returns:
        Shuffled DataFrame with equal class counts, index reset.
    """
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]

    per_class = (
        min(len(pos), len(neg))
        if sample_size == 0
        else min(len(pos), len(neg), sample_size // 2)
    )

    log.info(
        "_balance_dataframe | before: pos=%d neg=%d → per_class=%d",
        len(pos), len(neg), per_class,
    )

    balanced = pd.concat([
        pos.sample(n=per_class, random_state=42),
        neg.sample(n=per_class, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(
        "_balance_dataframe | after: rows=%d (pos=%d neg=%d)",
        len(balanced),
        int((balanced["target"] == 1).sum()),
        int((balanced["target"] == 0).sum()),
    )
    return balanced


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

def _resolve_datasets(
    proposal: dict,
    dm: DatasetManager,
    cfg,
    rcfg: dict,
    state: dict,
) -> tuple[pd.DataFrame | None, list[dict], int]:
    """Return (df, dataset_entries, cache_hit_count).

    Checks DatasetManager first; falls back to neuralsignal automation if needed.
    The returned DataFrame always includes a 'target' column (binary label).
    """
    fs_config = _build_feature_set_config(proposal)
    feature_set_name: str = fs_config["name"]
    detector_name: str = (
        proposal.get("detector_name") or rcfg.get("default_detector", "hallucination")
    )
    dataset_name: str = (
        proposal.get("dataset") or rcfg.get("default_dataset", "HaluBench")
    )

    # Stable cache identity (volatile keys excluded by DatasetManager.config_hash)
    exp_config = {
        "feature_set_name": feature_set_name,
        "detector_name": detector_name,
        "dataset": dataset_name,
        "zone_config": proposal.get("zone_config") or {},
        "field_to_process": fs_config.get("field_to_process", "outputs"),
    }

    # --- Cache hit ---
    cached = dm.find_cached(feature_set_name, exp_config)
    if cached:
        log.info(
            "ns_experiment_runner | Dataset cache hit — feature_set=%r, id=%s",
            feature_set_name, cached["dataset_id"],
        )
        df = dm.load_dataset(cached)
        if rcfg.get("balanced_dataset", False):
            df = _balance_dataframe(df, rcfg.get("balanced_sample_size", 0))
        return df, [cached], 1

    # --- Cache miss: create via automation ---
    experiments_dir = Path(cfg.experiments_dir) / "datasets"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    app_name, sub_app_name = _get_scan_source(proposal, rcfg)
    log.info(
        "ns_experiment_runner | Scan source — application_name=%r, sub_application_name=%r",
        app_name, sub_app_name,
    )

    automation_cfg = {
        "run_data_collection": False,
        "indirect_config": {
            "indirect_model": "",
            "indirect_batch_size": 1,
            "quantization": "int8",
            "device": "cuda:0",
        },
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
        "zone_size": fs_config["target_zone_size"].get("default", 1024),
        "row_limit": rcfg.get("dataset_row_limit", 0),
        "dataset_row_limit": rcfg.get("dataset_row_limit", 0),
        "write_to_file": True,
        "build_in_memory": False,
        "use_gt_as_target": True,
        "query": proposal.get("mongo_query") or {},
        # flat filename only — neuralsignal's string_to_filename() sanitises paths by
        # replacing separators with underscores, so we pass a bare name and run the
        # bridge with cwd=experiments_dir so the file lands in the right place.
        "file_out": f"{feature_set_name}_{detector_name}",
    }

    # Wire up the feature set — registered names use feature_set_configs directly;
    # novel names load/generate a FeatureSetBase subclass dynamically.
    if feature_set_name in _REGISTERED_FS:
        automation_cfg["feature_set_configs"] = [fs_config]
    elif NS_AVAILABLE:
        try:
            fp = _build_feature_processor(proposal, state, cfg)
            automation_cfg["feature_processor"] = fp
            log.info(
                "ns_experiment_runner | Using generated feature set class: %r", feature_set_name,
            )
        except Exception as exc:
            log.error(
                "ns_experiment_runner | Failed to build feature processor for %r: %s",
                feature_set_name, exc, exc_info=True,
            )
            return None, [], 0
    else:
        try:
            script_path = _get_or_generate_feature_set_script(proposal, state, cfg)
            automation_cfg["feature_set_class_path"] = str(script_path.resolve())
            automation_cfg["feature_set_class_name"] = proposal.get("class_name", "")
            # Pass layer patterns so the bridge can inject them into fs_cfg
            ds_entry = get_dataset_or_first(
                proposal.get("dataset") or rcfg.get("default_dataset")
            )
            if ds_entry:
                patterns = ds_entry.get("layer_name_patterns") or {}
                if patterns.get("ffn"):
                    automation_cfg["ffn_layer_patterns"] = patterns["ffn"]
                if patterns.get("attn"):
                    automation_cfg["attn_layer_patterns"] = patterns["attn"]
            log.info(
                "ns_experiment_runner | Passing generated feature set to bridge: %r → %s",
                feature_set_name, script_path,
            )
        except Exception as exc:
            log.error(
                "ns_experiment_runner | Failed to generate feature set script for %r: %s",
                feature_set_name, exc, exc_info=True,
            )
            return None, [], 0

    log.info(
        "ns_experiment_runner | Creating dataset — feature_set=%r, detector=%r, dataset=%r",
        feature_set_name, detector_name, dataset_name,
    )

    experiments_dir.mkdir(parents=True, exist_ok=True)
    bridge_cwd = str(experiments_dir.resolve())

    try:
        if NS_AVAILABLE:
            file_paths = ns_create_dataset(automation_cfg, create_dataset=True)
        else:
            log.info("ns_experiment_runner | Using subprocess bridge for create_dataset")
            result = _call_bridge("create_dataset", automation_cfg, cfg, cwd=bridge_cwd)
            # Bridge returns paths relative to its cwd (experiments_dir) — make absolute
            file_paths = [
                str(experiments_dir / p) for p in result.get("file_paths", [])
            ]
    except Exception as exc:
        log.error("ns_experiment_runner | create_dataset failed: %s", exc, exc_info=True)
        return None, [], 0

    if not file_paths:
        log.error("ns_experiment_runner | create_dataset returned no output files")
        return None, [], 0

    try:
        df = pd.read_csv(file_paths[0])
    except Exception as exc:
        log.error("ns_experiment_runner | Failed to read dataset CSV %s: %s", file_paths[0], exc, exc_info=True)
        return None, [], 0

    log.info(
        "ns_experiment_runner | Dataset created — shape=%s, file=%s",
        df.shape, file_paths[0],
    )

    if rcfg.get("balanced_dataset", False):
        df = _balance_dataframe(df, rcfg.get("balanced_sample_size", 0))

    # Save all columns (including 'target') to DatasetManager so that cache hits
    # return a complete DataFrame ready for S1 training.
    all_cols = list(df.columns)
    rows = df[all_cols].values.tolist()
    scan_ids = df.index.astype(str).tolist()
    new_exp_id = str(uuid4())

    try:
        entry = dm.save_dataset(
            new_exp_id, feature_set_name, exp_config, all_cols, rows, scan_ids,
        )
        log.info("ns_experiment_runner | Dataset saved to registry — id=%s", entry["dataset_id"])
    except Exception as exc:
        log.warning("ns_experiment_runner | DatasetManager save failed: %s", exc, exc_info=True)
        entry = {"dataset_id": new_exp_id}

    return df, [entry], 0


# ---------------------------------------------------------------------------
# S1 model training
# ---------------------------------------------------------------------------

def _train_s1_model(
    df: pd.DataFrame,
    proposal: dict,
    experiment_id: str,
    cfg,
    rcfg: dict,
    state: dict,
):
    """Train XGBoost S1 model on df, return best S1Model (or _S1Result) or None on failure."""
    fs_config = _build_feature_set_config(proposal)
    feature_set_name: str = fs_config["name"]
    detector_name: str = (
        proposal.get("detector_name") or rcfg.get("default_detector", "hallucination")
    )
    dataset_name: str = (
        proposal.get("dataset") or rcfg.get("default_dataset", "HaluBench")
    )
    proposal_name: str = proposal.get("name") or feature_set_name

    feature_cols = [c for c in df.columns if c != "target"]
    log.info(
        "ns_experiment_runner | Training S1 model — proposal=%r, rows=%d, features=%d",
        proposal_name, len(df), len(feature_cols),
    )

    app_name, sub_app_name = _get_scan_source(proposal, rcfg)
    experiments_dir = Path(cfg.experiments_dir) / "datasets"

    s1_cfg_base = {
        # Required by S1Trainer.__init__
        "application_name": app_name,
        "sub_application_name": sub_app_name,
        "indirect_config": {"indirect_model": "", "quantization": "int8"},
        "indirect_instrumentation_config": {},
        "model_name": f"{feature_set_name}_{experiment_id[:8]}",
        # Dataset / naming keys consumed by create_s1_model
        "dataset": dataset_name,
        "zone_size": fs_config["target_zone_size"].get("default", 1024),
        "use_full_zone_names": False,
        "file_out": f"{feature_set_name}_{detector_name}",
        # Automation iteration keys
        "detector_names": [detector_name],
        "modeling_row_limits": [rcfg.get("modeling_row_limit", 10000)],
        # Training hyperparameters
        "optimization_metric": rcfg.get("optimization_metric", "auc"),
        "max_evals": rcfg.get("max_evals", 20),
        "test_set_size": 0.33,
        "seed": 42,
        "run_cross_validation": True,
        "cv_folds": 3,
        "create_reduced_feature_model": False,
        # Skip backend persistence — we handle it
        "save_to_backend": False,
        "description": proposal.get("description", ""),
        "tags": {
            "proposal_name": proposal_name,
            "experiment_id": experiment_id,
            "source": "ns_researcher",
        },
    }

    if NS_AVAILABLE:
        # In-process path: pass DataFrame directly via 'dataframe' shortcut
        s1_cfg = {
            **s1_cfg_base,
            "dataset_path": "n/a",  # bypassed when 'dataframe' key is present
            "dataframe": df,
        }
        # Wire feature set — registered: feature_set_configs; novel: FeatureProcessor
        if feature_set_name in _REGISTERED_FS:
            s1_cfg["feature_set_configs"] = [fs_config]
        else:
            try:
                fp = _build_feature_processor(proposal, state, cfg)
                s1_cfg["feature_processor"] = fp
            except Exception as exc:
                log.error(
                    "ns_experiment_runner | Failed to build feature processor for S1 (%r): %s",
                    feature_set_name, exc, exc_info=True,
                )
                return None

        try:
            models = ns_create_s1_model(s1_cfg)
        except Exception as exc:
            log.error(
                "ns_experiment_runner | S1 training failed (proposal=%r): %s", proposal_name, exc,
                exc_info=True,
            )
            return None

        if not models:
            log.error("ns_experiment_runner | No models returned for proposal=%r", proposal_name)
            return None

        best = max(models, key=lambda m: m.metrics.get("test_auc", 0.0))
        log.info(
            "ns_experiment_runner | Best model — test_auc=%.4f, train_auc=%.4f",
            best.metrics.get("test_auc", 0.0), best.metrics.get("train_auc", 0.0),
        )
        return best

    else:
        # Subprocess bridge path: write DataFrame to a temp CSV, pass path to bridge
        log.info("ns_experiment_runner | Using subprocess bridge for create_s1_model")
        tmp_csv = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as fh:
                tmp_csv = fh.name
                df.to_csv(fh, index=False)

            s1_cfg_bridge = {
                **s1_cfg_base,
                "dataset_path": tmp_csv,
            }
            # Wire feature set for bridge
            if feature_set_name in _REGISTERED_FS:
                s1_cfg_bridge["feature_set_configs"] = [fs_config]
            else:
                try:
                    script_path = _get_or_generate_feature_set_script(proposal, state, cfg)
                    s1_cfg_bridge["feature_set_class_path"] = str(script_path.resolve())
                    s1_cfg_bridge["feature_set_class_name"] = proposal.get("class_name", "")
                    # Pass layer patterns so the bridge can inject them into fs_cfg
                    ds_entry = get_dataset_or_first(
                        proposal.get("dataset") or rcfg.get("default_dataset")
                    )
                    if ds_entry:
                        patterns = ds_entry.get("layer_name_patterns") or {}
                        if patterns.get("ffn"):
                            s1_cfg_bridge["ffn_layer_patterns"] = patterns["ffn"]
                        if patterns.get("attn"):
                            s1_cfg_bridge["attn_layer_patterns"] = patterns["attn"]
                except Exception as exc:
                    log.error(
                        "ns_experiment_runner | Failed to generate feature set script for S1 (%r): %s",
                        feature_set_name, exc, exc_info=True,
                    )
                    return None

            result = _call_bridge("create_s1_model", s1_cfg_bridge, cfg)
        except Exception as exc:
            log.error(
                "ns_experiment_runner | S1 bridge failed (proposal=%r): %s", proposal_name, exc,
                exc_info=True,
            )
            return None
        finally:
            if tmp_csv:
                try:
                    Path(tmp_csv).unlink(missing_ok=True)
                except OSError:
                    pass

        best = _S1Result(
            metrics=result.get("metrics", {}),
            params=result.get("params", {}),
            feature_importance=result.get("feature_importance", {}),
        )
        log.info(
            "ns_experiment_runner | Best model (bridge) — test_auc=%.4f, train_auc=%.4f",
            best.metrics.get("test_auc", 0.0), best.metrics.get("train_auc", 0.0),
        )
        return best


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def _log_mlflow(
    result: dict,
    state: ResearchState,
    cfg,
    rcfg: dict,
) -> str:
    """Log result to MLflow. Returns run_id or '' on failure."""
    experiment_name: str = rcfg.get("mlflow_experiment", "neuralsignal_agent_experiments")
    proposal = result["proposal"]
    metrics = result["metrics"]
    experiment_id = result["experiment_id"]
    proposal_name = result["proposal_name"]

    try:
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(experiment_name)

        run_name = f"{proposal_name}_{experiment_id[:8]}"
        with mlflow.start_run(run_name=run_name) as run:
            # Parameters
            mlflow.log_params({
                "proposal_name": proposal_name,
                "feature_set_name": proposal.get("feature_set_name", ""),
                "detector_name": result.get("detector_name", ""),
                "dataset": result.get("dataset_name", ""),
                "n_features": result.get("n_features", 0),
                "rows_dataset": metrics.get("rows_dataset", 0),
                "optimization_metric": rcfg.get("optimization_metric", "auc"),
            })

            # Metrics — all numeric values from S1Model.metrics
            numeric_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)

            # Tags
            research_direction = state.get("research_direction", "")
            mlflow.set_tags({
                "research_direction": research_direction[:250],
                "experiment_id": experiment_id,
                "source": "ns_experiment_runner",
                "cache_hits": str(result.get("cache_hits", 0)),
            })

            # Artifacts
            if result.get("feature_importance"):
                mlflow.log_dict(result["feature_importance"], "feature_importance.json")
            if result.get("s1_params"):
                mlflow.log_dict(result["s1_params"], "model_params.json")

            proposal_json = json.dumps(
                {k: v for k, v in proposal.items() if not callable(v)},
                indent=2, default=str,
            )
            mlflow.log_text(proposal_json, "proposal.json")

            run_id = run.info.run_id

        log.info(
            "ns_experiment_runner | MLflow logged — experiment=%r, run_id=%s, test_auc=%.4f",
            experiment_name, run_id, metrics.get("test_auc", 0.0),
        )
        return run_id

    except Exception as exc:
        log.warning("ns_experiment_runner | MLflow logging failed: %s", exc, exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# ChromaDB logging
# ---------------------------------------------------------------------------

def _build_chroma_document(result: dict, state: ResearchState) -> tuple[str, dict]:
    """Return (document_text, metadata) for ChromaDB upsert."""
    proposal = result["proposal"]
    metrics = result["metrics"]
    research_direction = state.get("research_direction", "")

    document = (
        f"Research direction: {research_direction}\n"
        f"Proposal: {proposal.get('name', '')}\n"
        f"Description: {proposal.get('description', '')}\n"
        f"Feature set: {proposal.get('feature_set_name', '')}\n"
        f"Hypothesis: {proposal.get('hypothesis', '')}\n"
        f"Target behavior: {proposal.get('target_behavior', '')}\n"
        f"Metrics: test_auc={metrics.get('test_auc', 0.0):.4f}, "
        f"test_f1={metrics.get('test_f1', 0.0):.4f}, "
        f"test_accuracy={metrics.get('test_accuracy', 0.0):.4f}"
    )

    metadata = {
        "experiment_id": result["experiment_id"],
        "proposal_name": result["proposal_name"],
        "feature_set_name": proposal.get("feature_set_name", ""),
        "test_auc": float(metrics.get("test_auc", 0.0)),
        "train_auc": float(metrics.get("train_auc", 0.0)),
        "test_f1": float(metrics.get("test_f1", 0.0)),
        "test_accuracy": float(metrics.get("test_accuracy", 0.0)),
        "mlflow_run_id": result.get("mlflow_run_id", ""),
        "detector_name": result.get("detector_name", ""),
        "dataset": result.get("dataset_name", ""),
        "inserted_at": result.get("inserted_at", ""),
    }
    return document, metadata


def _log_chromadb(result: dict, state: ResearchState, rcfg: dict) -> str:
    """Upsert result into ChromaDB. Returns experiment_id (used as record ID)."""
    experiment_id = result["experiment_id"]
    try:
        collection_name = rcfg.get("chroma_collection", "agent_experiments")
        store = ChromaStore(collection_name=collection_name)
        document, metadata = _build_chroma_document(result, state)
        store.upsert(experiment_id, document, metadata)
        log.info("ns_experiment_runner | ChromaDB upsert complete — id=%s", experiment_id)
    except Exception as exc:
        log.warning("ns_experiment_runner | ChromaDB upsert failed: %s", exc, exc_info=True)
    return experiment_id


# ---------------------------------------------------------------------------
# MongoDB logging
# ---------------------------------------------------------------------------

def _log_mongo(result: dict, cfg, rcfg: dict) -> None:
    """Insert full result into MongoDB agent_experiments.experiments. Never raises."""
    db_name: str = rcfg.get("agent_experiments_db", "agent_experiments")
    try:
        client = pymongo.MongoClient(cfg.mongo_url)
        db = client[db_name]
        # Strip any non-serialisable runtime objects before inserting
        doc = {k: v for k, v in result.items() if k != "proposal" or isinstance(v, dict)}
        db["experiments"].insert_one(doc)
        client.close()
        log.info(
            "ns_experiment_runner | MongoDB logged — db=%r, experiment_id=%s",
            db_name, result["experiment_id"],
        )
    except Exception as exc:
        log.warning("ns_experiment_runner | MongoDB logging failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def ns_experiment_runner_node(state: ResearchState) -> dict:
    """Run neuralsignal automation experiments for every feature proposal.

    Reads:
        state['feature_proposals']   — proposals from feature_proposal_node
        state['research_direction']  — for MLflow tags and ChromaDB embedding

    Writes:
        state['ns_experiment_results']    — one result dict per successful proposal
        state['ns_best_auc']              — highest test_auc across proposals
        state['ns_mlflow_run_ids']        — MLflow run IDs (one per proposal)
        state['ns_chroma_record_ids']     — ChromaDB record IDs (one per proposal)
        state['execution_success']        — True if ≥1 proposal produced results
        state['dataset_ids']              — all dataset UUIDs touched in this run
        state['dataset_cache_hits']       — total DatasetManager cache hits
    """
    cfg = get_config()
    rcfg = _runner_cfg()

    proposals = state.get("feature_proposals") or []
    if not proposals:
        log.error("ns_experiment_runner_node | No feature_proposals in state")
        return {"execution_success": False, "errors": ["No feature_proposals in state"]}

    dm = DatasetManager(cfg.mongo_url, cfg.datasets_db_name)

    all_results: list[dict] = []
    all_dataset_ids: list[str] = []
    total_cache_hits = 0
    mlflow_run_ids: list[str] = []
    chroma_ids: list[str] = []

    log.info("ns_experiment_runner_node | Processing %d proposal(s)", len(proposals))

    for idx, proposal in enumerate(proposals):
        proposal_name: str = proposal.get("name") or proposal.get("feature_set_name", f"proposal_{idx}")
        log.info(
            "ns_experiment_runner_node | Proposal %d/%d — %r",
            idx + 1, len(proposals), proposal_name,
        )

        experiment_id = str(uuid4())

        # 1. Resolve dataset (cache or create via automation)
        df, entries, cache_hits = _resolve_datasets(proposal, dm, cfg, rcfg, state)
        total_cache_hits += cache_hits
        all_dataset_ids.extend(e.get("dataset_id", "") for e in entries)

        if df is None or df.empty:
            log.warning(
                "ns_experiment_runner_node | No dataset for proposal=%r — skipping",
                proposal_name,
            )
            continue

        # 2. Train S1 model (XGBoost, AUC-optimised)
        s1_model = _train_s1_model(df, proposal, experiment_id, cfg, rcfg, state)
        if s1_model is None:
            log.warning(
                "ns_experiment_runner_node | Training failed for proposal=%r — skipping",
                proposal_name,
            )
            continue

        # 3. Assemble result record
        feature_cols = [c for c in df.columns if c != "target"]
        detector_name: str = (
            proposal.get("detector_name") or rcfg.get("default_detector", "hallucination")
        )
        dataset_name: str = (
            proposal.get("dataset") or rcfg.get("default_dataset", "HaluBench")
        )

        result: dict = {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "proposal": {k: v for k, v in proposal.items() if not callable(v)},
            "metrics": dict(s1_model.metrics),
            "s1_params": dict(s1_model.params) if s1_model.params else {},
            "feature_importance": (
                s1_model.artifacts.get("feature_importance", {})
                if s1_model.artifacts else {}
            ),
            "dataset_ids": [e.get("dataset_id", "") for e in entries],
            "cache_hits": cache_hits,
            "n_features": len(feature_cols),
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            "inserted_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": "",
            "chroma_record_id": "",
        }

        # 4. Log to MLflow
        run_id = _log_mlflow(result, state, cfg, rcfg)
        result["mlflow_run_id"] = run_id
        if run_id:
            mlflow_run_ids.append(run_id)

        # 5. Log to ChromaDB
        chroma_id = _log_chromadb(result, state, rcfg)
        result["chroma_record_id"] = chroma_id
        chroma_ids.append(chroma_id)

        # 6. Log to MongoDB
        _log_mongo(result, cfg, rcfg)

        all_results.append(result)
        log.info(
            "ns_experiment_runner_node | Proposal %r done — test_auc=%.4f, mlflow=%s",
            proposal_name, result["metrics"].get("test_auc", 0.0), run_id,
        )

    # Summary
    n_ok = len(all_results)
    best_auc = max(
        (r["metrics"].get("test_auc", 0.0) for r in all_results),
        default=0.0,
    )
    log.info(
        "ns_experiment_runner_node | Complete — %d/%d proposals succeeded, best_auc=%.4f",
        n_ok, len(proposals), best_auc,
    )

    return {
        "ns_experiment_results": all_results,
        "ns_best_auc": best_auc,
        "ns_mlflow_run_ids": mlflow_run_ids,
        "ns_chroma_record_ids": chroma_ids,
        "execution_success": n_ok > 0,
        "dataset_ids": [did for did in all_dataset_ids if did],
        "dataset_cache_hits": total_cache_hits,
    }
