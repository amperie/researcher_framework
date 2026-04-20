"""NeuralSignal research plugin skeleton.

This module is the domain adapter for the generic research graph. It should own
all NeuralSignal-specific experiment preparation and execution:

- converting proposals into NeuralSignal dataset/model configs
- loading generated ``FeatureSetBase`` implementations
- calling generic subprocess tasks in a separate NeuralSignal-capable process
- normalizing dataset/model outputs into graph state keys

Heavy NeuralSignal imports live in ``plugins/neuralsignal/tasks.py``. This
plugin should mostly orchestrate config, paths, state deltas, and error handling.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from configs.config import get_config
from plugins.base import ResearchAdapter
from utils.logger import get_logger

log = get_logger(__name__)

_BRIDGE_SCRIPT = Path(__file__).parent / "bridge.py"
_TASK_RUNNER = Path(__file__).resolve().parents[1] / "task_runner.py"
_CREATE_DATASET_TASK = "plugins.neuralsignal.tasks.create_dataset"
_CREATE_S1_MODEL_TASK = "plugins.neuralsignal.tasks.create_s1_model"


class NeuralSignalPlugin(ResearchAdapter):
    """ResearchAdapter skeleton for NeuralSignal experiments.

    The generic graph calls this plugin as:

    ``prepare_experiment(profile, state)``
        Build feature datasets or dataset creation manifests from proposals and
        generated FeatureSetBase implementations.

    ``execute_experiment(profile, state)``
        Train/evaluate NeuralSignal models from prepared artifacts and return
        normalized experiment results.
    """

    def validate_environment(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Return cheap diagnostics for research context and troubleshooting."""
        cfg = get_config()
        ns_src = Path(cfg.neuralsignal_src_path).resolve()
        return {
            "bridge_script": str(_BRIDGE_SCRIPT.resolve()),
            "bridge_exists": _BRIDGE_SCRIPT.exists(),
            "task_runner": str(_TASK_RUNNER.resolve()),
            "task_runner_exists": _TASK_RUNNER.exists(),
            "dataset_task": _CREATE_DATASET_TASK,
            "model_task": _CREATE_S1_MODEL_TASK,
            "neuralsignal_python": cfg.neuralsignal_python,
            "neuralsignal_src_path": str(ns_src),
            "neuralsignal_src_exists": ns_src.exists(),
            "experiments_dir": cfg.experiments_dir,
        }

    def build_context(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Expose NeuralSignal constraints to research/ideation tools."""
        return {
            "datasets": profile.get("datasets") or [],
            "base_classes": profile.get("base_classes") or [],
            "evaluation": profile.get("evaluation") or {},
            "bridge": self.validate_environment(profile, state),
        }

    def prepare_experiment(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Prepare NeuralSignal dataset artifacts from proposals.

        TODO:
        - Load each generated FeatureSetBase implementation.
        - Build a task payload for NeuralSignal dataset creation.
        - Call ``plugins/task_runner.py plugins.neuralsignal.tasks.create_dataset``.
        - Read/validate returned CSV paths.
        - Return artifacts with dataset metadata.

        Current skeleton behavior:
        - Builds deterministic dataset manifests without calling the bridge.
        - Marks each artifact as ``pending_bridge`` so execution cannot silently
          pretend a dataset exists.
        """
        proposals = state.get("proposals") or []
        implementations = state.get("implementations") or []
        impl_by_name = _implementations_by_proposal(implementations)

        artifacts: list[dict[str, Any]] = []
        errors = list(state.get("errors") or [])

        for proposal in proposals:
            proposal_name = proposal.get("name", "unknown")
            implementation = impl_by_name.get(proposal_name)
            artifact_id = f"{proposal_name}_dataset_manifest"

            try:
                dataset_cfg = self._build_dataset_config(profile, proposal, implementation)
                artifacts.append({
                    "artifact_id": artifact_id,
                    "artifact_type": "neuralsignal_dataset_manifest",
                    "proposal_name": proposal_name,
                    "status": "pending_bridge",
                    "dataset_config": dataset_cfg,
                    "implementation": _implementation_summary(implementation),
                })
            except Exception as exc:
                log.error("NeuralSignalPlugin.prepare_experiment | %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"prepare_experiment: {proposal_name} failed: {exc}")

        return {
            "experiment_artifacts": artifacts,
            "errors": errors,
        }

    def execute_experiment(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Execute NeuralSignal experiments from prepared artifacts.

        TODO:
        - For each prepared dataset artifact, run ``plugins.neuralsignal.tasks.create_dataset``
          if the dataset has not already been materialized.
        - Build and run ``plugins.neuralsignal.tasks.create_s1_model`` for model training.
        - Normalize bridge output into ``experiment_results`` and optional
          ``models`` entries.

        Current skeleton behavior:
        - Returns one explicit non-fatal error per pending artifact.
        - Does not call the bridge yet.
        """
        artifacts = state.get("experiment_artifacts") or []
        errors = list(state.get("errors") or [])
        results: list[dict[str, Any]] = []
        models: list[dict[str, Any]] = []

        for artifact in artifacts:
            proposal_name = artifact.get("proposal_name", "unknown")
            if artifact.get("status") == "pending_bridge":
                errors.append(
                    "execute_experiment: "
                    f"{proposal_name} is pending bridge implementation in NeuralSignalPlugin"
                )
                continue

            # Future task-runner-backed execution should populate this block.
            # Keeping the shape here documents what downstream nodes expect.
            experiment_id = str(uuid4())
            results.append({
                "experiment_id": experiment_id,
                "proposal_name": proposal_name,
                "metrics": {},
                "feature_importance": {},
                "artifact": _serializable_artifact_summary(artifact),
            })

        return {
            "experiment_results": results,
            "models": models,
            "errors": errors,
        }

    def summarize_result(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Return a compact NeuralSignal-specific result summary."""
        primary_metric = (profile.get("evaluation") or {}).get("primary_metric", "test_auc")
        results = state.get("experiment_results") or []
        return {
            "primary_metric": primary_metric,
            "n_results": len(results),
            "results": [
                {
                    "proposal_name": result.get("proposal_name"),
                    "primary_metric_value": (result.get("metrics") or {}).get(primary_metric),
                    "metrics": result.get("metrics") or {},
                    "feature_importance_keys": sorted((result.get("feature_importance") or {}).keys())[:20],
                }
                for result in results
            ],
        }

    def _build_dataset_config(
        self,
        profile: dict[str, Any],
        proposal: dict[str, Any],
        implementation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the future ``plugins.neuralsignal.tasks.create_dataset`` payload.

        TODO: fill this with the exact NeuralSignal automation config once the
        plugin is wired back to real execution.
        """
        dataset_meta = _dataset_for_proposal(profile, proposal)
        detector = proposal.get("detector") or _first(dataset_meta.get("available_detectors") or [])
        script_path = (implementation or {}).get("script_path", "")

        return {
            "action": "create_dataset",
            "proposal_name": proposal.get("name", "unknown"),
            "dataset": proposal.get("dataset") or dataset_meta.get("name", ""),
            "detector": detector,
            "feature_set_class_path": str(Path(script_path).resolve()) if script_path else "",
            "feature_set_class_name": (implementation or {}).get("class_name", ""),
            "hyperparameters": proposal.get("hyperparameters") or {},
            "mongo_query": proposal.get("mongo_query") or {},
            "storage": dataset_meta.get("storage") or {},
            "layer_name_patterns": dataset_meta.get("layer_name_patterns") or {},
        }

    def _build_model_config(
        self,
        profile: dict[str, Any],
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the future ``plugins.neuralsignal.tasks.create_s1_model`` payload.

        TODO: include dataset path, optimization metric, CV settings, and backend
        config required by NeuralSignal automation.
        """
        return {
            "action": "create_s1_model",
            "proposal_name": artifact.get("proposal_name", "unknown"),
            "optimization_metric": (profile.get("evaluation") or {}).get("primary_metric", "test_auc"),
            "dataset_path": artifact.get("file_path", ""),
        }

    def _call_task(
        self,
        task_path: str,
        payload: dict[str, Any],
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Run a dotted task path in a NeuralSignal-capable subprocess."""
        cfg = get_config()
        timeout = timeout or cfg.experiment_timeout_seconds
        env = os.environ.copy()

        ns_src = str(Path(cfg.neuralsignal_src_path).resolve())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{ns_src}{os.pathsep}{existing}" if existing else ns_src

        cmd = cfg.neuralsignal_python.split() + ["-u", str(_TASK_RUNNER.resolve()), task_path]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
        )

        stdout_chunks: list[str] = []

        def _read_stdout() -> None:
            if proc.stdout:
                stdout_chunks.append(proc.stdout.read())

        def _relay_stderr() -> None:
            if not proc.stderr:
                return
            for line in iter(proc.stderr.readline, ""):
                stripped = line.rstrip("\n")
                if stripped:
                    log.info("[neuralsignal bridge] %s", stripped)

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_relay_stderr, daemon=True)
        t_out.start()
        t_err.start()

        if proc.stdin:
            try:
                proc.stdin.write(json.dumps(payload, default=str))
                proc.stdin.close()
            except BrokenPipeError:
                pass

        t_out.join(timeout=timeout)
        t_err.join(timeout=10)
        proc.wait(timeout=10)

        stdout_data = "".join(stdout_chunks)
        if proc.returncode != 0:
            raise RuntimeError(f"Task {task_path!r} exited {proc.returncode}: {stdout_data[:500]}")

        json_line = next(
            (line for line in reversed(stdout_data.splitlines()) if line.lstrip().startswith(("{", "["))),
            None,
        )
        if json_line is None:
            raise RuntimeError(f"Task {task_path!r} produced no JSON output")

        result = json.loads(json_line)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Task {task_path!r} error: {result['error']}")
        return result


def get_adapter() -> NeuralSignalPlugin:
    """Factory used by ``plugins.loader.load_adapter``."""
    return NeuralSignalPlugin()


def _implementations_by_proposal(implementations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        impl.get("proposal_name", ""): impl
        for impl in implementations
        if impl.get("proposal_name")
    }


def _dataset_for_proposal(profile: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    datasets = profile.get("datasets") or []
    requested = proposal.get("dataset")
    if requested:
        match = next((ds for ds in datasets if ds.get("name") == requested), None)
        if match:
            return match
    return datasets[0] if datasets else {}


def _implementation_summary(implementation: dict[str, Any] | None) -> dict[str, Any]:
    if not implementation:
        return {}
    return {
        "proposal_name": implementation.get("proposal_name"),
        "class_name": implementation.get("class_name"),
        "script_path": implementation.get("script_path"),
        "validated": implementation.get("validated"),
    }


def _serializable_artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if not key.startswith("_")
    }


def _first(items: list[Any]) -> Any:
    return items[0] if items else None
