"""NeuralSignal research plugin.

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

import csv
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from configs.config import get_config
from plugins.base import ResearchAdapter
from plugins.job_runner import TERMINAL_STATUSES, get_runner
from utils.logger import get_logger

log = get_logger(__name__)

_BRIDGE_SCRIPT = Path(__file__).parent / "bridge.py"
_TASK_RUNNER = Path(__file__).resolve().parents[1] / "task_runner.py"
_CREATE_DATASET_TASK = "plugins.neuralsignal.tasks.create_dataset"
_CREATE_S1_MODEL_TASK = "plugins.neuralsignal.tasks.create_s1_model"


class NeuralSignalPlugin(ResearchAdapter):
    """ResearchAdapter implementation for NeuralSignal experiments.

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

        Each proposal is paired with its generated ``FeatureSetBase``
        implementation, converted to a NeuralSignal dataset-automation payload,
        executed in the NeuralSignal subprocess, and normalized into dataset
        artifacts that downstream graph nodes can consume.
        """
        proposals = state.get("proposals") or []
        implementations = state.get("implementations") or []
        impl_by_name = _implementations_by_proposal(implementations)

        artifacts: list[dict[str, Any]] = []
        datasets: list[dict[str, Any]] = []
        errors = list(state.get("errors") or [])
        cfg = get_config()
        cwd = str(_neuralsignal_workdir(cfg))

        for proposal in proposals:
            proposal_name = proposal.get("name", "unknown")
            implementation = impl_by_name.get(proposal_name)

            try:
                dataset_cfg = self._build_dataset_config(profile, proposal, implementation)
                task_result = self._call_task(_CREATE_DATASET_TASK, dataset_cfg, cwd=cwd)
                file_paths = _as_list(task_result.get("file_paths") or task_result.get("paths"))
                if not file_paths:
                    raise RuntimeError("dataset task returned no file_paths")

                for idx, file_path in enumerate(file_paths):
                    resolved_path = _resolve_task_path(file_path, cwd)
                    metadata = _csv_metadata(resolved_path)
                    artifact = {
                        "artifact_id": f"{proposal_name}_dataset_{idx}",
                        "artifact_type": "dataset",
                        "dataset_id": f"{proposal_name}_{idx}",
                        "proposal_name": proposal_name,
                        "status": "ready" if metadata.get("exists") else "missing_file",
                        "file_path": str(resolved_path),
                        "dataset_path": str(resolved_path),
                        "task_file_path": str(file_path),
                        "dataset": dataset_cfg.get("dataset", ""),
                        "detector": dataset_cfg.get("detector_names", [None])[0],
                        "rows": metadata.get("rows"),
                        "columns": metadata.get("columns"),
                        "column_names": metadata.get("column_names", []),
                        "dataset_config": dataset_cfg,
                        "task_result": _json_safe(task_result),
                        "implementation": _implementation_summary(implementation),
                    }
                    artifacts.append(artifact)
                    datasets.append(artifact)
                    if not metadata.get("exists"):
                        errors.append(f"prepare_experiment: {proposal_name} returned missing dataset file {resolved_path}")
            except Exception as exc:
                log.error("NeuralSignalPlugin.prepare_experiment | %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"prepare_experiment: {proposal_name} failed: {exc}")

        return {
            "experiment_artifacts": artifacts,
            "datasets": datasets,
            "errors": errors,
        }

    def execute_experiment(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Execute NeuralSignal experiments from prepared artifacts.

        For each ready dataset artifact, build a NeuralSignal S1 model payload,
        execute it in the NeuralSignal subprocess, and normalize metrics,
        feature importance, and model metadata for generic evaluation/storage.
        """
        artifacts = state.get("experiment_artifacts") or []
        errors = list(state.get("errors") or [])
        results: list[dict[str, Any]] = []
        models: list[dict[str, Any]] = []
        cfg = get_config()
        cwd = str(_neuralsignal_workdir(cfg))

        for artifact in artifacts:
            proposal_name = artifact.get("proposal_name", "unknown")
            if artifact.get("artifact_type") != "dataset":
                continue
            if artifact.get("status") != "ready":
                errors.append(f"execute_experiment: {proposal_name} dataset artifact is not ready")
                continue

            experiment_id = str(uuid4())
            try:
                model_cfg = self._build_model_config(profile, artifact, experiment_id)
                task_result = self._call_task(_CREATE_S1_MODEL_TASK, model_cfg, cwd=cwd)
                metrics = _as_dict(task_result.get("metrics"))
                feature_importance = _as_dict(task_result.get("feature_importance"))
                params = _as_dict(task_result.get("params"))

                model = {
                    "model_id": model_cfg.get("model_name", experiment_id),
                    "experiment_id": experiment_id,
                    "proposal_name": proposal_name,
                    "metrics": metrics,
                    "params": params,
                    "feature_importance": feature_importance,
                    "task_result": _json_safe(task_result),
                }
                models.append(model)
                results.append({
                    "experiment_id": experiment_id,
                    "proposal_name": proposal_name,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "params": params,
                    "artifact": _serializable_artifact_summary(artifact),
                    "model": model,
                })
            except Exception as exc:
                log.error("NeuralSignalPlugin.execute_experiment | %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"execute_experiment: {proposal_name} failed: {exc}")

        return {
            "experiment_results": results,
            "models": models,
            "errors": errors,
        }

    def submit_experiment_jobs(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Submit long-running NeuralSignal dataset/model jobs and return immediately."""
        jobs = list(state.get("experiment_jobs") or [])
        artifacts = list(state.get("experiment_artifacts") or [])
        errors = list(state.get("errors") or [])
        runner = get_runner(_execution_cfg(profile).get("runner", "local_process"))

        active_count = sum(1 for job in jobs if job.get("status") not in TERMINAL_STATUSES)
        max_parallel = int(_execution_cfg(profile).get("max_parallel_jobs", 1) or 1)

        proposals = state.get("proposals") or []
        implementations = state.get("implementations") or []
        impl_by_name = _implementations_by_proposal(implementations)

        submitted: list[dict[str, Any]] = []
        for proposal in proposals:
            if active_count >= max_parallel:
                break
            proposal_name = proposal.get("name", "unknown")
            if _has_dataset_artifact(artifacts, proposal_name) or _has_job(jobs, "dataset", proposal_name):
                continue
            try:
                payload = self._build_dataset_config(profile, proposal, impl_by_name.get(proposal_name))
                job = runner.submit(self._job_spec(profile, "dataset", proposal_name, _CREATE_DATASET_TASK, payload))
                jobs.append(job)
                submitted.append(job)
                active_count += 1
            except Exception as exc:
                log.error("NeuralSignalPlugin.submit_experiment_jobs | dataset %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"submit_experiment_jobs: dataset {proposal_name} failed: {exc}")

        for artifact in artifacts:
            if active_count >= max_parallel:
                break
            if artifact.get("artifact_type") != "dataset" or artifact.get("status") != "ready":
                continue
            proposal_name = artifact.get("proposal_name", "unknown")
            if _has_result(state.get("experiment_results") or [], proposal_name) or _has_job(jobs, "model", proposal_name, artifact.get("artifact_id")):
                continue
            try:
                experiment_id = str(uuid4())
                payload = self._build_model_config(profile, artifact, experiment_id)
                spec = self._job_spec(
                    profile,
                    "model",
                    proposal_name,
                    _CREATE_S1_MODEL_TASK,
                    payload,
                    artifact_id=artifact.get("artifact_id"),
                    experiment_id=experiment_id,
                )
                job = runner.submit(spec)
                jobs.append(job)
                submitted.append(job)
                active_count += 1
            except Exception as exc:
                log.error("NeuralSignalPlugin.submit_experiment_jobs | model %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"submit_experiment_jobs: model {proposal_name} failed: {exc}")

        return {
            "experiment_jobs": jobs,
            "submitted_jobs": submitted,
            "errors": errors,
        }

    def check_experiment_jobs(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Poll NeuralSignal jobs and collect completed dataset/model outputs."""
        jobs = list(state.get("experiment_jobs") or [])
        artifacts = list(state.get("experiment_artifacts") or [])
        datasets = list(state.get("datasets") or [])
        results = list(state.get("experiment_results") or [])
        models = list(state.get("models") or [])
        errors = list(state.get("errors") or [])
        runner = get_runner(_execution_cfg(profile).get("runner", "local_process"))

        updated_jobs: list[dict[str, Any]] = []
        for job in jobs:
            checked = runner.check(job)
            updated_jobs.append(checked)
            if checked.get("status") == "succeeded" and not checked.get("collected"):
                try:
                    result = _read_result(checked)
                    if checked.get("stage") == "dataset":
                        new_artifacts = self._dataset_artifacts_from_job(checked, result)
                        artifacts.extend(new_artifacts)
                        datasets.extend(new_artifacts)
                    elif checked.get("stage") == "model":
                        experiment_result, model = self._model_result_from_job(checked, result, artifacts)
                        results.append(experiment_result)
                        models.append(model)
                    checked["collected"] = True
                    _write_job_status(checked)
                except Exception as exc:
                    log.error("NeuralSignalPlugin.check_experiment_jobs | collect %s failed: %s", checked.get("job_id"), exc, exc_info=True)
                    checked["collected"] = True
                    _write_job_status(checked)
                    errors.append(f"check_experiment_jobs: collect {checked.get('job_id')} failed: {exc}")
            elif checked.get("status") == "failed" and not checked.get("reported"):
                checked["reported"] = True
                _write_job_status(checked)
                errors.append(f"check_experiment_jobs: {checked.get('proposal_name')} {checked.get('stage')} failed: {checked.get('error')}")

        delta = {
            "experiment_jobs": updated_jobs,
            "experiment_artifacts": artifacts,
            "datasets": datasets,
            "experiment_results": results,
            "models": models,
            "errors": errors,
        }
        if _execution_cfg(profile).get("auto_submit_next_stage", True):
            submitted_delta = self.submit_experiment_jobs(profile, delta)
            delta["experiment_jobs"] = submitted_delta.get("experiment_jobs", updated_jobs)
            delta["submitted_jobs"] = submitted_delta.get("submitted_jobs", [])
            delta["errors"] = submitted_delta.get("errors", errors)
        return delta

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
        """Build the ``plugins.neuralsignal.tasks.create_dataset`` payload."""
        cfg = get_config()
        dataset_meta = _dataset_for_proposal(profile, proposal)
        detector = proposal.get("detector") or _first(dataset_meta.get("available_detectors") or [])
        script_path = (implementation or {}).get("script_path", "")
        storage = dataset_meta.get("storage") or {}
        hyperparameters = proposal.get("hyperparameters") or {}
        proposal_name = proposal.get("name", "unknown")
        dataset_name = proposal.get("dataset") or dataset_meta.get("name", "")
        app_name = storage.get("application_name", dataset_name)
        sub_app_name = storage.get("sub_application_name", "")
        class_name = (implementation or {}).get("class_name", "")
        feature_set_name = proposal_name
        layer_patterns = dataset_meta.get("layer_name_patterns") or {}

        return {
            "run_data_collection": False,
            "indirect_config": {
                "indirect_model": "",
                "indirect_batch_size": 1,
                "quantization": "int8",
                "device": hyperparameters.get("device", "cuda:0"),
            },
            "indirect_instrumentation_config": {},
            "create_dataset": True,
            "create_s1_model": False,
            "detector_names": [detector] if detector else [],
            "dataset": dataset_name,
            "application_name": app_name,
            "sub_application_name": sub_app_name,
            "backend_config": _backend_config(cfg),
            "zone_size": hyperparameters.get("zone_size", 1024),
            "dataset_row_limit": int(hyperparameters.get("dataset_row_limit", hyperparameters.get("row_limit", dataset_meta.get("row_limit", 0))) or 0),
            "row_limit": int(hyperparameters.get("dataset_row_limit", hyperparameters.get("row_limit", dataset_meta.get("row_limit", 0))) or 0),
            "write_to_file": True,
            "build_in_memory": False,
            "use_gt_as_target": True,
            "query": proposal.get("mongo_query") or {},
            "file_out": _slug(f"{feature_set_name}_{detector or 'detector'}"),
            "feature_set_class_path": str(Path(script_path).resolve()) if script_path else "",
            "feature_set_class_name": class_name,
            "feature_set_configs": None,
            "ffn_layer_patterns": layer_patterns.get("ffn", []),
            "attn_layer_patterns": layer_patterns.get("attn", []),
            "proposal_name": proposal_name,
        }

    def _build_model_config(
        self,
        profile: dict[str, Any],
        artifact: dict[str, Any],
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the ``plugins.neuralsignal.tasks.create_s1_model`` payload."""
        cfg = get_config()
        dataset_cfg = artifact.get("dataset_config") or {}
        experiment_id = experiment_id or str(uuid4())
        proposal_name = artifact.get("proposal_name", "unknown")
        optimization_metric = (profile.get("evaluation") or {}).get("primary_metric", "test_auc")

        return {
            "application_name": dataset_cfg.get("application_name", ""),
            "sub_application_name": dataset_cfg.get("sub_application_name", ""),
            "indirect_config": {
                "indirect_model": "",
                "quantization": "int8",
            },
            "indirect_instrumentation_config": {},
            "model_name": _slug(f"{proposal_name}_{experiment_id[:8]}"),
            "dataset": dataset_cfg.get("dataset") or artifact.get("dataset", ""),
            "zone_size": dataset_cfg.get("zone_size", 1024),
            "use_full_zone_names": False,
            "file_out": _slug(f"{proposal_name}_{artifact.get('detector') or 'detector'}"),
            "feature_set_class_path": dataset_cfg.get("feature_set_class_path", ""),
            "feature_set_class_name": dataset_cfg.get("feature_set_class_name", ""),
            "feature_set_configs": None,
            "ffn_layer_patterns": dataset_cfg.get("ffn_layer_patterns", []),
            "attn_layer_patterns": dataset_cfg.get("attn_layer_patterns", []),
            "detector_names": dataset_cfg.get("detector_names") or ([artifact.get("detector")] if artifact.get("detector") else []),
            "modeling_row_limits": [0],
            "optimization_metric": optimization_metric,
            "max_evals": int(dataset_cfg.get("max_evals", 20) or 20),
            "test_set_size": 0.33,
            "seed": 42,
            "run_cross_validation": True,
            "cv_folds": 3,
            "create_reduced_feature_model": False,
            "save_to_backend": False,
            "backend_config": dataset_cfg.get("backend_config") or _backend_config(cfg),
            "dataset_path": artifact.get("dataset_path") or artifact.get("file_path", ""),
            "proposal_name": proposal_name,
        }

    def _job_spec(
        self,
        profile: dict[str, Any],
        stage: str,
        proposal_name: str,
        task_path: str,
        payload: dict[str, Any],
        artifact_id: str | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        cfg = get_config()
        jobs_dir = Path(cfg.experiments_dir) / profile.get("name", "neuralsignal") / "jobs"
        job_id = _slug("_".join(item for item in (stage, proposal_name, artifact_id or experiment_id or str(uuid4())[:8]) if item))
        env = os.environ.copy()
        pythonpath_entries = [str(path) for path in _pythonpath_entries(cfg)]
        existing = env.get("PYTHONPATH", "")
        if existing:
            pythonpath_entries.append(existing)
        return {
            "job_id": job_id,
            "job_dir": str(jobs_dir / job_id),
            "runner": _execution_cfg(profile).get("runner", "local_process"),
            "stage": stage,
            "proposal_name": proposal_name,
            "artifact_id": artifact_id,
            "experiment_id": experiment_id,
            "task_path": task_path,
            "payload": payload,
            "python": cfg.neuralsignal_python,
            "cwd": str(_neuralsignal_workdir(cfg)),
            "env": {"PYTHONPATH": os.pathsep.join(pythonpath_entries)},
        }

    def _dataset_artifacts_from_job(self, job: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
        cfg = get_config()
        cwd = str(_neuralsignal_workdir(cfg))
        file_paths = _as_list(result.get("file_paths") or result.get("paths"))
        if not file_paths:
            raise RuntimeError("dataset job returned no file_paths")

        spec = _read_job_spec(job)
        payload = spec.get("payload") or _read_payload(job)
        artifacts: list[dict[str, Any]] = []
        for idx, file_path in enumerate(file_paths):
            resolved_path = _resolve_task_path(file_path, cwd)
            metadata = _csv_metadata(resolved_path)
            artifact = {
                "artifact_id": f"{job.get('proposal_name', 'unknown')}_dataset_{idx}",
                "artifact_type": "dataset",
                "dataset_id": f"{job.get('proposal_name', 'unknown')}_{idx}",
                "proposal_name": job.get("proposal_name", "unknown"),
                "status": "ready" if metadata.get("exists") else "missing_file",
                "file_path": str(resolved_path),
                "dataset_path": str(resolved_path),
                "task_file_path": str(file_path),
                "dataset": payload.get("dataset", ""),
                "detector": (payload.get("detector_names") or [None])[0],
                "rows": metadata.get("rows"),
                "columns": metadata.get("columns"),
                "column_names": metadata.get("column_names", []),
                "dataset_config": payload,
                "task_result": _json_safe(result),
            }
            artifacts.append(artifact)
        return artifacts

    def _model_result_from_job(
        self,
        job: dict[str, Any],
        result: dict[str, Any],
        artifacts: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        experiment_id = job.get("experiment_id") or str(uuid4())
        proposal_name = job.get("proposal_name", "unknown")
        artifact = next((a for a in artifacts if a.get("artifact_id") == job.get("artifact_id")), {})
        metrics = _as_dict(result.get("metrics"))
        feature_importance = _as_dict(result.get("feature_importance"))
        params = _as_dict(result.get("params"))
        model = {
            "model_id": f"{proposal_name}_{experiment_id}",
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "metrics": metrics,
            "params": params,
            "feature_importance": feature_importance,
            "task_result": _json_safe(result),
            "job_id": job.get("job_id"),
        }
        experiment_result = {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "params": params,
            "artifact": _serializable_artifact_summary(artifact),
            "model": model,
            "job_id": job.get("job_id"),
        }
        return experiment_result, model

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

        pythonpath_entries = [str(path) for path in _pythonpath_entries(cfg)]
        existing = env.get("PYTHONPATH", "")
        if existing:
            pythonpath_entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        cmd = cfg.neuralsignal_python.split() + ["-u", "-m", "plugins.task_runner", task_path]
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


def _execution_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    return profile.get("execution") or {}


def _has_dataset_artifact(artifacts: list[dict[str, Any]], proposal_name: str) -> bool:
    return any(
        artifact.get("artifact_type") == "dataset"
        and artifact.get("proposal_name") == proposal_name
        and artifact.get("status") == "ready"
        for artifact in artifacts
    )


def _has_job(
    jobs: list[dict[str, Any]],
    stage: str,
    proposal_name: str,
    artifact_id: str | None = None,
) -> bool:
    for job in jobs:
        if job.get("stage") != stage or job.get("proposal_name") != proposal_name:
            continue
        if artifact_id is not None and job.get("artifact_id") != artifact_id:
            continue
        return True
    return False


def _has_result(results: list[dict[str, Any]], proposal_name: str) -> bool:
    return any(result.get("proposal_name") == proposal_name for result in results)


def _read_result(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["result_path"])
    return json.loads(path.read_text(encoding="utf-8"))


def _read_job_spec(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["job_dir"]) / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_payload(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["job_dir"]) / "payload.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job_status(job: dict[str, Any]) -> None:
    path = Path(job["job_dir"]) / "status.json"
    path.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")


def _backend_config(cfg: Any) -> dict[str, Any]:
    return {
        "backend_type": "neuralsignal_v1",
        "mongo_url": getattr(cfg, "mongo_url", ""),
        "mlflow_uri": getattr(cfg, "mlflow_uri", "http://hp.lan:8899/"),
        "mlflow_register_model": False,
    }


def _neuralsignal_workdir(cfg: Any) -> Path:
    configured = Path(getattr(cfg, "neuralsignal_src_path", "")).resolve()
    if _is_package_dir(configured):
        return configured.parent
    return configured


def _pythonpath_entries(cfg: Any) -> list[Path]:
    configured = Path(getattr(cfg, "neuralsignal_src_path", "")).resolve()
    workdir = _neuralsignal_workdir(cfg)
    researcher_root = _TASK_RUNNER.resolve().parents[1]

    entries = [workdir, configured, researcher_root]
    if _is_package_dir(configured):
        entries.insert(0, configured.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for entry in entries:
        key = str(entry)
        if key not in seen:
            unique.append(entry)
            seen.add(key)
    return unique


def _is_package_dir(path: Path) -> bool:
    return path.name == "neuralsignal" and (path / "__init__.py").exists()


def _csv_metadata(file_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {
            "exists": False,
            "rows": None,
            "columns": None,
            "column_names": [],
        }

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "exists": True,
                "rows": 0,
                "columns": 0,
                "column_names": [],
            }
        rows = sum(1 for _ in reader)

    return {
        "exists": True,
        "rows": rows,
        "columns": len(header),
        "column_names": header,
    }


def _resolve_task_path(file_path: Any, cwd: str | os.PathLike[str] | None) -> Path:
    path = Path(str(file_path))
    if path.is_absolute() or cwd is None:
        return path
    return Path(cwd) / path


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
    except TypeError:
        return json.loads(json.dumps(value, default=str))
    return value


def _slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "neuralsignal_experiment"


def _first(items: list[Any]) -> Any:
    return items[0] if items else None
