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
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow

from configs.config import dev_path, get_config
from core.artifacts import get_artifact_store
from core.memory import (
    MemoryService,
    build_core_memory_records,
    dedupe_memory_records,
    default_memory_record_to_artifact,
    fingerprint_json,
)
from core.plugins.base import ResearchAdapter
from core.plugins.execution import check_task, read_task_result, run_task, submit_task
from core.plugins.job_runner import TERMINAL_STATUSES
from core.utils.logger import get_logger, setup_plugin_file_logging

log = get_logger(__name__)

_BRIDGE_SCRIPT = Path(__file__).parent / "bridge.py"
_TASK_RUNNER = Path(__file__).resolve().parents[1] / "task_runner.py"
_CREATE_DATASET_TASK = "core.plugins.neuralsignal.tasks.create_dataset"
_CREATE_S1_MODEL_TASK = "core.plugins.neuralsignal.tasks.create_s1_model"
_RUN_PROPOSAL_BRANCH_TASK = "core.plugins.neuralsignal.tasks.run_proposal_branch"
_PLUGIN_NAME = "neuralsignal"
_PLUGIN_LOGGER_PREFIXES = [
    "core.plugins.neuralsignal",
    "core.plugins.task_runner",
    "core.plugins.job_runner",
]


def _ensure_plugin_logging() -> None:
    setup_plugin_file_logging(_PLUGIN_NAME, logger_prefixes=_PLUGIN_LOGGER_PREFIXES)


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
        _ensure_plugin_logging()
        cfg = get_config()
        ns_src = Path(cfg.neuralsignal_src_path).resolve()
        return {
            "bridge_script": str(_BRIDGE_SCRIPT.resolve()),
            "bridge_exists": _BRIDGE_SCRIPT.exists(),
            "task_runner": str(_TASK_RUNNER.resolve()),
            "task_runner_exists": _TASK_RUNNER.exists(),
            "dataset_task": _CREATE_DATASET_TASK,
            "model_task": _CREATE_S1_MODEL_TASK,
            "proposal_branch_task": _RUN_PROPOSAL_BRANCH_TASK,
            "neuralsignal_python": cfg.neuralsignal_python,
            "neuralsignal_src_path": str(ns_src),
            "neuralsignal_src_exists": ns_src.exists(),
            "experiments_dir": cfg.experiments_dir,
        }

    def build_context(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Expose NeuralSignal constraints to research/ideation tools."""
        _ensure_plugin_logging()
        return {
            "datasets": profile.get("datasets") or [],
            "base_classes": profile.get("base_classes") or [],
            "evaluation": profile.get("evaluation") or {},
            "bridge": self.validate_environment(profile, state),
        }

    def external_runtime_spec(self, profile: dict[str, Any], purpose: str) -> dict[str, Any]:
        cfg = get_config()
        return {
            "python": cfg.neuralsignal_python,
            "cwd": str(_neuralsignal_workdir(cfg)),
            "pythonpath_entries": [str(path) for path in _pythonpath_entries(cfg)],
            "plugin_name": _PLUGIN_NAME,
            "logger_prefixes": list(_PLUGIN_LOGGER_PREFIXES),
        }

    def knowledge_graph_config(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Return NeuralSignal-specific KG defaults while staying profile-driven."""
        evaluation = profile.get("evaluation") or {}
        primary_metric = str(evaluation.get("primary_metric") or "test_auc")
        return {
            "metrics": {
                primary_metric: {"direction": "higher_is_better"},
            },
            "metric_bands": [
                {
                    "metric_name": primary_metric,
                    "operator": ">=",
                    "threshold": 0.95,
                    "display_name": f"{primary_metric} >= 0.95",
                    "band_key": f"{primary_metric}_gte_0_95",
                },
                {
                    "metric_name": primary_metric,
                    "operator": ">=",
                    "threshold": 0.90,
                    "display_name": f"{primary_metric} >= 0.90",
                    "band_key": f"{primary_metric}_gte_0_90",
                },
                {
                    "metric_name": primary_metric,
                    "operator": ">=",
                    "threshold": 0.75,
                    "display_name": f"{primary_metric} >= 0.75",
                    "band_key": f"{primary_metric}_gte_0_75",
                },
            ],
            "methods": {
                "aliases": {
                    "mlp activation sparsity": "activation sparsity",
                    "activation_sparsity": "activation sparsity",
                }
            },
        }

    def prepare_experiment(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Prepare NeuralSignal dataset artifacts from proposals.

        Each proposal is paired with its generated ``FeatureSetBase``
        implementation, converted to a NeuralSignal dataset-automation payload,
        executed in the NeuralSignal subprocess, and normalized into dataset
        artifacts that downstream graph nodes can consume.
        """
        _ensure_plugin_logging()
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
                memory_artifact = None
                if not _should_overwrite_existing_dataset(dataset_cfg):
                    memory_artifact = _dataset_artifact_from_memory(
                        profile=profile,
                        proposal_name=proposal_name,
                        dataset_cfg=dataset_cfg,
                    )
                expected_dataset_path = _expected_dataset_path(dataset_cfg)
                if memory_artifact is not None:
                    artifact = memory_artifact
                    if implementation is not None:
                        artifact["implementation"] = _implementation_summary(implementation)
                    _register_dataset_artifact(profile, artifact, errors)
                    artifacts.append(artifact)
                    datasets.append(artifact)
                    self._persist_available_memory(
                        profile,
                        state,
                        {
                            "experiment_artifacts": artifacts,
                            "datasets": datasets,
                            "errors": errors,
                        },
                    )
                    continue
                if expected_dataset_path.exists() and not _should_overwrite_existing_dataset(dataset_cfg):
                    log.info(
                        "NeuralSignalPlugin.prepare_experiment | %s reusing existing dataset %s",
                        proposal_name,
                        expected_dataset_path,
                    )
                    task_result = {
                        "skipped_existing_dataset": True,
                        "file_paths": [str(expected_dataset_path)],
                    }
                    file_paths = [str(expected_dataset_path)]
                else:
                    task_result = self._call_task(
                        profile,
                        _CREATE_DATASET_TASK,
                        dataset_cfg,
                        timeout=_task_timeout(profile, "dataset", cfg),
                        cwd=cwd,
                    )
                    file_paths = _as_list(task_result.get("file_paths") or task_result.get("paths"))
                    if not file_paths:
                        raise RuntimeError("dataset task returned no file_paths")

                for idx, file_path in enumerate(file_paths):
                    artifact = _dataset_artifact(
                        proposal_name=proposal_name,
                        file_path=file_path,
                        cwd=cwd,
                        dataset_cfg=dataset_cfg,
                        task_result=task_result,
                        idx=idx,
                        implementation=implementation,
                    )
                    _register_dataset_artifact(profile, artifact, errors)
                    artifacts.append(artifact)
                    datasets.append(artifact)
                    self._persist_available_memory(
                        profile,
                        state,
                        {
                            "experiment_artifacts": artifacts,
                            "datasets": datasets,
                            "errors": errors,
                        },
                    )
                    if artifact.get("status") != "ready":
                        errors.append(
                            f"prepare_experiment: {proposal_name} returned missing dataset file {artifact.get('dataset_path')}"
                        )
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
        _ensure_plugin_logging()
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
                task_result = self._call_task(
                    profile,
                    _CREATE_S1_MODEL_TASK,
                    model_cfg,
                    timeout=_task_timeout(profile, "model", cfg),
                    cwd=str(_model_task_workdir(artifact, cfg)),
                )
                metrics = _as_dict(task_result.get("metrics"))
                feature_importance = _as_dict(task_result.get("feature_importance"))
                params = _as_dict(task_result.get("params"))
                artifacts_payload = _as_dict(task_result.get("artifacts"))
                model_config_payload = _as_dict(task_result.get("model_config"))
                figure_paths = _as_dict(task_result.get("figure_paths"))

                model = {
                    "model_id": model_cfg.get("model_name", experiment_id),
                    "experiment_id": experiment_id,
                    "proposal_name": proposal_name,
                    "metrics": metrics,
                    "params": params,
                    "feature_importance": feature_importance,
                    "artifacts": artifacts_payload,
                    "model_config": model_config_payload,
                    "figure_paths": figure_paths,
                    "task_result": _json_safe(task_result),
                }
                mlflow_run_id = _log_result_to_mlflow(
                    profile=profile,
                    state=state,
                    artifact=artifact,
                    experiment_id=experiment_id,
                    proposal_name=proposal_name,
                    metrics=metrics,
                    params=params,
                    feature_importance=feature_importance,
                    artifacts_payload=artifacts_payload,
                    model_config_payload=model_config_payload,
                    figure_paths=figure_paths,
                    model=model,
                )
                if mlflow_run_id:
                    model["mlflow_run_id"] = mlflow_run_id
                result_payload = {
                    "experiment_id": experiment_id,
                    "proposal_name": proposal_name,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "params": params,
                }
                models.append(model)
                result = {
                    **result_payload,
                    "artifact": _serializable_artifact_summary(artifact),
                    "model": model,
                    "artifacts": artifacts_payload,
                    "model_config": model_config_payload,
                    "figure_paths": figure_paths,
                }
                if mlflow_run_id:
                    result["mlflow_run_id"] = mlflow_run_id
                results.append(result)
                self._persist_available_memory(
                    profile,
                    state,
                    {
                        "experiment_results": results,
                        "models": models,
                        "errors": errors,
                    },
                )
            except Exception as exc:
                log.error("NeuralSignalPlugin.execute_experiment | %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"execute_experiment: {proposal_name} failed: {exc}")

        return {
            "experiment_results": results,
            "models": models,
            "errors": errors,
        }

    def submit_experiment_jobs(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Submit proposal-branch NeuralSignal jobs and return immediately."""
        _ensure_plugin_logging()
        jobs = list(state.get("experiment_jobs") or [])
        errors = list(state.get("errors") or [])
        active_count = sum(1 for job in jobs if job.get("status") not in TERMINAL_STATUSES)
        max_parallel = int(_execution_cfg(profile).get("max_parallel_jobs", 1) or 1)

        proposals = state.get("proposals") or []
        implementations = state.get("implementations") or []
        validation_results = state.get("validation_results") or []
        impl_by_name = _implementations_by_proposal(implementations)
        validation_by_name = _validation_by_proposal(validation_results)

        submitted: list[dict[str, Any]] = []
        for proposal in proposals:
            if active_count >= max_parallel:
                break
            proposal_name = proposal.get("name", "unknown")
            if _has_result(state.get("experiment_results") or [], proposal_name) or _has_job(jobs, "proposal_branch", proposal_name):
                continue
            try:
                implementation = impl_by_name.get(proposal_name)
                validation = validation_by_name.get(proposal_name)
                _require_async_implementation(proposal_name, implementation, validation)
                payload = self._build_proposal_branch_payload(profile, state, proposal, implementation)
                job = submit_task(
                    self._job_spec(
                        profile,
                        "proposal_branch",
                        proposal_name,
                        _RUN_PROPOSAL_BRANCH_TASK,
                        payload,
                        experiment_id=payload.get("experiment_id"),
                    ),
                    profile,
                    "experiment",
                    default_runner="local_process",
                )
                jobs.append(job)
                submitted.append(job)
                active_count += 1
            except Exception as exc:
                log.error("NeuralSignalPlugin.submit_experiment_jobs | branch %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"submit_experiment_jobs: branch {proposal_name} failed: {exc}")

        return {
            "experiment_jobs": jobs,
            "submitted_jobs": submitted,
            "errors": errors,
        }

    def check_experiment_jobs(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Poll NeuralSignal jobs and collect completed proposal-branch outputs."""
        _ensure_plugin_logging()
        jobs = list(state.get("experiment_jobs") or [])
        artifacts = list(state.get("experiment_artifacts") or [])
        datasets = list(state.get("datasets") or [])
        results = list(state.get("experiment_results") or [])
        models = list(state.get("models") or [])
        errors = list(state.get("errors") or [])
        updated_jobs: list[dict[str, Any]] = []
        for job in jobs:
            checked = check_task(job, profile, "experiment", default_runner="local_process")
            updated_jobs.append(checked)
            if checked.get("status") == "succeeded" and not checked.get("collected"):
                try:
                    result = read_task_result(checked)
                    if checked.get("stage") == "proposal_branch":
                        branch = self._proposal_branch_outputs_from_job(profile, checked, result)
                        artifact = branch.get("artifact") or {}
                        if artifact:
                            _register_dataset_artifact(profile, artifact, errors)
                            artifacts = _replace_by_key(artifacts, artifact, "proposal_name")
                            datasets = _replace_by_key(datasets, artifact, "proposal_name")
                        experiment_result = branch.get("experiment_result") or {}
                        model = branch.get("model") or {}
                        mlflow_run_id = _log_result_to_mlflow(
                            profile=profile,
                            state=state,
                            artifact=experiment_result.get("artifact") or artifact,
                            experiment_id=experiment_result.get("experiment_id", ""),
                            proposal_name=experiment_result.get("proposal_name", "unknown"),
                            metrics=experiment_result.get("metrics") or {},
                            params=experiment_result.get("params") or {},
                            feature_importance=experiment_result.get("feature_importance") or {},
                            artifacts_payload=experiment_result.get("artifacts") or {},
                            model_config_payload=experiment_result.get("model_config") or {},
                            figure_paths=experiment_result.get("figure_paths") or {},
                            model=model,
                        )
                        if mlflow_run_id:
                            model["mlflow_run_id"] = mlflow_run_id
                            experiment_result["mlflow_run_id"] = mlflow_run_id
                        if experiment_result:
                            results = _replace_by_key(results, experiment_result, "proposal_name")
                        if model:
                            models = _replace_by_key(models, model, "proposal_name")
                        self._persist_available_memory(
                            profile,
                            state,
                            {
                                "experiment_artifacts": artifacts,
                                "datasets": datasets,
                                "experiment_results": results,
                                "models": models,
                                "errors": errors,
                            },
                        )
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

    def _build_proposal_branch_payload(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
        proposal: dict[str, Any],
        implementation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        proposal_name = proposal.get("name", "unknown")
        experiment_id = str(uuid4())
        dataset_cfg = self._build_dataset_config(profile, proposal, implementation)
        reused_artifact = None
        if not _should_overwrite_existing_dataset(dataset_cfg):
            reused_artifact = _dataset_artifact_from_memory(
                profile=profile,
                proposal_name=proposal_name,
                dataset_cfg=dataset_cfg,
            )
            if reused_artifact is None:
                expected_path = _expected_dataset_path(dataset_cfg)
                if expected_path.exists():
                    reused_artifact = _dataset_artifact(
                        proposal_name=proposal_name,
                        file_path=str(expected_path),
                        cwd=str(_neuralsignal_workdir(get_config())),
                        dataset_cfg=dataset_cfg,
                        task_result={"skipped_existing_dataset": True, "file_paths": [str(expected_path)]},
                        idx=0,
                        implementation=implementation,
                    )
        artifact_hint = reused_artifact or {
            "proposal_name": proposal_name,
            "artifact_type": "dataset",
            "status": "ready",
            "dataset": dataset_cfg.get("dataset", ""),
            "detector": _first(dataset_cfg.get("detector_names") or []) or proposal.get("detector", ""),
            "dataset_path": str(_expected_dataset_path(dataset_cfg)),
            "file_path": str(_expected_dataset_path(dataset_cfg)),
            "dataset_config": dataset_cfg,
        }
        model_cfg = self._build_model_config(profile, artifact_hint, experiment_id)
        return {
            "proposal_name": proposal_name,
            "experiment_id": experiment_id,
            "research_direction": str(state.get("research_direction", "")),
            "proposal": proposal,
            "implementation": _implementation_summary(implementation),
            "dataset_config": dataset_cfg,
            "model_config_base": model_cfg,
            "reused_dataset_artifact": _serializable_artifact_summary(reused_artifact) if reused_artifact else {},
        }

    def _proposal_branch_outputs_from_job(
        self,
        profile: dict[str, Any],
        job: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        spec = _read_job_spec(job)
        payload = spec.get("payload") or _read_payload(job)
        proposal_name = job.get("proposal_name", payload.get("proposal_name", "unknown"))
        experiment_id = job.get("experiment_id") or payload.get("experiment_id") or str(uuid4())
        implementation = _as_dict(payload.get("implementation"))
        dataset_cfg = _as_dict(payload.get("dataset_config"))
        dataset_result = _as_dict(result.get("dataset_result"))
        model_result = _as_dict(result.get("model_result"))
        reused_artifact = _as_dict(payload.get("reused_dataset_artifact"))
        cwd = str(_neuralsignal_workdir(get_config()))

        artifact: dict[str, Any]
        if dataset_result.get("file_paths") or dataset_result.get("paths"):
            artifact = _dataset_artifact(
                proposal_name=proposal_name,
                file_path=_first(_as_list(dataset_result.get("file_paths") or dataset_result.get("paths"))),
                cwd=cwd,
                dataset_cfg=dataset_cfg,
                task_result=dataset_result,
                idx=0,
                implementation=implementation,
            )
        elif reused_artifact:
            artifact = dict(reused_artifact)
            artifact.setdefault("proposal_name", proposal_name)
            artifact.setdefault("artifact_type", "dataset")
            artifact.setdefault("status", "ready")
            artifact.setdefault("dataset_config", dataset_cfg)
            if implementation:
                artifact["implementation"] = implementation
        else:
            raise RuntimeError("proposal branch job returned no dataset output")

        metrics = _as_dict(model_result.get("metrics"))
        feature_importance = _as_dict(model_result.get("feature_importance"))
        params = _as_dict(model_result.get("params"))
        model_config_payload = _as_dict(model_result.get("model_config"))
        figure_paths = _as_dict(model_result.get("figure_paths"))
        model = {
            "model_id": model_config_payload.get("model_name") or f"{proposal_name}_{experiment_id}",
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "metrics": metrics,
            "params": params,
            "feature_importance": feature_importance,
            "artifacts": _as_dict(model_result.get("artifacts")),
            "model_config": model_config_payload,
            "figure_paths": figure_paths,
            "task_result": _json_safe(model_result),
            "job_id": job.get("job_id"),
        }
        experiment_result = {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "proposal": _as_dict(payload.get("proposal")),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "params": params,
            "artifact": _serializable_artifact_summary(artifact),
            "model": model,
            "artifacts": _as_dict(model_result.get("artifacts")),
            "model_config": model_config_payload,
            "figure_paths": figure_paths,
            "job_id": job.get("job_id"),
        }
        return {"artifact": artifact, "experiment_result": experiment_result, "model": model}

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

    def build_memory_records(self, profile: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        """Build NeuralSignal-specific canonical memory records."""
        direction = str(state.get("research_direction", ""))
        evaluation_summary = state.get("evaluation_summary") or {}
        results = state.get("experiment_results") or []
        proposals_by_name = {
            item.get("name"): item
            for item in (state.get("proposals") or [])
            if item.get("name")
        }
        models_by_name = {
            item.get("proposal_name"): item
            for item in (state.get("models") or [])
            if item.get("proposal_name")
        }
        artifacts_by_name = {
            item.get("proposal_name"): item
            for item in (state.get("experiment_artifacts") or [])
            if item.get("proposal_name") and item.get("artifact_type") == "dataset"
        }
        implementations_by_name = _implementations_by_proposal(state.get("implementations") or [])

        records: list[dict[str, Any]] = []
        emitted_dataset_keys: set[str] = set()
        emitted_featureset_keys: set[str] = set()
        root_run_family_id = str(state.get("root_run_family_id") or "")
        root_research_direction = str(state.get("root_research_direction") or direction or "")
        source_next_step_record_id = str(state.get("source_next_step_record_id") or "")
        source_next_step_title = str(state.get("source_next_step_title") or "")

        for proposal_name, implementation in implementations_by_name.items():
            proposal = proposals_by_name.get(proposal_name) or {}
            dataset_name = str(proposal.get("dataset") or "")
            detector = str(proposal.get("detector") or "")
            feature_set_fingerprint = _implementation_fingerprint(implementation)
            if feature_set_fingerprint and feature_set_fingerprint not in emitted_featureset_keys:
                records.append(_neuralsignal_featureset_memory_record(
                    profile=profile,
                    proposal_name=proposal_name,
                    implementation=implementation,
                    dataset_name=dataset_name,
                    detector=detector,
                    feature_set_fingerprint=feature_set_fingerprint,
                ))
                emitted_featureset_keys.add(feature_set_fingerprint)

        for proposal_name, artifact in artifacts_by_name.items():
            proposal = proposals_by_name.get(proposal_name) or {}
            dataset_config = _as_dict(artifact.get("dataset_config"))
            dataset_name = str(
                artifact.get("dataset")
                or dataset_config.get("dataset")
                or proposal.get("dataset")
                or ""
            )
            detector = str(
                artifact.get("detector")
                or _first(dataset_config.get("detector_names") or [])
                or proposal.get("detector")
                or ""
            )
            implementation = implementations_by_name.get(proposal_name)
            dataset_fingerprint = _dataset_config_fingerprint(dataset_config) if dataset_config else ""
            if dataset_fingerprint and dataset_fingerprint not in emitted_dataset_keys:
                records.append(_neuralsignal_dataset_memory_record(
                    profile=profile,
                    proposal_name=proposal_name,
                    artifact=artifact,
                    dataset_config=dataset_config,
                    dataset_name=dataset_name,
                    detector=detector,
                    implementation=implementation,
                    dataset_fingerprint=dataset_fingerprint,
                ))
                emitted_dataset_keys.add(dataset_fingerprint)

        for result in results:
            proposal_name = str(result.get("proposal_name") or "unknown")
            proposal = proposals_by_name.get(proposal_name) or result.get("proposal") or {}
            artifact = artifacts_by_name.get(proposal_name) or _as_dict(result.get("artifact"))
            dataset_config = _as_dict(artifact.get("dataset_config"))
            model = models_by_name.get(proposal_name) or _as_dict(result.get("model"))
            model_config = _as_dict(result.get("model_config") or model.get("model_config"))
            feature_importance = _as_dict(result.get("feature_importance"))
            metrics = _as_dict(result.get("metrics"))
            lessons = _proposal_lessons_from_evaluation(proposal_name, evaluation_summary, feature_importance)
            assessment = _proposal_assessment_from_evaluation(proposal_name, evaluation_summary)
            hypothesis_supported = _proposal_hypothesis_supported_from_evaluation(proposal_name, evaluation_summary)
            implementation = implementations_by_name.get(proposal_name)
            class_name = (
                dataset_config.get("feature_set_class_name")
                or (implementation or {}).get("class_name")
                or ""
            )
            dataset_name = str(
                artifact.get("dataset")
                or dataset_config.get("dataset")
                or proposal.get("dataset")
                or ""
            )
            detector = str(
                artifact.get("detector")
                or _first(dataset_config.get("detector_names") or [])
                or proposal.get("detector")
                or ""
            )
            figure_paths = _as_dict(result.get("figure_paths") or model.get("figure_paths"))
            record_id = str(result.get("experiment_id") or proposal_name)
            dataset_fingerprint = _dataset_config_fingerprint(dataset_config)
            feature_set_fingerprint = _implementation_fingerprint(implementation)
            model_fingerprint = _model_config_fingerprint(model_config, dataset_fingerprint=dataset_fingerprint)

            record = {
                "record_id": record_id,
                "domain": profile.get("name", "neuralsignal"),
                "kind": "neuralsignal_experiment",
                "object_type": "experiment_result",
                "object_key": record_id,
                "object_role": "result",
                "schema_version": "1",
                "title": proposal_name,
                "summary": _neuralsignal_memory_summary(
                    direction=direction,
                    proposal_name=proposal_name,
                    dataset_name=dataset_name,
                    detector=detector,
                    class_name=class_name,
                    metrics=metrics,
                    assessment=assessment,
                    lessons=lessons,
                ),
                "content": {
                    "proposal": proposal,
                    "dataset_artifact": _serializable_artifact_summary(artifact) if artifact else {},
                    "dataset_config": dataset_config,
                    "model_config": model_config,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "artifacts": _as_dict(result.get("artifacts")),
                    "figure_paths": figure_paths,
                    "mlflow": {
                        "run_id": result.get("mlflow_run_id") or model.get("mlflow_run_id") or "",
                        "tracking_uri": str(getattr(get_config(), "mlflow_uri", "") or ""),
                        "experiment_name": str((profile.get("storage") or {}).get("mlflow_experiment", "researcher_experiments")),
                    },
                    "implementation": _implementation_summary(implementations_by_name.get(proposal_name)),
                    "evaluation_summary": evaluation_summary,
                    "root_run_family_id": root_run_family_id,
                    "root_research_direction": root_research_direction,
                },
                "metadata": {
                    "profile": profile.get("name", "neuralsignal"),
                    "memory_kind": "neuralsignal_experiment",
                    "experiment_id": result.get("experiment_id", ""),
                    "proposal_name": proposal_name,
                    "research_direction": direction,
                    "dataset": dataset_name,
                    "detector": detector,
                    "feature_set_class_name": class_name,
                    "dataset_config_fingerprint": dataset_fingerprint,
                    "feature_set_fingerprint": feature_set_fingerprint,
                    "model_config_fingerprint": model_fingerprint,
                    "assessment": assessment,
                    "hypothesis_supported": hypothesis_supported,
                    "lessons": lessons,
                    "root_run_family_id": root_run_family_id,
                    "root_research_direction": root_research_direction,
                    "mlflow_run_id": result.get("mlflow_run_id") or model.get("mlflow_run_id") or "",
                    "mlflow_tracking_uri": str(getattr(get_config(), "mlflow_uri", "") or ""),
                    "mlflow_experiment": str((profile.get("storage") or {}).get("mlflow_experiment", "researcher_experiments")),
                    "source_next_step_record_id": source_next_step_record_id,
                    "source_next_step_title": source_next_step_title,
                    "figure_names": sorted(figure_paths.keys()),
                    "feature_importance_keys": sorted(feature_importance.keys())[:20],
                    **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                },
                "tags": [
                    "neuralsignal",
                    dataset_name or "dataset_unknown",
                    detector or "detector_unknown",
                ],
                "created_at": _now_iso(),
                "source_run_id": result.get("mlflow_run_id") or model.get("mlflow_run_id"),
                "blob_refs": _neuralsignal_blob_refs(artifact),
                "entities": _neuralsignal_entities(
                    proposal_name=proposal_name,
                    dataset_name=dataset_name,
                    detector=detector,
                    class_name=class_name,
                    model_name=str(model.get("model_id") or model_config.get("model_name") or ""),
                ),
                "relations": _neuralsignal_relations(
                    experiment_id=record_id,
                    proposal_name=proposal_name,
                    dataset_name=dataset_name,
                    detector=detector,
                    class_name=class_name,
                    model_name=str(model.get("model_id") or model_config.get("model_name") or ""),
                    source_next_step_record_id=source_next_step_record_id,
                    source_next_step_title=source_next_step_title,
                ),
            }
            records.append(record)
        return records

    def _persist_available_memory(self, profile: dict[str, Any], state: dict[str, Any], delta: dict[str, Any]) -> None:
        merged = {**state, **delta}
        records = dedupe_memory_records([
            *build_core_memory_records(profile, merged),
            *(self.build_memory_records(profile, merged) or []),
        ])
        if not records:
            log.debug("NeuralSignalPlugin | No memory records to persist for incremental state")
            return
        log.info(
            "NeuralSignalPlugin | Persisting %d memory record(s) immediately for available results",
            len(records),
        )
        MemoryService.for_profile(profile).persist_records(records)

    def memory_record_to_artifact(
        self,
        profile: dict[str, Any],
        record: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Render a NeuralSignal memory record into research-artifact form."""
        artifact = default_memory_record_to_artifact(record, source_name="memory")
        metadata = dict(artifact.get("metadata") or {})
        dataset = metadata.get("dataset")
        detector = metadata.get("detector")
        class_name = metadata.get("feature_set_class_name")
        primary_metric = (profile.get("evaluation") or {}).get("primary_metric", "test_auc")
        metric_value = metadata.get(primary_metric)

        title_parts = [str(record.get("title") or record.get("record_id") or "memory")]
        if dataset:
            title_parts.append(f"dataset={dataset}")
        if detector:
            title_parts.append(f"detector={detector}")
        artifact["title"] = " | ".join(title_parts)

        summary_lines = [str(record.get("summary") or artifact.get("summary") or "").strip()]
        if class_name:
            summary_lines.append(f"Feature set class: {class_name}")
        if metric_value is not None:
            summary_lines.append(f"{primary_metric}: {metric_value}")
        if metadata.get("mlflow_run_id"):
            summary_lines.append(f"MLflow run: {metadata['mlflow_run_id']}")
        artifact["summary"] = "\n".join(line for line in summary_lines if line)
        artifact["source_type"] = record.get("kind", "neuralsignal_experiment")
        artifact.pop("source", None)
        artifact["metadata"] = metadata
        artifact["raw"] = record
        return artifact

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
        dataset_output_dir = Path(cfg.experiments_dir) / profile.get("name", "neuralsignal") / "datasets"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

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
            "backend_config": _dataset_backend_config(cfg, profile, dataset_meta, proposal),
            "zone_size": hyperparameters.get("zone_size", 1024),
            "dataset_row_limit": int(hyperparameters.get("dataset_row_limit", hyperparameters.get("row_limit", dataset_meta.get("row_limit", 0))) or 0),
            "row_limit": int(hyperparameters.get("dataset_row_limit", hyperparameters.get("row_limit", dataset_meta.get("row_limit", 0))) or 0),
            "write_to_file": True,
            "build_in_memory": False,
            "use_gt_as_target": True,
            "query": proposal.get("mongo_query") or {},
            "balanced_target": hyperparameters.get(
                "balanced_target",
                dataset_meta.get(
                    "balanced_target",
                    {"enabled": True, "field": "ground_truth", "values": [0, 1]},
                ),
            ),
            "file_out": _csv_filename(f"{feature_set_name}_{detector or 'detector'}"),
            "dataset_output_dir": str(dataset_output_dir.resolve()),
            "feature_set_class_path": str(Path(script_path).resolve()) if script_path else "",
            "feature_set_class_name": class_name,
            "feature_set_source_hash": _implementation_fingerprint(implementation),
            "feature_set_configs": None,
            "ffn_layer_patterns": layer_patterns.get("ffn", []),
            "attn_layer_patterns": layer_patterns.get("attn", []),
            "proposal_name": proposal_name,
            "overwrite_existing_dataset": bool(dataset_meta.get("overwrite_existing_dataset", False)),
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
        dataset_path = artifact.get("dataset_path") or artifact.get("file_path", "")
        dataset_filename = Path(str(dataset_path)).name if dataset_path else ""
        feature_set_class_path = str(dataset_cfg.get("feature_set_class_path") or "")
        feature_set_class_name = str(dataset_cfg.get("feature_set_class_name") or "")
        if not feature_set_class_path or not feature_set_class_name:
            raise RuntimeError(
                f"Proposal {proposal_name!r} is missing implementation metadata for model execution"
            )

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
            # NeuralSignal's current create_s1_model implementation overwrites
            # cfg["dataset_path"] from cfg["file_out"] after sanitizing it with
            # get_name_from_template(..., file_safe=True). Pass only the dataset
            # filename here and run the task from the dataset directory.
            "file_out": dataset_filename,
            "feature_set_class_path": feature_set_class_path,
            "feature_set_class_name": feature_set_class_name,
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
            "dataset_path": dataset_path,
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
        cwd: str | None = None,
    ) -> dict[str, Any]:
        cfg = get_config()
        jobs_dir = Path(cfg.experiments_dir) / profile.get("name", "neuralsignal") / "jobs"
        job_id = _slug("_".join(item for item in (stage, proposal_name, artifact_id or experiment_id or str(uuid4())[:8]) if item))
        runtime = self.external_runtime_spec(profile, "experiment")
        return {
            "job_id": job_id,
            "job_dir": str(jobs_dir / job_id),
            "plugin_name": profile.get("name", _PLUGIN_NAME),
            "stage": stage,
            "proposal_name": proposal_name,
            "artifact_id": artifact_id,
            "experiment_id": experiment_id,
            "task_path": task_path,
            "payload": payload,
            "python": runtime["python"],
            "cwd": cwd or runtime["cwd"],
            "pythonpath_entries": list(runtime.get("pythonpath_entries") or []),
            "logger_prefixes": list(runtime.get("logger_prefixes") or []),
            "timeout": _task_timeout(profile, stage, cfg),
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
            artifacts.append(
                _dataset_artifact(
                    proposal_name=job.get("proposal_name", "unknown"),
                    file_path=file_path,
                    cwd=cwd,
                    dataset_cfg=payload,
                    task_result=result,
                    idx=idx,
                )
            )
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
        profile: dict[str, Any],
        task_path: str,
        payload: dict[str, Any],
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Run a dotted task path in a NeuralSignal-capable subprocess."""
        _ensure_plugin_logging()
        cfg = get_config()
        timeout = timeout or cfg.experiment_timeout_seconds
        return run_task(
            {
                "task_path": task_path,
                "payload": payload,
                "python": cfg.neuralsignal_python,
                "timeout": timeout,
                "plugin_name": _PLUGIN_NAME,
                "logger_prefixes": list(_PLUGIN_LOGGER_PREFIXES),
                "cwd": cwd,
                "pythonpath_entries": [str(path) for path in _pythonpath_entries(cfg)],
                "job_id": f"ns_{Path(task_path).name}_{uuid4().hex[:8]}",
                "job_dir": str(Path(cfg.experiments_dir) / _PLUGIN_NAME / "sync_tasks" / uuid4().hex[:8]),
            },
            profile=profile,
            purpose="task",
            default_runner="sync",
        )


def get_adapter() -> NeuralSignalPlugin:
    """Factory used by ``plugins.loader.load_adapter``."""
    return NeuralSignalPlugin()


def _implementations_by_proposal(implementations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        impl.get("proposal_name", ""): impl
        for impl in implementations
        if impl.get("proposal_name")
    }


def _validation_by_proposal(validation_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item.get("proposal_name", ""): item
        for item in validation_results
        if item.get("proposal_name")
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
        "stored_artifact_id": implementation.get("stored_artifact_id"),
        "stored_artifact_uri": implementation.get("stored_artifact_uri"),
    }


def _require_async_implementation(
    proposal_name: str,
    implementation: dict[str, Any] | None,
    validation: dict[str, Any] | None = None,
) -> None:
    if not implementation:
        raise RuntimeError(f"proposal {proposal_name!r} has no generated implementation")
    implementation_error = str(implementation.get("error") or "").strip()
    if implementation_error:
        raise RuntimeError(
            f"proposal {proposal_name!r} implementation generation failed: {implementation_error}"
        )
    script_path = str(implementation.get("script_path") or "").strip()
    class_name = str(implementation.get("class_name") or "").strip()
    if not script_path or not class_name:
        raise RuntimeError(
            f"proposal {proposal_name!r} is missing implementation script_path/class_name"
        )
    if not Path(script_path).exists():
        raise RuntimeError(
            f"proposal {proposal_name!r} implementation script does not exist: {script_path}"
        )
    if validation is not None and validation.get("passed") is False:
        summary = str(validation.get("test_output") or "").strip()
        if len(summary) > 280:
            summary = summary[:277] + "..."
        raise RuntimeError(
            f"proposal {proposal_name!r} did not pass validation"
            + (f": {summary}" if summary else "")
        )


def _proposal_analysis_from_evaluation(proposal_name: str, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    llm_analysis = evaluation_summary.get("llm_analysis") or {}
    for item in llm_analysis.get("per_proposal") or []:
        if item.get("proposal_name") == proposal_name:
            return item
    return {}


def _proposal_assessment_from_evaluation(proposal_name: str, evaluation_summary: dict[str, Any]) -> str:
    return str(_proposal_analysis_from_evaluation(proposal_name, evaluation_summary).get("assessment") or "")


def _proposal_hypothesis_supported_from_evaluation(proposal_name: str, evaluation_summary: dict[str, Any]) -> bool | None:
    analysis = _proposal_analysis_from_evaluation(proposal_name, evaluation_summary)
    if "hypothesis_supported" in analysis:
        return bool(analysis.get("hypothesis_supported"))
    return None


def _proposal_lessons_from_evaluation(
    proposal_name: str,
    evaluation_summary: dict[str, Any],
    feature_importance: dict[str, Any],
) -> list[str]:
    analysis = _proposal_analysis_from_evaluation(proposal_name, evaluation_summary)
    lessons: list[str] = []
    interpretation = analysis.get("interpretation")
    if interpretation:
        lessons.append(str(interpretation))
    for feature in analysis.get("key_features") or []:
        lessons.append(f"Important feature: {feature}")
    for feature in sorted(feature_importance.keys())[:3]:
        lessons.append(f"Feature importance tracked: {feature}")
    return lessons


def _neuralsignal_memory_summary(
    *,
    direction: str,
    proposal_name: str,
    dataset_name: str,
    detector: str,
    class_name: str,
    metrics: dict[str, Any],
    assessment: str,
    lessons: list[str],
) -> str:
    lines = [
        f"Direction: {direction}",
        f"Proposal: {proposal_name}",
        f"Dataset: {dataset_name}",
        f"Detector: {detector}",
        f"Feature set class: {class_name}",
        f"Metrics: {metrics}",
    ]
    if assessment:
        lines.append(f"Assessment: {assessment}")
    if lessons:
        lines.append("Lessons:")
        lines.extend(f"- {lesson}" for lesson in lessons[:5])
    return "\n".join(line for line in lines if line and line.strip())


def _neuralsignal_blob_refs(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    uri = artifact.get("stored_artifact_uri")
    artifact_id = artifact.get("stored_artifact_id")
    if uri or artifact_id:
        refs.append({
            "name": "dataset_artifact",
            "uri": str(uri or ""),
            "artifact_id": str(artifact_id or ""),
            "metadata": {"kind": "dataset_artifact"},
        })
    return refs


def _neuralsignal_entities(
    *,
    proposal_name: str,
    dataset_name: str,
    detector: str,
    class_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    entities = [{
        "entity_type": "proposal",
        "key": proposal_name,
        "name": proposal_name,
        "metadata": {"domain": "neuralsignal"},
    }]
    if dataset_name:
        entities.append({
            "entity_type": "dataset",
            "key": dataset_name,
            "name": dataset_name,
            "metadata": {"domain": "neuralsignal"},
        })
    if detector:
        entities.append({
            "entity_type": "detector",
            "key": detector,
            "name": detector,
            "metadata": {"domain": "neuralsignal"},
        })
    if class_name:
        entities.append({
            "entity_type": "feature_set",
            "key": class_name,
            "name": class_name,
            "metadata": {"domain": "neuralsignal"},
        })
    if model_name:
        entities.append({
            "entity_type": "model",
            "key": model_name,
            "name": model_name,
            "metadata": {"domain": "neuralsignal"},
        })
    return entities


def _neuralsignal_relations(
    *,
    experiment_id: str,
    proposal_name: str,
    dataset_name: str,
    detector: str,
    class_name: str,
    model_name: str,
    source_next_step_record_id: str = "",
    source_next_step_title: str = "",
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    if source_next_step_title:
        relations.append({
            "relation_type": "inspires_proposal",
            "source_type": "next_step",
            "source_key": source_next_step_title,
            "target_type": "proposal",
            "target_key": proposal_name,
            "metadata": {
                "domain": "neuralsignal",
                "source_next_step_record_id": source_next_step_record_id,
            },
        })
    if experiment_id:
        relations.append({
            "relation_type": "executed_as",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "experiment_result",
            "target_key": experiment_id,
            "metadata": {"domain": "neuralsignal"},
        })
    if dataset_name:
        relations.append({
            "relation_type": "tested_on",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "dataset",
            "target_key": dataset_name,
            "metadata": {"domain": "neuralsignal"},
        })
    if detector:
        relations.append({
            "relation_type": "used_detector",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "detector",
            "target_key": detector,
            "metadata": {"domain": "neuralsignal"},
        })
    if class_name:
        relations.append({
            "relation_type": "implemented_by",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "feature_set",
            "target_key": class_name,
            "metadata": {"domain": "neuralsignal"},
        })
    if model_name:
        relations.append({
            "relation_type": "produced_model",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "model",
            "target_key": model_name,
            "metadata": {"domain": "neuralsignal"},
        })
    return relations


def _serializable_artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if not key.startswith("_")
    }


def _dataset_memory_spec(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset_cfg.get("dataset", ""),
        "application_name": dataset_cfg.get("application_name", ""),
        "sub_application_name": dataset_cfg.get("sub_application_name", ""),
        "detector_names": list(dataset_cfg.get("detector_names") or []),
        "query": _as_dict(dataset_cfg.get("query")),
        "row_limit": dataset_cfg.get("row_limit"),
        "dataset_row_limit": dataset_cfg.get("dataset_row_limit"),
        "balanced_target": _as_dict(dataset_cfg.get("balanced_target")),
        "zone_size": dataset_cfg.get("zone_size"),
        "feature_set_class_name": dataset_cfg.get("feature_set_class_name", ""),
        "feature_set_source_hash": dataset_cfg.get("feature_set_source_hash", ""),
        "feature_set_configs": _as_dict(dataset_cfg.get("feature_set_configs")),
        "ffn_layer_patterns": list(dataset_cfg.get("ffn_layer_patterns") or []),
        "attn_layer_patterns": list(dataset_cfg.get("attn_layer_patterns") or []),
        "backend_type": _as_dict(dataset_cfg.get("backend_config")).get("backend_type", ""),
    }


def _dataset_config_fingerprint(dataset_cfg: dict[str, Any]) -> str:
    return fingerprint_json(_dataset_memory_spec(dataset_cfg))


def _implementation_fingerprint(implementation: dict[str, Any] | None) -> str:
    if not implementation:
        return ""
    script_path = implementation.get("script_path")
    if not isinstance(script_path, str) or not script_path or not Path(script_path).exists():
        return fingerprint_json({
            "class_name": implementation.get("class_name", ""),
            "proposal_name": implementation.get("proposal_name", ""),
        })
    return fingerprint_json({
        "class_name": implementation.get("class_name", ""),
        "proposal_name": implementation.get("proposal_name", ""),
        "source": Path(script_path).read_text(encoding="utf-8"),
    })


def _model_config_fingerprint(model_config: dict[str, Any], *, dataset_fingerprint: str) -> str:
    return fingerprint_json({
        "dataset_fingerprint": dataset_fingerprint,
        "model_config": _as_dict(model_config),
    })


def _dataset_artifact_from_memory(
    *,
    profile: dict[str, Any],
    proposal_name: str,
    dataset_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        service = MemoryService.for_profile(profile)
        fingerprint = _dataset_config_fingerprint(dataset_cfg)
        reuse_lookup = getattr(service, "find_reusable", None)
        reuse = reuse_lookup(
            domain=str(profile.get("name", "neuralsignal")),
            object_type="dataset",
            fingerprint=fingerprint,
            fingerprint_metadata_key="dataset_config_fingerprint",
            status_metadata_key="dataset_status",
            ready_statuses=["ready"],
        ) if callable(reuse_lookup) else None
        if isinstance(reuse, dict):
            record = reuse.get("record") if reuse.get("reusable") else None
        else:
            record = service.find_one_record({
                "domain": profile.get("name", "neuralsignal"),
                "object_type": "dataset",
                "metadata.dataset_config_fingerprint": fingerprint,
                "metadata.dataset_status": "ready",
            })
    except Exception as exc:
        log.debug("NeuralSignalPlugin | memory dataset lookup failed for %s: %s", proposal_name, exc)
        return None

    if not record:
        return None

    content = _as_dict(record.get("content"))
    stored_artifact = _as_dict(content.get("dataset_artifact"))
    candidate_paths = [
        stored_artifact.get("dataset_path"),
        stored_artifact.get("file_path"),
        stored_artifact.get("stored_artifact_uri"),
        stored_artifact.get("stored_artifact_path"),
    ]
    resolved_path = None
    for value in candidate_paths:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if path.exists():
            resolved_path = path
            break
    if resolved_path is None:
        return None

    metadata = _csv_metadata(resolved_path)
    return {
        "artifact_id": f"{proposal_name}_dataset_memory_0",
        "artifact_type": "dataset",
        "dataset_id": str(record.get("object_key") or record.get("record_id") or proposal_name),
        "proposal_name": proposal_name,
        "dataset_source": "memory_reuse",
        "status": "ready" if metadata.get("exists") else "missing_file",
        "file_path": str(resolved_path),
        "dataset_path": str(resolved_path),
        "task_file_path": str(resolved_path),
        "dataset": dataset_cfg.get("dataset", ""),
        "detector": (dataset_cfg.get("detector_names") or [None])[0],
        "rows": metadata.get("rows"),
        "columns": metadata.get("columns"),
        "column_names": metadata.get("column_names", []),
        "dataset_config": dataset_cfg,
        "stored_artifact_id": stored_artifact.get("stored_artifact_id") or metadata.get("stored_artifact_id") or "",
        "stored_artifact_uri": stored_artifact.get("stored_artifact_uri") or str(resolved_path),
        "memory_record_id": record.get("record_id", ""),
        "task_result": {
            "reused_from_memory": True,
            "memory_record_id": record.get("record_id", ""),
            "matched_dataset_config_fingerprint": _dataset_config_fingerprint(dataset_cfg),
        },
    }


def _neuralsignal_dataset_memory_record(
    *,
    profile: dict[str, Any],
    proposal_name: str,
    artifact: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset_name: str,
    detector: str,
    implementation: dict[str, Any] | None,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    class_name = str(
        dataset_config.get("feature_set_class_name")
        or (implementation or {}).get("class_name")
        or ""
    )
    stored_artifact_uri = artifact.get("stored_artifact_uri") or artifact.get("dataset_path") or artifact.get("file_path") or ""
    return {
        "record_id": f"dataset:{dataset_fingerprint}",
        "domain": profile.get("name", "neuralsignal"),
        "kind": "neuralsignal_dataset",
        "object_type": "dataset",
        "object_key": dataset_fingerprint,
        "object_role": "artifact",
        "schema_version": "1",
        "title": f"{proposal_name} dataset",
        "summary": (
            f"Dataset: {dataset_name}\n"
            f"Detector: {detector}\n"
            f"Feature set class: {class_name}\n"
            f"Rows: {artifact.get('rows')}\n"
            f"Columns: {artifact.get('columns')}"
        ),
        "content": {
            "dataset_artifact": _serializable_artifact_summary(artifact),
            "dataset_config": dataset_config,
            "implementation": _implementation_summary(implementation),
        },
        "metadata": {
            "profile": profile.get("name", "neuralsignal"),
            "memory_kind": "neuralsignal_dataset",
            "dataset": dataset_name,
            "detector": detector,
            "proposal_name": proposal_name,
            "feature_set_class_name": class_name,
            "dataset_source": artifact.get("dataset_source", ""),
            "dataset_status": artifact.get("status", ""),
            "rows": artifact.get("rows"),
            "columns": artifact.get("columns"),
            "dataset_config_fingerprint": dataset_fingerprint,
            "stored_artifact_id": artifact.get("stored_artifact_id", ""),
            "stored_artifact_uri": stored_artifact_uri,
        },
        "tags": ["neuralsignal", "dataset", dataset_name or "dataset_unknown", detector or "detector_unknown"],
        "created_at": _now_iso(),
        "blob_refs": _memory_blob_refs_from_artifact("dataset_artifact", artifact),
        "entities": _neuralsignal_entities(
            proposal_name=proposal_name,
            dataset_name=dataset_name,
            detector=detector,
            class_name=class_name,
            model_name="",
        ),
        "relations": _neuralsignal_relations(
            experiment_id="",
            proposal_name=proposal_name,
            dataset_name=dataset_name,
            detector=detector,
            class_name=class_name,
            model_name="",
        ),
    }


def _neuralsignal_featureset_memory_record(
    *,
    profile: dict[str, Any],
    proposal_name: str,
    implementation: dict[str, Any],
    dataset_name: str,
    detector: str,
    feature_set_fingerprint: str,
) -> dict[str, Any]:
    class_name = str(implementation.get("class_name") or proposal_name)
    return {
        "record_id": f"featureset:{feature_set_fingerprint}",
        "domain": profile.get("name", "neuralsignal"),
        "kind": "neuralsignal_featureset",
        "object_type": "featureset",
        "object_key": class_name,
        "object_role": "implementation",
        "schema_version": "1",
        "title": class_name,
        "summary": (
            f"Feature set class: {class_name}\n"
            f"Proposal: {proposal_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Detector: {detector}\n"
            f"Validated: {implementation.get('validated')}"
        ),
        "content": {
            "implementation": _implementation_summary(implementation),
        },
        "metadata": {
            "profile": profile.get("name", "neuralsignal"),
            "memory_kind": "neuralsignal_featureset",
            "proposal_name": proposal_name,
            "dataset": dataset_name,
            "detector": detector,
            "feature_set_class_name": class_name,
            "feature_set_fingerprint": feature_set_fingerprint,
            "validated": bool(implementation.get("validated")),
            "script_path": implementation.get("script_path", ""),
            "stored_artifact_id": implementation.get("stored_artifact_id", ""),
            "stored_artifact_uri": implementation.get("stored_artifact_uri", ""),
        },
        "tags": ["neuralsignal", "featureset", dataset_name or "dataset_unknown", detector or "detector_unknown"],
        "created_at": _now_iso(),
        "blob_refs": _memory_blob_refs_from_artifact("implementation_artifact", implementation),
        "entities": _neuralsignal_entities(
            proposal_name=proposal_name,
            dataset_name=dataset_name,
            detector=detector,
            class_name=class_name,
            model_name="",
        ),
        "relations": _neuralsignal_relations(
            experiment_id="",
            proposal_name=proposal_name,
            dataset_name=dataset_name,
            detector=detector,
            class_name=class_name,
            model_name="",
        ),
    }


def _neuralsignal_model_memory_record(
    *,
    profile: dict[str, Any],
    proposal_name: str,
    model: dict[str, Any],
    model_config: dict[str, Any],
    dataset_name: str,
    detector: str,
    dataset_fingerprint: str,
    model_fingerprint: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    model_name = str(model.get("model_id") or model_config.get("model_name") or proposal_name)
    return {
        "record_id": f"model:{model_fingerprint or model_name}",
        "domain": profile.get("name", "neuralsignal"),
        "kind": "neuralsignal_model",
        "object_type": "model",
        "object_key": model_name,
        "object_role": "artifact",
        "schema_version": "1",
        "title": model_name,
        "summary": (
            f"Model: {model_name}\n"
            f"Proposal: {proposal_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Detector: {detector}\n"
            f"Metrics: {metrics}"
        ),
        "content": {
            "model": _json_safe(model),
            "model_config": model_config,
            "metrics": metrics,
            "stored_figure_artifacts": _json_safe(_as_list(model.get("stored_figure_artifacts"))),
        },
        "metadata": {
            "profile": profile.get("name", "neuralsignal"),
            "memory_kind": "neuralsignal_model",
            "proposal_name": proposal_name,
            "dataset": dataset_name,
            "detector": detector,
            "model_name": model_name,
            "dataset_config_fingerprint": dataset_fingerprint,
            "model_config_fingerprint": model_fingerprint,
            "stored_artifact_id": model.get("stored_artifact_id", ""),
            "stored_artifact_uri": model.get("stored_artifact_uri", ""),
            "figure_artifact_ids": [item.get("artifact_id", "") for item in _as_list(model.get("stored_figure_artifacts")) if isinstance(item, dict)],
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        },
        "tags": ["neuralsignal", "model", dataset_name or "dataset_unknown", detector or "detector_unknown"],
        "created_at": _now_iso(),
        "blob_refs": (
            _memory_blob_refs_from_artifact("model_artifact", model)
            + _blob_refs_from_artifact_records(_as_list(model.get("stored_figure_artifacts")))
        ),
        "entities": _neuralsignal_entities(
            proposal_name=proposal_name,
            dataset_name=dataset_name,
            detector=detector,
            class_name="",
            model_name=model_name,
        ),
        "relations": _neuralsignal_relations(
            experiment_id="",
            proposal_name=proposal_name,
            dataset_name=dataset_name,
            detector=detector,
            class_name="",
            model_name=model_name,
        ),
    }


def _memory_blob_refs_from_artifact(kind: str, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    uri = artifact.get("stored_artifact_uri")
    artifact_id = artifact.get("stored_artifact_id")
    if not isinstance(uri, str) or not uri:
        return []
    if "model" in kind:
        content_type = "application/json"
    elif "implementation" in kind:
        content_type = "text/x-python"
    else:
        content_type = "text/csv"
    return [{
        "blob_id": str(artifact_id or uri),
        "name": kind,
        "uri": uri,
        "artifact_id": str(artifact_id or ""),
        "content_type": content_type,
        "metadata": {},
    }]


def _blob_refs_from_artifact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        uri = record.get("uri")
        if not isinstance(uri, str) or not uri:
            continue
        refs.append({
            "blob_id": str(record.get("artifact_id") or uri),
            "name": str(record.get("name") or "artifact"),
            "uri": uri,
            "artifact_id": str(record.get("artifact_id") or ""),
            "content_type": str(record.get("mime_type") or ""),
            "metadata": {
                "artifact_type": str(record.get("artifact_type") or ""),
                "figure_name": str(record.get("figure_name") or ""),
            },
        })
    return refs


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dataset_artifact(
    *,
    proposal_name: str,
    file_path: Any,
    cwd: str | os.PathLike[str] | None,
    dataset_cfg: dict[str, Any],
    task_result: dict[str, Any],
    idx: int,
    implementation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_path = _resolve_task_path(file_path, cwd)
    metadata = _csv_metadata(resolved_path)
    dataset_source = "existing" if task_result.get("skipped_existing_dataset") else "generated"
    artifact = {
        "artifact_id": f"{proposal_name}_dataset_{idx}",
        "artifact_type": "dataset",
        "dataset_id": f"{proposal_name}_{idx}",
        "proposal_name": proposal_name,
        "dataset_source": dataset_source,
        "status": "ready" if metadata.get("exists") else "missing_file",
        "file_path": str(resolved_path),
        "dataset_path": str(resolved_path),
        "task_file_path": str(file_path),
        "dataset": dataset_cfg.get("dataset", ""),
        "detector": (dataset_cfg.get("detector_names") or [None])[0],
        "rows": metadata.get("rows"),
        "columns": metadata.get("columns"),
        "column_names": metadata.get("column_names", []),
        "dataset_config": dataset_cfg,
        "task_result": _json_safe(task_result),
    }
    if implementation is not None:
        artifact["implementation"] = _implementation_summary(implementation)
    return artifact


def _register_dataset_artifact(profile: dict[str, Any], artifact: dict[str, Any], errors: list[str]) -> None:
    dataset_path = artifact.get("dataset_path") or artifact.get("file_path")
    if not dataset_path:
        return
    try:
        record = get_artifact_store(profile).store_file(
            dataset_path,
            artifact_type="dataset",
            profile_name=profile.get("name", ""),
            proposal_name=artifact.get("proposal_name", ""),
            experiment_id=artifact.get("experiment_id", ""),
            metadata={
                "dataset": artifact.get("dataset", ""),
                "detector": artifact.get("detector", ""),
                "rows": artifact.get("rows"),
                "columns": artifact.get("columns"),
                "column_names": artifact.get("column_names", []),
                "dataset_source": artifact.get("dataset_source", ""),
            },
            tags=["dataset", profile.get("name", "")],
        )
        artifact["stored_artifact_id"] = record["artifact_id"]
        artifact["stored_artifact_uri"] = record["uri"]
        artifact["stored_artifact_key"] = record.get("storage_key", "")
        artifact["stored_artifact_bucket"] = record.get("storage_bucket", "")
        artifact["stored_artifact_endpoint_url"] = record.get("storage_endpoint_url", "")
    except Exception as exc:
        log.warning("NeuralSignalPlugin | dataset artifact storage failed for %s: %s", artifact.get("proposal_name"), exc)
        errors.append(f"artifact_store: dataset {artifact.get('proposal_name', 'unknown')} failed: {exc}")


def _register_model_artifact(
    profile: dict[str, Any],
    result: dict[str, Any],
    model: dict[str, Any],
    errors: list[str],
) -> None:
    try:
        record = get_artifact_store(profile).store_json(
            {
                "experiment_result": result,
                "model": model,
            },
            artifact_type="model",
            profile_name=profile.get("name", ""),
            proposal_name=result.get("proposal_name", ""),
            experiment_id=result.get("experiment_id", ""),
            artifact_name=f"{_slug(result.get('proposal_name', 'model'))}_{result.get('experiment_id', '')[:8]}.json",
            metadata={
                "metrics": result.get("metrics", {}),
                "feature_importance_keys": sorted((result.get("feature_importance") or {}).keys()),
            },
            tags=["model", profile.get("name", "")],
        )
        model["stored_artifact_id"] = record["artifact_id"]
        model["stored_artifact_uri"] = record["uri"]
        model["stored_artifact_key"] = record.get("storage_key", "")
        model["stored_artifact_bucket"] = record.get("storage_bucket", "")
        model["stored_artifact_endpoint_url"] = record.get("storage_endpoint_url", "")
        result["stored_artifact_id"] = record["artifact_id"]
        result["stored_artifact_uri"] = record["uri"]
        result["stored_artifact_key"] = record.get("storage_key", "")
        result["stored_artifact_bucket"] = record.get("storage_bucket", "")
        result["stored_artifact_endpoint_url"] = record.get("storage_endpoint_url", "")
    except Exception as exc:
        log.warning("NeuralSignalPlugin | model artifact storage failed for %s: %s", result.get("proposal_name"), exc)
        errors.append(f"artifact_store: model {result.get('proposal_name', 'unknown')} failed: {exc}")


def _register_model_sidecar_artifacts(
    profile: dict[str, Any],
    *,
    result: dict[str, Any],
    model: dict[str, Any],
    figure_paths: dict[str, Any],
    errors: list[str],
) -> None:
    stored_figures: list[dict[str, Any]] = []
    for figure_name, figure_path in figure_paths.items():
        if not isinstance(figure_path, str) or not figure_path:
            continue
        path = Path(figure_path)
        if not path.exists():
            continue
        try:
            record = get_artifact_store(profile).store_file(
                path,
                artifact_type="model_figure",
                profile_name=profile.get("name", ""),
                proposal_name=result.get("proposal_name", ""),
                experiment_id=result.get("experiment_id", ""),
                artifact_name=path.name,
                metadata={
                    "figure_name": str(figure_name),
                    "model_name": model.get("model_id", ""),
                },
                tags=["model_figure", profile.get("name", "")],
            )
            stored_figures.append({
                "artifact_id": record["artifact_id"],
                "uri": record["uri"],
                "storage_key": record.get("storage_key", ""),
                "storage_bucket": record.get("storage_bucket", ""),
                "storage_endpoint_url": record.get("storage_endpoint_url", ""),
                "artifact_type": "model_figure",
                "mime_type": record.get("mime_type", ""),
                "name": f"figure:{figure_name}",
                "figure_name": str(figure_name),
                "source_path": str(path),
            })
        except Exception as exc:
            log.warning(
                "NeuralSignalPlugin | model figure artifact storage failed for %s (%s): %s",
                result.get("proposal_name"),
                figure_name,
                exc,
            )
            errors.append(
                f"artifact_store: model_figure {result.get('proposal_name', 'unknown')} {figure_name} failed: {exc}"
            )
    if stored_figures:
        model["stored_figure_artifacts"] = stored_figures
        result["stored_figure_artifacts"] = stored_figures


def _log_result_to_mlflow(
    *,
    profile: dict[str, Any],
    state: dict[str, Any],
    artifact: dict[str, Any],
    experiment_id: str,
    proposal_name: str,
    metrics: dict[str, Any],
    params: dict[str, Any],
    feature_importance: dict[str, Any],
    artifacts_payload: dict[str, Any],
    model_config_payload: dict[str, Any],
    figure_paths: dict[str, Any],
    model: dict[str, Any],
) -> str:
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    experiment_name = storage_cfg.get("mlflow_experiment", "researcher_experiments")
    direction = str(state.get("research_direction", ""))
    proposal = next(
        (item for item in (state.get("proposals") or []) if item.get("name") == proposal_name),
        {},
    )

    try:
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{proposal_name}_{experiment_id[:8]}") as run:
            mlflow.log_params({
                "proposal_name": proposal_name,
                "profile": profile.get("name", ""),
                "research_direction": direction[:250],
                "dataset": artifact.get("dataset", ""),
                "detector": artifact.get("detector", ""),
                "model_name": model_config_payload.get("model", ""),
                **{k: v for k, v in params.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)},
            })
            numeric_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
            model_metrics = {
                f"model_{k}": float(v)
                for k, v in (model.get("metrics") or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if model_metrics:
                mlflow.log_metrics(model_metrics)
            mlflow.set_tags({
                "experiment_id": experiment_id,
                "profile": profile.get("name", ""),
                "source": "neuralsignal_execute_experiment",
                "proposal_name": proposal_name,
                **_string_tags(model_config_payload.get("tags")),
            })
            if feature_importance:
                mlflow.log_dict(_json_safe(feature_importance), "feature_importance.json")
            if params:
                mlflow.log_dict(_json_safe(params), "model_params.json")
            if artifacts_payload:
                mlflow.log_dict(_json_safe(artifacts_payload), "model_artifacts.json")
            if model_config_payload:
                mlflow.log_dict(_json_safe(model_config_payload), "model_config.json")
            dataset_config = _as_dict(artifact.get("dataset_config"))
            if dataset_config:
                mlflow.log_dict(_json_safe(dataset_config), "dataset_config.json")
            if proposal:
                mlflow.log_dict(_json_safe(proposal), "proposal.json")
            agent_state = _agent_state_payload(state, proposal_name)
            if agent_state:
                mlflow.log_dict(_json_safe(agent_state), "agent_state.json")
            description = model_config_payload.get("description")
            if isinstance(description, str) and description.strip():
                mlflow.log_text(description, "model_description.txt")
            _log_mlflow_figures(metrics, artifacts_payload, figure_paths)
            _log_mlflow_dataset_artifacts(artifact)
            return run.info.run_id
    except Exception as exc:
        log.warning("NeuralSignalPlugin | MLflow logging failed for %s: %s", proposal_name, exc)
        return ""


def _expected_dataset_path(dataset_cfg: dict[str, Any]) -> Path:
    output_dir = Path(str(dataset_cfg.get("dataset_output_dir") or ""))
    file_out = dataset_cfg.get("file_out")
    if not file_out:
        return output_dir
    return output_dir / str(file_out)


def _should_overwrite_existing_dataset(dataset_cfg: dict[str, Any]) -> bool:
    return bool(dataset_cfg.get("overwrite_existing_dataset", False))


def _agent_state_payload(state: dict[str, Any], proposal_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if state.get("research_direction"):
        payload["research_direction"] = state.get("research_direction")
    if state.get("research_summary"):
        payload["research_summary"] = state.get("research_summary")
    proposal = next((item for item in (state.get("proposals") or []) if item.get("name") == proposal_name), None)
    if proposal:
        payload["proposal"] = proposal
    plan = next((item for item in (state.get("implementation_plans") or []) if item.get("proposal_name") == proposal_name), None)
    if plan:
        payload["implementation_plan"] = plan
    implementation = next((item for item in (state.get("implementations") or []) if item.get("proposal_name") == proposal_name), None)
    if implementation:
        payload["implementation"] = _implementation_summary(implementation)
    validation = next((item for item in (state.get("validation_results") or []) if item.get("proposal_name") == proposal_name), None)
    if validation:
        payload["validation_result"] = validation
    artifacts = state.get("research_artifacts") or []
    if artifacts:
        payload["research_artifact_ids"] = [item.get("artifact_id") for item in artifacts[:20] if item.get("artifact_id")]
    return payload


def _log_mlflow_figures(metrics: dict[str, Any], artifacts_payload: dict[str, Any], figure_paths: dict[str, Any]) -> None:
    logged_figure_names = _log_mlflow_figure_files(figure_paths)
    logged_confusion = "confusion_matrix" in logged_figure_names
    roc_figure = _roc_figure(metrics, artifacts_payload)
    if roc_figure is not None:
        mlflow.log_figure(roc_figure, "auc_curve.png")
        roc_figure.clf()

    confusion_figure = _confusion_figure(metrics, artifacts_payload)
    if confusion_figure is not None and not logged_confusion:
        mlflow.log_figure(confusion_figure, "confusion_matrix.png")
        confusion_figure.clf()


def _log_mlflow_figure_files(figure_paths: dict[str, Any]) -> set[str]:
    logged: set[str] = set()
    for name, path_value in figure_paths.items():
        if not isinstance(path_value, str) or not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        try:
            mlflow.log_artifact(str(path), artifact_path="figures")
            logged.add(str(name))
        except Exception as exc:
            log.debug("NeuralSignalPlugin | failed to log figure artifact %s: %s", path, exc)
    return logged


def _log_mlflow_dataset_artifacts(artifact: dict[str, Any]) -> None:
    dataset_path = artifact.get("dataset_path") or artifact.get("file_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        return
    path = Path(dataset_path)
    if not path.exists():
        return
    try:
        mlflow.log_artifact(str(path), artifact_path="dataset")
    except Exception as exc:
        log.debug("NeuralSignalPlugin | failed to log dataset artifact %s: %s", path, exc)


def _roc_figure(metrics: dict[str, Any], artifacts_payload: dict[str, Any]) -> Any | None:
    fpr = _float_list(
        artifacts_payload.get("roc_curve_fpr")
        or artifacts_payload.get("fpr")
        or ((artifacts_payload.get("roc_curve") or {}).get("fpr"))
    )
    tpr = _float_list(
        artifacts_payload.get("roc_curve_tpr")
        or artifacts_payload.get("tpr")
        or ((artifacts_payload.get("roc_curve") or {}).get("tpr"))
    )
    auc_value = _maybe_float(metrics.get("test_auc"))
    if not fpr or not tpr or len(fpr) != len(tpr):
        if auc_value is None:
            return None
        return _auc_summary_figure(auc_value, _maybe_float(metrics.get("train_auc")))

    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc_value:.3f})" if auc_value is not None else "ROC")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AUC / ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _auc_summary_figure(test_auc: float, train_auc: float | None) -> Any | None:
    plt = _plt()
    if plt is None:
        return None
    labels = ["test_auc"]
    values = [test_auc]
    if train_auc is not None:
        labels.append("train_auc")
        values.append(train_auc)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(labels, values, color=["#2563eb", "#64748b"][: len(values)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("AUC Summary")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def _confusion_figure(metrics: dict[str, Any], artifacts_payload: dict[str, Any]) -> Any | None:
    matrix = _confusion_matrix_values(metrics, artifacts_payload)
    if matrix is None:
        return None
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def _confusion_matrix_values(metrics: dict[str, Any], artifacts_payload: dict[str, Any]) -> list[list[int]] | None:
    matrix = artifacts_payload.get("confusion_matrix")
    if isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix):
        try:
            return [[int(matrix[0][0]), int(matrix[0][1])], [int(matrix[1][0]), int(matrix[1][1])]]
        except Exception:
            return None

    tn = _maybe_int(artifacts_payload.get("tn", metrics.get("test_tn")))
    fp = _maybe_int(artifacts_payload.get("fp", metrics.get("test_fp")))
    fn = _maybe_int(artifacts_payload.get("fn", metrics.get("test_fn")))
    tp = _maybe_int(artifacts_payload.get("tp", metrics.get("test_tp")))
    if None not in (tn, fp, fn, tp):
        return [[tn, fp], [fn, tp]]
    return None


def _plt() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        log.debug("NeuralSignalPlugin | matplotlib unavailable for MLflow figures: %s", exc)
        return None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for item in value:
        try:
            output.append(float(item))
        except Exception:
            return []
    return output


def _maybe_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return float(value)
    except Exception:
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except Exception:
        return None


def _string_tags(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return {f"tag_{idx}": str(item) for idx, item in enumerate(value)}
    if value is None:
        return {}
    return {"tags": str(value)}


def _execution_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    return profile.get("execution") or {}


def _task_timeout(profile: dict[str, Any], stage: str, cfg: Any) -> int:
    execution = _execution_cfg(profile)
    stage_key = f"{stage}_timeout_seconds"
    if execution.get(stage_key):
        return int(execution[stage_key])
    if execution.get("job_timeout_seconds"):
        return int(execution["job_timeout_seconds"])
    return int(cfg.experiment_timeout_seconds)


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


def _replace_by_key(items: list[dict[str, Any]], item: dict[str, Any], key: str) -> list[dict[str, Any]]:
    marker = item.get(key)
    output: list[dict[str, Any]] = []
    replaced = False
    for existing in items:
        if existing.get(key) == marker:
            if not replaced:
                output.append(item)
                replaced = True
            continue
        output.append(existing)
    if not replaced:
        output.append(item)
    return output


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


def _dataset_backend_config(
    cfg: Any,
    profile: dict[str, Any],
    dataset_meta: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    storage = dataset_meta.get("storage") or {}
    hyperparameters = proposal.get("hyperparameters") or {}
    backend = _backend_config(cfg)
    for source in (profile, storage, dataset_meta, hyperparameters):
        backend.update(_scan_cache_config(source))
        backend.update(_as_dict(source.get("backend_config")))
    _ensure_valid_scan_cache_directory(backend, profile)
    return backend


def _scan_cache_config(source: dict[str, Any]) -> dict[str, Any]:
    direct_keys = (
        "cache_scan_on_load",
        "cache_scan_on_write",
        "scan_cache_size",
        "scan_hd_cache_size",
        "scan_cache_directory",
    )
    config = {key: source[key] for key in direct_keys if key in source}
    scan_cache = _as_dict(source.get("scan_cache"))
    if not scan_cache:
        return config
    if "enabled" in scan_cache and not bool(scan_cache["enabled"]):
        config["scan_cache_size"] = 0
        config["scan_hd_cache_size"] = 0
    aliases = {
        "cache_scan_on_load": ("cache_scan_on_load", "on_load"),
        "cache_scan_on_write": ("cache_scan_on_write", "on_write"),
        "scan_cache_size": ("scan_cache_size", "memory_size", "ram_size"),
        "scan_hd_cache_size": ("scan_hd_cache_size", "disk_size", "hd_size"),
        "scan_cache_directory": ("scan_cache_directory", "directory", "path"),
    }
    for target, keys in aliases.items():
        for key in keys:
            if key in scan_cache:
                config[target] = scan_cache[key]
                break
    return config


def _ensure_valid_scan_cache_directory(backend: dict[str, Any], profile: dict[str, Any]) -> None:
    path_value = str(backend.get("scan_cache_directory") or "").strip()
    if not path_value or not _path_root_exists(path_value):
        path = _default_scan_cache_directory(profile)
        path.mkdir(parents=True, exist_ok=True)
        backend["scan_cache_directory"] = str(path)


def _default_scan_cache_directory(profile: dict[str, Any]) -> Path:
    for preferred in (Path("F:/temp"), Path("/tmp")):
        if _path_root_exists(preferred.as_posix()):
            return preferred
    return dev_path("scan_cache", str(profile.get("name") or "neuralsignal")).resolve()


def _path_root_exists(path_value: str) -> bool:
    path = Path(path_value).expanduser()
    anchor = path.anchor
    if anchor:
        return Path(anchor).exists()
    return True


def _neuralsignal_workdir(cfg: Any) -> Path:
    configured = Path(getattr(cfg, "neuralsignal_src_path", "")).resolve()
    if _is_package_dir(configured):
        return configured.parent
    return configured


def _pythonpath_entries(cfg: Any) -> list[Path]:
    configured = Path(getattr(cfg, "neuralsignal_src_path", "")).resolve()
    workdir = _neuralsignal_workdir(cfg)
    researcher_root = _TASK_RUNNER.resolve().parents[2]

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


def _csv_filename(value: str) -> str:
    slug = _slug(value)
    return slug if slug.lower().endswith(".csv") else f"{slug}.csv"


def _model_task_workdir(artifact: dict[str, Any], cfg: Any) -> Path:
    dataset_path = artifact.get("dataset_path") or artifact.get("file_path")
    if dataset_path:
        parent = Path(str(dataset_path)).resolve().parent
        if parent.exists():
            return parent
    return _neuralsignal_workdir(cfg)


def _first(items: list[Any]) -> Any:
    return items[0] if items else None
