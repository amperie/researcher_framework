"""Tests for the NeuralSignal research adapter.

The real NeuralSignal runtime is intentionally not imported here. These tests
mock the subprocess boundary and verify that the adapter builds payloads and
normalizes task outputs correctly.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from plugins.neuralsignal.adapter import NeuralSignalPlugin


def _cfg(tmp_path):
    ns_src = tmp_path / "neuralsignal_src"
    ns_src.mkdir()
    return SimpleNamespace(
        neuralsignal_src_path=str(ns_src),
        neuralsignal_python="python",
        experiment_timeout_seconds=30,
        mongo_url="mongodb://localhost:27017",
        mlflow_uri="http://localhost:5000",
        experiments_dir="dev/experiments",
    )


def _profile():
    return {
        "datasets": [
            {
                "name": "HaluBench",
                "storage": {
                    "application_name": "HaluBench",
                    "sub_application_name": "GranularAttention",
                },
                "available_detectors": ["hallucination"],
                "layer_name_patterns": {
                    "ffn": ["mlp", "fc"],
                    "attn": ["attn", ".q"],
                },
            }
        ],
        "evaluation": {"primary_metric": "test_auc"},
        "execution": {"job_timeout_seconds": 7200},
    }


def _proposal():
    return {
        "name": "activation_sparsity",
        "dataset": "HaluBench",
        "detector": "hallucination",
        "hyperparameters": {"zone_size": 512, "row_limit": 25},
        "mongo_query": {"split": "train"},
    }


def _implementation(tmp_path):
    script_path = tmp_path / "ActivationSparsity.py"
    script_path.write_text("class ActivationSparsity: pass\n", encoding="utf-8")
    return {
        "proposal_name": "activation_sparsity",
        "class_name": "ActivationSparsity",
        "script_path": str(script_path),
        "validated": True,
    }


def test_build_dataset_config_contains_neuralsignal_payload(tmp_path):
    adapter = NeuralSignalPlugin()
    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        payload = adapter._build_dataset_config(_profile(), _proposal(), _implementation(tmp_path))

    assert payload["create_dataset"] is True
    assert payload["create_s1_model"] is False
    assert payload["dataset"] == "HaluBench"
    assert payload["detector_names"] == ["hallucination"]
    assert payload["application_name"] == "HaluBench"
    assert payload["sub_application_name"] == "GranularAttention"
    assert payload["zone_size"] == 512
    assert payload["row_limit"] == 25
    assert payload["dataset_row_limit"] == 25
    assert payload["query"] == {"split": "train"}
    assert payload["balanced_target"] == {"enabled": True, "field": "ground_truth", "values": [0, 1]}
    assert payload["file_out"] == "activation_sparsity_hallucination.csv"
    assert payload["dataset_output_dir"] == str((Path("dev") / "experiments" / "neuralsignal" / "datasets").resolve())
    assert payload["feature_set_class_name"] == "ActivationSparsity"
    assert payload["feature_set_configs"] is None
    assert payload["ffn_layer_patterns"] == ["mlp", "fc"]
    assert payload["attn_layer_patterns"] == ["attn", ".q"]
    assert payload["backend_config"]["mongo_url"] == "mongodb://localhost:27017"


def test_prepare_experiment_runs_dataset_task_and_records_csv_metadata(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}

    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch.object(adapter, "_call_task", return_value={"file_paths": [str(csv_path)]}) as call_task:
            delta = adapter.prepare_experiment(_profile(), state)

    call_task.assert_called_once()
    assert call_task.call_args.args[0] == "plugins.neuralsignal.tasks.create_dataset"
    assert call_task.call_args.kwargs["timeout"] == 7200
    assert call_task.call_args.kwargs["cwd"] == str(tmp_path / "neuralsignal_src")
    assert delta["errors"] == []
    assert len(delta["experiment_artifacts"]) == 1
    artifact = delta["experiment_artifacts"][0]
    assert artifact["artifact_type"] == "dataset"
    assert artifact["status"] == "ready"
    assert artifact["rows"] == 2
    assert artifact["columns"] == 2
    assert artifact["column_names"] == ["a", "b"]
    assert delta["datasets"] == [artifact]


def test_prepare_experiment_normalizes_dataset_task_failure(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}

    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch.object(adapter, "_call_task", side_effect=RuntimeError("boom")):
            delta = adapter.prepare_experiment(_profile(), state)

    assert delta["experiment_artifacts"] == []
    assert any("activation_sparsity failed: boom" in error for error in delta["errors"])


def test_execute_experiment_runs_model_task_and_normalizes_result(tmp_path):
    adapter = NeuralSignalPlugin()
    artifact = {
        "artifact_id": "activation_sparsity_dataset_0",
        "artifact_type": "dataset",
        "status": "ready",
        "proposal_name": "activation_sparsity",
        "dataset_path": str(tmp_path / "features.csv"),
        "dataset": "HaluBench",
        "detector": "hallucination",
        "dataset_config": {
            "dataset": "HaluBench",
            "application_name": "HaluBench",
            "sub_application_name": "GranularAttention",
            "detector_names": ["hallucination"],
            "zone_size": 512,
            "feature_set_class_path": str(tmp_path / "ActivationSparsity.py"),
            "feature_set_class_name": "ActivationSparsity",
            "ffn_layer_patterns": ["mlp"],
            "attn_layer_patterns": ["attn"],
        },
    }
    task_result = {
        "metrics": {"test_auc": 0.72, "test_f1": 0.61},
        "params": {"max_depth": 3},
        "feature_importance": {"a": 0.8},
    }

    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch.object(adapter, "_call_task", return_value=task_result) as call_task:
            delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    call_task.assert_called_once()
    assert call_task.call_args.args[0] == "plugins.neuralsignal.tasks.create_s1_model"
    payload = call_task.call_args.args[1]
    assert call_task.call_args.kwargs["timeout"] == 7200
    assert call_task.call_args.kwargs["cwd"] == str(tmp_path)
    assert payload["dataset_path"] == str(tmp_path / "features.csv")
    assert payload["file_out"] == "features.csv"
    assert payload["optimization_metric"] == "test_auc"
    assert payload["feature_set_class_name"] == "ActivationSparsity"
    assert payload["feature_set_configs"] is None
    assert payload["ffn_layer_patterns"] == ["mlp"]
    assert delta["errors"] == []
    assert delta["experiment_results"][0]["metrics"]["test_auc"] == 0.72
    assert delta["experiment_results"][0]["feature_importance"] == {"a": 0.8}
    assert delta["models"][0]["params"] == {"max_depth": 3}
    assert delta["models"][0]["experiment_id"] == delta["experiment_results"][0]["experiment_id"]


def test_execute_experiment_records_not_ready_dataset_error():
    adapter = NeuralSignalPlugin()
    artifact = {
        "artifact_type": "dataset",
        "status": "missing_file",
        "proposal_name": "activation_sparsity",
    }

    delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    assert delta["experiment_results"] == []
    assert delta["models"] == []
    assert any("dataset artifact is not ready" in error for error in delta["errors"])


def test_task_timeout_prefers_stage_override_then_job_timeout(tmp_path):
    adapter = NeuralSignalPlugin()
    artifact = {
        "artifact_id": "activation_sparsity_dataset_0",
        "artifact_type": "dataset",
        "status": "ready",
        "proposal_name": "activation_sparsity",
        "dataset_path": str(tmp_path / "features.csv"),
        "dataset": "HaluBench",
        "detector": "hallucination",
        "dataset_config": {
            "dataset": "HaluBench",
            "application_name": "HaluBench",
            "sub_application_name": "GranularAttention",
            "detector_names": ["hallucination"],
            "zone_size": 512,
            "feature_set_class_path": str(tmp_path / "ActivationSparsity.py"),
            "feature_set_class_name": "ActivationSparsity",
        },
    }
    task_result = {"metrics": {"test_auc": 0.72}, "params": {}, "feature_importance": {}}
    profile = _profile()
    profile["execution"] = {
        "job_timeout_seconds": 7200,
        "model_timeout_seconds": 14400,
    }

    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch.object(adapter, "_call_task", return_value=task_result) as call_task:
            adapter.execute_experiment(profile, {"experiment_artifacts": [artifact]})

    assert call_task.call_args.kwargs["timeout"] == 14400


def test_submit_experiment_jobs_submits_dataset_job(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}
    runner = MagicMock()
    runner.submit.return_value = {
        "job_id": "dataset_activation_sparsity",
        "job_dir": str(tmp_path / "job"),
        "status": "submitted",
        "stage": "dataset",
        "proposal_name": "activation_sparsity",
    }

    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("plugins.neuralsignal.adapter.get_runner", return_value=runner):
            delta = adapter.submit_experiment_jobs(
                {**_profile(), "execution": {"runner": "local_process", "max_parallel_jobs": 1}},
                state,
            )

    runner.submit.assert_called_once()
    spec = runner.submit.call_args.args[0]
    assert spec["stage"] == "dataset"
    assert spec["task_path"] == "plugins.neuralsignal.tasks.create_dataset"
    assert spec["payload"]["dataset"] == "HaluBench"
    assert delta["experiment_jobs"][0]["status"] == "submitted"


def test_check_experiment_jobs_collects_dataset_and_submits_model_job(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    job_dir = tmp_path / "dataset_job"
    job_dir.mkdir()
    payload = {
        "dataset": "HaluBench",
        "detector_names": ["hallucination"],
        "application_name": "HaluBench",
        "sub_application_name": "GranularAttention",
        "zone_size": 512,
        "feature_set_class_path": str(tmp_path / "Feature.py"),
        "feature_set_class_name": "Feature",
    }
    (job_dir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps({
            "job_id": "dataset_job",
            "job_dir": str(job_dir),
            "task_path": "plugins.neuralsignal.tasks.create_dataset",
        }),
        encoding="utf-8",
    )
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps({"file_paths": [str(csv_path)]}), encoding="utf-8")

    runner = MagicMock()
    runner.check.return_value = {
        "job_id": "dataset_job",
        "job_dir": str(job_dir),
        "result_path": str(result_path),
        "status": "succeeded",
        "stage": "dataset",
        "proposal_name": "activation_sparsity",
    }
    runner.submit.return_value = {
        "job_id": "model_job",
        "job_dir": str(tmp_path / "model_job"),
        "status": "submitted",
        "stage": "model",
        "proposal_name": "activation_sparsity",
        "artifact_id": "activation_sparsity_dataset_0",
    }

    adapter = NeuralSignalPlugin()
    with patch("plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("plugins.neuralsignal.adapter.get_runner", return_value=runner):
            delta = adapter.check_experiment_jobs(
                {**_profile(), "execution": {"runner": "local_process", "max_parallel_jobs": 1}},
                {"experiment_jobs": [{"job_id": "dataset_job", "job_dir": str(job_dir)}]},
            )

    assert delta["experiment_artifacts"][0]["status"] == "ready"
    assert delta["experiment_artifacts"][0]["rows"] == 1
    runner.submit.assert_called_once()
    assert runner.submit.call_args.args[0]["stage"] == "model"
    assert runner.submit.call_args.args[0]["cwd"] == str(tmp_path)
    assert delta["submitted_jobs"][0]["job_id"] == "model_job"


def test_call_task_sets_neuralsignal_src_on_pythonpath(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "existing_path")

    class FakeProc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"ok": true}\n')
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    with patch("plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("plugins.neuralsignal.adapter.subprocess.Popen", return_value=FakeProc()) as popen:
            result = NeuralSignalPlugin()._call_task("some.module.task", {"x": 1}, cwd=str(tmp_path))

    assert result == {"ok": True}
    env = popen.call_args.kwargs["env"]
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == cfg.neuralsignal_src_path
    assert str(os.getcwd()) in pythonpath
    assert "existing_path" in pythonpath
    assert popen.call_args.args[0][-3:] == ["-m", "plugins.task_runner", "some.module.task"]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)


def test_call_task_uses_full_timeout_for_process_wait(tmp_path):
    cfg = _cfg(tmp_path)
    seen = {}

    class FakeProc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"ok": true}\n')
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            seen["timeout"] = timeout
            return self.returncode

    with patch("plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("plugins.neuralsignal.adapter.subprocess.Popen", return_value=FakeProc()):
            result = NeuralSignalPlugin()._call_task("some.module.task", {"x": 1}, timeout=123)

    assert result == {"ok": True}
    assert seen["timeout"] == 123


def test_call_task_supports_package_dir_as_neuralsignal_src_path(tmp_path, monkeypatch):
    package_dir = tmp_path / "repo" / "neuralsignal"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    cfg = _cfg(tmp_path)
    cfg.neuralsignal_src_path = str(package_dir)

    class FakeProc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"ok": true}\n')
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.delenv("PYTHONPATH", raising=False)
    with patch("plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("plugins.neuralsignal.adapter.subprocess.Popen", return_value=FakeProc()) as popen:
            result = NeuralSignalPlugin()._call_task("some.module.task", {"x": 1})

    assert result == {"ok": True}
    pythonpath = popen.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(package_dir.parent) in pythonpath
    assert str(package_dir) in pythonpath
    assert str(os.getcwd()) in pythonpath
    assert popen.call_args.args[0][-3:] == ["-m", "plugins.task_runner", "some.module.task"]
