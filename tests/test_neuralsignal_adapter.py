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

from core.plugins.neuralsignal import adapter as ns_adapter
from core.plugins.neuralsignal.adapter import NeuralSignalPlugin


def _cfg(tmp_path):
    ns_src = tmp_path / "neuralsignal_src"
    ns_src.mkdir()
    return SimpleNamespace(
        dev_root="dev",
        neuralsignal_src_path=str(ns_src),
        neuralsignal_python="python",
        experiment_timeout_seconds=30,
        mongo_url="mongodb://localhost:27017",
        mlflow_uri="http://localhost:5000",
        experiments_dir="dev/experiments",
        artifacts_db_name="researcher_artifacts",
        artifacts_collection="artifacts",
        artifact_store_backend="filesystem",
        artifact_store_root=str(tmp_path / "artifacts"),
    )


def _mock_artifact_store():
    store = MagicMock()
    def _store_file(*args, **kwargs):
        artifact_type = kwargs.get("artifact_type", "")
        artifact_name = kwargs.get("artifact_name", "")
        if artifact_type == "model_figure":
            return {
                "artifact_id": f"stored-figure-{artifact_name or '1'}",
                "uri": f"file:///stored/{artifact_name or 'figure.png'}",
                "storage_key": f"neuralsignal/model_figure/{artifact_name or 'figure.png'}",
                "storage_bucket": "researcher-artifacts",
                "storage_endpoint_url": "http://hp.lan:9000",
                "mime_type": "image/png",
            }
        if artifact_type == "implementation":
            return {
                "artifact_id": "stored-implementation-1",
                "uri": "file:///stored/implementation.py",
                "storage_key": "neuralsignal/implementation/stored-implementation-1/implementation.py",
                "storage_bucket": "researcher-artifacts",
                "storage_endpoint_url": "http://hp.lan:9000",
            }
        return {
            "artifact_id": "stored-dataset-1",
            "uri": "file:///stored/dataset.csv",
            "storage_key": "neuralsignal/dataset/stored-dataset-1/dataset.csv",
            "storage_bucket": "researcher-artifacts",
            "storage_endpoint_url": "http://hp.lan:9000",
        }

    store.store_file.side_effect = _store_file
    store.store_json.side_effect = lambda *args, **kwargs: {
        "artifact_id": "stored-model-1",
        "uri": "file:///stored/model.json",
        "storage_key": "neuralsignal/model/stored-model-1/model.json",
        "storage_bucket": "researcher-artifacts",
        "storage_endpoint_url": "http://hp.lan:9000",
    }
    return store


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
        "stored_artifact_id": "stored-implementation-1",
        "stored_artifact_uri": "file:///stored/implementation.py",
    }


def test_build_dataset_config_contains_neuralsignal_payload(tmp_path):
    adapter = NeuralSignalPlugin()
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
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
    assert payload["overwrite_existing_dataset"] is False
    assert payload["feature_set_class_name"] == "ActivationSparsity"
    assert payload["feature_set_source_hash"]
    assert payload["feature_set_configs"] is None
    assert payload["ffn_layer_patterns"] == ["mlp", "fc"]
    assert payload["attn_layer_patterns"] == ["attn", ".q"]
    assert payload["backend_config"]["mongo_url"] == "mongodb://localhost:27017"


def test_build_dataset_config_forwards_scan_cache_backend_options(tmp_path):
    adapter = NeuralSignalPlugin()
    profile = _profile()
    profile["backend_config"] = {"cache_scan_on_write": False}
    profile["datasets"][0]["scan_cache"] = {
        "enabled": True,
        "memory_size": 8,
        "disk_size": 64,
        "directory": str(tmp_path / "scan-cache"),
        "on_load": True,
    }
    proposal = _proposal()
    proposal["hyperparameters"]["backend_config"] = {"scan_hd_cache_size": 128}

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        payload = adapter._build_dataset_config(profile, proposal, _implementation(tmp_path))

    backend = payload["backend_config"]
    assert backend["cache_scan_on_load"] is True
    assert backend["cache_scan_on_write"] is False
    assert backend["scan_cache_size"] == 8
    assert backend["scan_hd_cache_size"] == 128
    assert backend["scan_cache_directory"] == str(tmp_path / "scan-cache")


def test_build_dataset_config_replaces_missing_drive_scan_cache_directory_with_f_temp(tmp_path):
    adapter = NeuralSignalPlugin()
    profile = _profile()
    profile["datasets"][0]["backend_config"] = {
        "scan_cache_size": 8,
        "scan_hd_cache_size": 64,
        "scan_cache_directory": "J:\\Temp\\scan_cache",
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter._path_root_exists", lambda path: path in {"F:/temp", "F:\\temp"}):
            payload = adapter._build_dataset_config(profile, _proposal(), _implementation(tmp_path))

    cache_dir = Path(payload["backend_config"]["scan_cache_directory"])
    assert cache_dir == Path("F:/temp")
    assert cache_dir.exists()
    assert payload["backend_config"]["scan_cache_size"] == 8
    assert payload["backend_config"]["scan_hd_cache_size"] == 64


def test_default_scan_cache_directory_uses_tmp_when_f_drive_missing():
    profile = _profile()
    with patch("core.plugins.neuralsignal.adapter._path_root_exists", lambda path: path == "/tmp"):
        cache_dir = ns_adapter._default_scan_cache_directory(profile)

    assert cache_dir == Path("/tmp")


def test_default_scan_cache_directory_falls_back_when_f_drive_missing(tmp_path):
    profile = _profile()
    with patch("core.plugins.neuralsignal.adapter._path_root_exists", return_value=False):
        with patch("core.plugins.neuralsignal.adapter.dev_path", lambda *parts: tmp_path.joinpath(*parts)):
            cache_dir = ns_adapter._default_scan_cache_directory(profile)

    assert cache_dir == tmp_path / "scan_cache" / "neuralsignal"


def test_prepare_experiment_runs_dataset_task_and_records_csv_metadata(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}

    mock_memory_service = MagicMock()
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter.MemoryService.for_profile", return_value=mock_memory_service):
                with patch.object(adapter, "_call_task", return_value={"file_paths": [str(csv_path)]}) as call_task:
                    delta = adapter.prepare_experiment(_profile(), state)

    call_task.assert_called_once()
    assert call_task.call_args.args[1] == "core.plugins.neuralsignal.tasks.create_dataset"
    assert call_task.call_args.kwargs["timeout"] == 7200
    assert call_task.call_args.kwargs["cwd"] == str(tmp_path / "neuralsignal_src")
    assert delta["errors"] == []
    assert len(delta["experiment_artifacts"]) == 1
    artifact = delta["experiment_artifacts"][0]
    assert artifact["artifact_type"] == "dataset"
    assert artifact["dataset_source"] == "generated"
    assert artifact["stored_artifact_id"] == "stored-dataset-1"
    assert artifact["stored_artifact_bucket"] == "researcher-artifacts"
    assert artifact["stored_artifact_key"] == "neuralsignal/dataset/stored-dataset-1/dataset.csv"
    assert artifact["status"] == "ready"
    assert artifact["rows"] == 2
    assert artifact["columns"] == 2
    assert artifact["column_names"] == ["a", "b"]
    assert delta["datasets"] == [artifact]
    mock_memory_service.persist_records.assert_called_once()


def test_prepare_experiment_normalizes_dataset_task_failure(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch.object(adapter, "_call_task", side_effect=RuntimeError("boom")):
                delta = adapter.prepare_experiment(_profile(), state)

    assert delta["experiment_artifacts"] == []
    assert any("activation_sparsity failed: boom" in error for error in delta["errors"])


def test_prepare_experiment_reuses_existing_dataset_when_overwrite_disabled(tmp_path):
    csv_path = tmp_path / "existing.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}
    dataset_cfg = {
        "dataset": "HaluBench",
        "detector_names": ["hallucination"],
        "dataset_output_dir": str(tmp_path),
        "file_out": "existing.csv",
        "overwrite_existing_dataset": False,
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch.object(adapter, "_build_dataset_config", return_value=dataset_cfg):
                with patch.object(adapter, "_call_task") as call_task:
                    delta = adapter.prepare_experiment(_profile(), state)

    call_task.assert_not_called()
    artifact = delta["experiment_artifacts"][0]
    assert artifact["dataset_source"] == "existing"
    assert artifact["stored_artifact_id"] == "stored-dataset-1"
    assert artifact["stored_artifact_bucket"] == "researcher-artifacts"
    assert artifact["status"] == "ready"
    assert artifact["dataset_path"] == str(csv_path)
    assert artifact["rows"] == 1
    assert artifact["task_result"]["skipped_existing_dataset"] is True
    assert delta["errors"] == []


def test_prepare_experiment_overwrites_existing_dataset_when_enabled(tmp_path):
    csv_path = tmp_path / "existing.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}
    dataset_cfg = {
        "dataset": "HaluBench",
        "detector_names": ["hallucination"],
        "dataset_output_dir": str(tmp_path),
        "file_out": "existing.csv",
        "overwrite_existing_dataset": True,
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch.object(adapter, "_build_dataset_config", return_value=dataset_cfg):
                with patch.object(adapter, "_call_task", return_value={"file_paths": [str(csv_path)]}) as call_task:
                    delta = adapter.prepare_experiment(_profile(), state)

    call_task.assert_called_once()
    assert delta["experiment_artifacts"][0]["dataset_source"] == "generated"
    assert delta["experiment_artifacts"][0]["dataset_path"] == str(csv_path)


def test_prepare_experiment_reuses_matching_dataset_from_memory(tmp_path):
    csv_path = tmp_path / "memory_dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}
    dataset_cfg = {
        "dataset": "HaluBench",
        "application_name": "HaluBench",
        "sub_application_name": "GranularAttention",
        "detector_names": ["hallucination"],
        "query": {"split": "train"},
        "row_limit": 25,
        "dataset_row_limit": 25,
        "balanced_target": {"enabled": True, "field": "ground_truth", "values": [0, 1]},
        "zone_size": 512,
        "feature_set_class_name": "ActivationSparsity",
        "feature_set_source_hash": "source-hash-1",
        "feature_set_configs": None,
        "ffn_layer_patterns": ["mlp", "fc"],
        "attn_layer_patterns": ["attn", ".q"],
        "backend_config": {"backend_type": "neuralsignal_v1"},
        "overwrite_existing_dataset": False,
    }
    mock_memory = MagicMock()
    mock_memory.find_one_record.return_value = {
        "record_id": "dataset:abc123",
        "object_type": "dataset",
        "content": {
            "dataset_artifact": {
                "dataset_path": str(csv_path),
                "stored_artifact_id": "stored-dataset-1",
                "stored_artifact_uri": "file:///stored/dataset.csv",
            }
        },
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter.MemoryService.for_profile", return_value=mock_memory):
                with patch.object(adapter, "_build_dataset_config", return_value=dataset_cfg):
                    with patch.object(adapter, "_call_task") as call_task:
                        delta = adapter.prepare_experiment(_profile(), state)

    call_task.assert_not_called()
    artifact = delta["experiment_artifacts"][0]
    assert artifact["dataset_source"] == "memory_reuse"
    assert artifact["dataset_path"] == str(csv_path)
    assert artifact["memory_record_id"] == "dataset:abc123"
    assert artifact["task_result"]["reused_from_memory"] is True


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
        "artifacts": {"feature_importance": {"a": 0.8}},
        "model_config": {"description": "desc", "model": "xgboost", "tags": ["neuralsignal"]},
        "figure_paths": {},
    }
    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: s
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "mlflow-run-123"

    mock_memory_service = MagicMock()
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter.MemoryService.for_profile", return_value=mock_memory_service):
                with patch("mlflow.set_tracking_uri"):
                    with patch("mlflow.set_experiment"):
                        with patch("mlflow.start_run", return_value=mock_run):
                            with patch("mlflow.log_params"):
                                with patch("mlflow.log_metrics"):
                                    with patch("mlflow.set_tags"):
                                        with patch("mlflow.log_dict"):
                                            with patch("mlflow.log_figure"):
                                                with patch.object(adapter, "_call_task", return_value=task_result) as call_task:
                                                    delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    call_task.assert_called_once()
    assert call_task.call_args.args[1] == "core.plugins.neuralsignal.tasks.create_s1_model"
    payload = call_task.call_args.args[2]
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
    assert delta["experiment_results"][0]["mlflow_run_id"] == "mlflow-run-123"
    assert delta["models"][0]["params"] == {"max_depth": 3}
    assert delta["models"][0]["mlflow_run_id"] == "mlflow-run-123"
    assert delta["models"][0]["experiment_id"] == delta["experiment_results"][0]["experiment_id"]
    mock_memory_service.persist_records.assert_called_once()


def test_execute_experiment_continues_when_mlflow_logging_fails(tmp_path):
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
    task_result = {
        "metrics": {"test_auc": 0.72},
        "params": {"max_depth": 3},
        "feature_importance": {"a": 0.8},
        "artifacts": {"feature_importance": {"a": 0.8}},
        "model_config": {"description": "desc", "model": "xgboost", "tags": ["neuralsignal"]},
        "figure_paths": {},
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("mlflow down")):
                with patch.object(adapter, "_call_task", return_value=task_result):
                    delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    assert delta["errors"] == []
    assert delta["experiment_results"][0]["metrics"]["test_auc"] == 0.72
    assert "mlflow_run_id" not in delta["experiment_results"][0]


def test_execute_experiment_logs_agent_state_and_figures_when_artifacts_exist(tmp_path):
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
    task_result = {
        "metrics": {"test_auc": 0.72, "train_auc": 0.81, "test_tp": 9, "test_fp": 1, "test_fn": 2, "test_tn": 8},
        "params": {"max_depth": 3},
        "feature_importance": {"a": 0.8},
        "artifacts": {
            "feature_importance": {"a": 0.8},
            "confusion_matrix": [[8, 1], [2, 9]],
            "roc_curve": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.8, 1.0]},
        },
        "model_config": {"description": "model description", "model": "xgboost", "tags": ["neuralsignal", "hallucination"]},
        "figure_paths": {},
    }
    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: s
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "mlflow-run-456"
    log_dict_calls: list[str] = []
    log_figure_calls: list[str] = []

    def _capture_log_dict(payload, path):
        log_dict_calls.append(path)

    def _capture_log_figure(fig, path):
        log_figure_calls.append(path)

    state = {
        "experiment_artifacts": [artifact],
        "research_direction": "find useful MLP sparsity probes",
        "research_summary": "summary text",
        "proposals": [{"name": "activation_sparsity", "description": "desc"}],
        "implementation_plans": [{"proposal_name": "activation_sparsity", "class_name": "ActivationSparsity"}],
        "implementations": [{"proposal_name": "activation_sparsity", "class_name": "ActivationSparsity", "script_path": "x.py", "validated": True}],
        "validation_results": [{"proposal_name": "activation_sparsity", "passed": True}],
        "research_artifacts": [{"artifact_id": "paper-1"}],
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("mlflow.log_dict", side_effect=_capture_log_dict):
                                        with patch("mlflow.log_figure", side_effect=_capture_log_figure):
                                            with patch.object(adapter, "_call_task", return_value=task_result):
                                                delta = adapter.execute_experiment(_profile(), state)

    assert delta["experiment_results"][0]["mlflow_run_id"] == "mlflow-run-456"
    assert "agent_state.json" in log_dict_calls
    assert "model_artifacts.json" in log_dict_calls
    assert "model_config.json" in log_dict_calls
    assert "auc_curve.png" in log_figure_calls
    assert "confusion_matrix.png" in log_figure_calls


def test_execute_experiment_logs_dataset_and_confusion_figure_artifacts(tmp_path):
    adapter = NeuralSignalPlugin()
    dataset_path = tmp_path / "features.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
    confusion_path = tmp_path / "confusion_matrix.png"
    confusion_path.write_text("fake image", encoding="utf-8")
    roc_path = tmp_path / "roc_curve.png"
    roc_path.write_text("fake image", encoding="utf-8")
    artifact = {
        "artifact_id": "activation_sparsity_dataset_0",
        "artifact_type": "dataset",
        "status": "ready",
        "proposal_name": "activation_sparsity",
        "dataset_path": str(dataset_path),
        "dataset": "HaluBench",
        "detector": "hallucination",
        "dataset_config": {"dataset": "HaluBench", "foo": "bar"},
    }
    task_result = {
        "metrics": {"test_auc": 0.72},
        "params": {"max_depth": 3},
        "feature_importance": {"a": 0.8},
        "artifacts": {"feature_importance": {"a": 0.8}},
        "model_config": {"description": "model description", "model": "xgboost", "tags": ["neuralsignal"]},
        "figure_paths": {"confusion_matrix": str(confusion_path), "roc_curve": str(roc_path)},
    }
    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: s
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "mlflow-run-789"
    artifact_calls: list[tuple[str, str | None]] = []
    text_calls: list[str] = []

    def _capture_log_artifact(path, artifact_path=None):
        artifact_calls.append((path, artifact_path))

    def _capture_log_text(text, path):
        text_calls.append(path)

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("mlflow.log_dict"):
                                        with patch("mlflow.log_figure"):
                                            with patch("mlflow.log_artifact", side_effect=_capture_log_artifact):
                                                with patch("mlflow.log_text", side_effect=_capture_log_text):
                                                    with patch.object(adapter, "_call_task", return_value=task_result):
                                                        delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    assert delta["experiment_results"][0]["mlflow_run_id"] == "mlflow-run-789"
    assert any(call[0] == str(confusion_path) and call[1] == "figures" for call in artifact_calls)
    assert any(call[0] == str(roc_path) and call[1] == "figures" for call in artifact_calls)
    assert any(call[0] == str(dataset_path) and call[1] == "dataset" for call in artifact_calls)
    assert "model_description.txt" in text_calls


def test_execute_experiment_does_not_register_figure_artifacts_in_state(tmp_path):
    adapter = NeuralSignalPlugin()
    dataset_path = tmp_path / "features.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
    confusion_path = tmp_path / "confusion_matrix.png"
    confusion_path.write_text("fake image", encoding="utf-8")
    artifact = {
        "artifact_id": "activation_sparsity_dataset_0",
        "artifact_type": "dataset",
        "status": "ready",
        "proposal_name": "activation_sparsity",
        "dataset_path": str(dataset_path),
        "dataset": "HaluBench",
        "detector": "hallucination",
        "dataset_config": {"dataset": "HaluBench"},
    }
    task_result = {
        "metrics": {"test_auc": 0.72},
        "params": {},
        "feature_importance": {},
        "artifacts": {},
        "model_config": {"model": "xgboost"},
        "figure_paths": {"confusion_matrix": str(confusion_path)},
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter._log_result_to_mlflow", return_value=""):
                with patch.object(adapter, "_call_task", return_value=task_result):
                    delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    assert "stored_figure_artifacts" not in delta["models"][0]
    assert "stored_figure_artifacts" not in delta["experiment_results"][0]


def test_build_memory_records_returns_neuralsignal_specific_records(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {
        "research_direction": "find useful MLP sparsity probes",
        "proposals": [{**_proposal(), "description": "Probe MLP activation sparsity"}],
        "implementations": [_implementation(tmp_path)],
        "experiment_artifacts": [{
            "artifact_id": "activation_sparsity_dataset_0",
            "artifact_type": "dataset",
            "proposal_name": "activation_sparsity",
            "dataset": "HaluBench",
            "detector": "hallucination",
            "dataset_path": str(tmp_path / "features.csv"),
            "stored_artifact_id": "stored-dataset-1",
            "stored_artifact_uri": "file:///stored/dataset.csv",
            "dataset_config": {
                "dataset": "HaluBench",
                "detector_names": ["hallucination"],
                "feature_set_class_name": "ActivationSparsity",
            },
        }],
        "experiment_results": [{
            "experiment_id": "exp-001",
            "proposal_name": "activation_sparsity",
            "metrics": {"test_auc": 0.72, "test_f1": 0.61},
            "feature_importance": {"layer_8_ffn_norm": 0.8},
            "artifacts": {"feature_importance": {"layer_8_ffn_norm": 0.8}},
            "model_config": {"model_name": "activation_sparsity_ab12cd34"},
            "figure_paths": {"roc_curve": "roc.png"},
            "mlflow_run_id": "mlflow-run-123",
        }],
        "models": [{
            "model_id": "activation_sparsity_ab12cd34",
            "proposal_name": "activation_sparsity",
            "mlflow_run_id": "mlflow-run-123",
        }],
        "evaluation_summary": {
            "llm_analysis": {
                "per_proposal": [{
                    "proposal_name": "activation_sparsity",
                    "assessment": "strong",
                    "interpretation": "Mid-layer MLP activations carry stable signal.",
                    "key_features": ["layer_8_ffn_norm"],
                    "hypothesis_supported": True,
                }]
            }
        },
    }

    records = adapter.build_memory_records(_profile(), state)

    kinds = {record["kind"] for record in records}
    assert "neuralsignal_dataset" in kinds
    assert "neuralsignal_featureset" in kinds
    assert "neuralsignal_experiment" in kinds

    experiment_record = next(record for record in records if record["kind"] == "neuralsignal_experiment")
    dataset_record = next(record for record in records if record["kind"] == "neuralsignal_dataset")
    featureset_record = next(record for record in records if record["kind"] == "neuralsignal_featureset")

    assert experiment_record["object_type"] == "experiment_result"
    assert experiment_record["metadata"]["dataset"] == "HaluBench"
    assert experiment_record["metadata"]["assessment"] == "strong"
    assert experiment_record["metadata"]["dataset_config_fingerprint"]
    assert experiment_record["metadata"]["mlflow_run_id"] == "mlflow-run-123"
    assert experiment_record["metadata"]["mlflow_experiment"] == "researcher_experiments"
    assert "Feature set class: ActivationSparsity" in experiment_record["summary"]
    assert any(entity["entity_type"] == "feature_set" for entity in experiment_record["entities"])
    assert any(rel["relation_type"] == "implemented_by" for rel in experiment_record["relations"])

    assert dataset_record["object_type"] == "dataset"
    assert dataset_record["metadata"]["dataset_config_fingerprint"]
    assert dataset_record["metadata"]["stored_artifact_uri"] == "file:///stored/dataset.csv"

    assert featureset_record["object_type"] == "featureset"
    assert featureset_record["metadata"]["feature_set_fingerprint"]
    assert featureset_record["metadata"]["stored_artifact_uri"] == "file:///stored/implementation.py"
    assert featureset_record["blob_refs"][0]["artifact_id"] == "stored-implementation-1"


def test_memory_record_to_artifact_returns_neuralsignal_specific_summary():
    adapter = NeuralSignalPlugin()
    record = {
        "record_id": "exp-001",
        "domain": "neuralsignal",
        "kind": "neuralsignal_experiment",
        "title": "activation_sparsity",
        "summary": "Direction: find useful MLP sparsity probes\nMetrics: {'test_auc': 0.72}",
        "metadata": {
            "dataset": "HaluBench",
            "detector": "hallucination",
            "feature_set_class_name": "ActivationSparsity",
            "mlflow_run_id": "mlflow-run-123",
            "test_auc": 0.72,
        },
    }

    artifact = adapter.memory_record_to_artifact(_profile(), record, {})

    assert artifact["source_type"] == "neuralsignal_experiment"
    assert "dataset=HaluBench" in artifact["title"]
    assert "detector=hallucination" in artifact["title"]
    assert "Feature set class: ActivationSparsity" in artifact["summary"]
    assert "test_auc: 0.72" in artifact["summary"]
    assert "mlflow-run-123" in artifact["summary"]


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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch.object(adapter, "_call_task", return_value=task_result) as call_task:
                adapter.execute_experiment(profile, {"experiment_artifacts": [artifact]})

    assert call_task.call_args.kwargs["timeout"] == 14400


def test_submit_experiment_jobs_submits_proposal_branch_job(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}
    submitted = {
        "job_id": "proposal_branch_activation_sparsity",
        "job_dir": str(tmp_path / "job"),
        "status": "submitted",
        "stage": "proposal_branch",
        "proposal_name": "activation_sparsity",
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.submit_task", return_value=submitted) as submit_task:
            delta = adapter.submit_experiment_jobs(
                {**_profile(), "execution": {"runner": "local_process", "max_parallel_jobs": 1}},
                state,
            )

    submit_task.assert_called_once()
    spec = submit_task.call_args.args[0]
    assert spec["stage"] == "proposal_branch"
    assert spec["task_path"] == "core.plugins.neuralsignal.tasks.run_proposal_branch"
    assert spec["payload"]["dataset_config"]["dataset"] == "HaluBench"
    assert spec["payload"]["model_config_base"]["dataset"] == "HaluBench"
    assert delta["experiment_jobs"][0]["status"] == "submitted"


def test_submit_experiment_jobs_reports_missing_implementation(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": []}
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.submit_task") as submit_task:
            delta = adapter.submit_experiment_jobs(
                {**_profile(), "execution": {"runner": "ray", "max_parallel_jobs": 1}},
                state,
            )

    submit_task.assert_not_called()
    assert delta["experiment_jobs"] == []
    assert any("has no generated implementation" in error for error in delta["errors"])


def test_submit_experiment_jobs_reports_generation_failure_before_submit(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {
        "proposals": [_proposal()],
        "implementations": [{
            "proposal_name": "activation_sparsity",
            "class_name": "ActivationSparsity",
            "script_path": "",
            "error": "LLM returned prose instead of Python",
        }],
    }
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.submit_task") as submit_task:
            delta = adapter.submit_experiment_jobs(
                {**_profile(), "execution": {"runner": "ray", "max_parallel_jobs": 1}},
                state,
            )

    submit_task.assert_not_called()
    assert delta["experiment_jobs"] == []
    assert any("implementation generation failed" in error for error in delta["errors"])


def test_build_model_config_requires_implementation_metadata(tmp_path):
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
            "feature_set_class_path": "",
            "feature_set_class_name": "",
        },
    }

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        try:
            adapter._build_model_config(_profile(), artifact, "exp-123")
        except RuntimeError as exc:
            assert "missing implementation metadata" in str(exc)
        else:
            raise AssertionError("Expected _build_model_config to reject empty feature set metadata")


def test_check_experiment_jobs_collects_completed_proposal_branch(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    feature_path = tmp_path / "Feature.py"
    feature_path.write_text("class Feature: pass\n", encoding="utf-8")
    job_dir = tmp_path / "proposal_branch_job"
    job_dir.mkdir()
    payload = {
        "proposal_name": "activation_sparsity",
        "experiment_id": "exp-123",
        "proposal": _proposal(),
        "implementation": {
            "proposal_name": "activation_sparsity",
            "class_name": "Feature",
            "script_path": str(feature_path),
            "validated": True,
        },
        "dataset_config": {
            "dataset": "HaluBench",
            "detector_names": ["hallucination"],
            "application_name": "HaluBench",
            "sub_application_name": "GranularAttention",
            "zone_size": 512,
            "feature_set_class_path": str(feature_path),
            "feature_set_class_name": "Feature",
            "file_out": "features.csv",
            "dataset_output_dir": str(tmp_path),
        },
        "model_config_base": {
            "dataset": "HaluBench",
            "file_out": "features.csv",
            "dataset_path": str(csv_path),
            "feature_set_class_name": "Feature",
        },
    }
    (job_dir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps({
            "job_id": "proposal_branch_job",
            "job_dir": str(job_dir),
            "task_path": "core.plugins.neuralsignal.tasks.run_proposal_branch",
            "payload": payload,
        }),
        encoding="utf-8",
    )
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps({
        "proposal_name": "activation_sparsity",
        "experiment_id": "exp-123",
        "dataset_result": {"file_paths": [str(csv_path)]},
        "model_result": {
            "metrics": {"test_auc": 0.72},
            "params": {"max_depth": 3},
            "feature_importance": {"a": 0.8},
            "artifacts": {"feature_importance": {"a": 0.8}},
            "model_config": {"model_name": "activation_sparsity_exp123", "model": "xgboost"},
            "figure_paths": {},
        },
    }), encoding="utf-8")

    checked_job = {
        "job_id": "proposal_branch_job",
        "job_dir": str(job_dir),
        "result_path": str(result_path),
        "status": "succeeded",
        "stage": "proposal_branch",
        "proposal_name": "activation_sparsity",
        "experiment_id": "exp-123",
    }

    adapter = NeuralSignalPlugin()
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter.check_task", return_value=checked_job):
                with patch("core.plugins.neuralsignal.adapter._log_result_to_mlflow", return_value="mlflow-run-branch"):
                    delta = adapter.check_experiment_jobs(
                        {**_profile(), "execution": {"runner": "local_process", "max_parallel_jobs": 1}},
                    {"experiment_jobs": [{"job_id": "proposal_branch_job", "job_dir": str(job_dir), "proposal_name": "activation_sparsity"}]},
                )

    assert delta["experiment_artifacts"][0]["status"] == "ready"
    assert delta["experiment_artifacts"][0]["rows"] == 1
    assert delta["experiment_results"][0]["experiment_id"] == "exp-123"
    assert delta["experiment_results"][0]["mlflow_run_id"] == "mlflow-run-branch"
    assert delta["models"][0]["mlflow_run_id"] == "mlflow-run-branch"


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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("core.plugins.external_tasks.subprocess.Popen", return_value=FakeProc()) as popen:
            result = NeuralSignalPlugin()._call_task({}, "some.module.task", {"x": 1}, cwd=str(tmp_path))

    assert result == {"ok": True}
    env = popen.call_args.kwargs["env"]
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == cfg.neuralsignal_src_path
    assert str((Path(os.getcwd()) / "core").resolve()) in pythonpath
    assert "existing_path" in pythonpath
    assert env["RESEARCH_PLUGIN_LOG"] == "neuralsignal"
    assert "core.plugins.neuralsignal" in env["RESEARCH_PLUGIN_LOGGERS"]
    assert env["RESEARCH_LOG_CONFIG"].endswith(os.path.join("configs", "config.yaml"))
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert popen.call_args.args[0][-3:] == ["-m", "core.plugins.task_runner", "some.module.task"]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert popen.call_args.kwargs["encoding"] == "utf-8"
    assert popen.call_args.kwargs["errors"] == "replace"


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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("core.plugins.external_tasks.subprocess.Popen", return_value=FakeProc()):
            result = NeuralSignalPlugin()._call_task({}, "some.module.task", {"x": 1}, timeout=123)

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
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("core.plugins.external_tasks.subprocess.Popen", return_value=FakeProc()) as popen:
            result = NeuralSignalPlugin()._call_task({}, "some.module.task", {"x": 1})

    assert result == {"ok": True}
    pythonpath = popen.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(package_dir.parent) in pythonpath
    assert str(package_dir) in pythonpath
    assert str((Path(os.getcwd()) / "core").resolve()) in pythonpath
    env = popen.call_args.kwargs["env"]
    assert env["RESEARCH_PLUGIN_LOG"] == "neuralsignal"
    assert "core.plugins.job_runner" in env["RESEARCH_PLUGIN_LOGGERS"]
    assert env["RESEARCH_LOG_CONFIG"].endswith(os.path.join("configs", "config.yaml"))
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert popen.call_args.args[0][-3:] == ["-m", "core.plugins.task_runner", "some.module.task"]
    assert popen.call_args.kwargs["encoding"] == "utf-8"
    assert popen.call_args.kwargs["errors"] == "replace"


def test_external_runtime_spec_exposes_shared_runner_settings(tmp_path):
    cfg = _cfg(tmp_path)
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        spec = NeuralSignalPlugin().external_runtime_spec({}, "validate")

    assert spec["python"] == "python"
    assert spec["plugin_name"] == "neuralsignal"
    assert str((Path(os.getcwd()) / "core").resolve()) in spec["pythonpath_entries"]
