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

from core.plugins.neuralsignal.adapter import NeuralSignalPlugin


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
                "mime_type": "image/png",
            }
        if artifact_type == "implementation":
            return {
                "artifact_id": "stored-implementation-1",
                "uri": "file:///stored/implementation.py",
            }
        return {
            "artifact_id": "stored-dataset-1",
            "uri": "file:///stored/dataset.csv",
        }

    store.store_file.side_effect = _store_file
    store.store_json.side_effect = lambda *args, **kwargs: {
        "artifact_id": "stored-model-1",
        "uri": "file:///stored/model.json",
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


def test_prepare_experiment_runs_dataset_task_and_records_csv_metadata(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot") as write_snapshot:
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
    assert artifact["dataset_source"] == "generated"
    assert artifact["stored_artifact_id"] == "stored-dataset-1"
    assert artifact["status"] == "ready"
    assert artifact["rows"] == 2
    assert artifact["columns"] == 2
    assert artifact["column_names"] == ["a", "b"]
    assert delta["datasets"] == [artifact]
    write_snapshot.assert_called_once()
    assert write_snapshot.call_args.args[0] == "prepare_experiment"


def test_prepare_experiment_normalizes_dataset_task_failure(tmp_path):
    adapter = NeuralSignalPlugin()
    state = {"proposals": [_proposal()], "implementations": [_implementation(tmp_path)]}

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot") as write_snapshot:
                with patch.object(adapter, "_call_task", side_effect=RuntimeError("boom")):
                    delta = adapter.prepare_experiment(_profile(), state)

    assert delta["experiment_artifacts"] == []
    assert any("activation_sparsity failed: boom" in error for error in delta["errors"])
    write_snapshot.assert_called_once()


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
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
                with patch.object(adapter, "_build_dataset_config", return_value=dataset_cfg):
                    with patch.object(adapter, "_call_task") as call_task:
                        delta = adapter.prepare_experiment(_profile(), state)

    call_task.assert_not_called()
    artifact = delta["experiment_artifacts"][0]
    assert artifact["dataset_source"] == "existing"
    assert artifact["stored_artifact_id"] == "stored-dataset-1"
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
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
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
                with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot") as write_snapshot:
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
    assert delta["experiment_results"][0]["mlflow_run_id"] == "mlflow-run-123"
    assert delta["experiment_results"][0]["stored_artifact_id"] == "stored-model-1"
    assert delta["models"][0]["params"] == {"max_depth": 3}
    assert delta["models"][0]["mlflow_run_id"] == "mlflow-run-123"
    assert delta["models"][0]["stored_artifact_id"] == "stored-model-1"
    assert delta["models"][0]["experiment_id"] == delta["experiment_results"][0]["experiment_id"]
    write_snapshot.assert_called_once()
    assert write_snapshot.call_args.args[0] == "execute_experiment"


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
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
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
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
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
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
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


def test_execute_experiment_registers_figure_artifacts(tmp_path):
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
            with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot"):
                with patch("core.plugins.neuralsignal.adapter._log_result_to_mlflow", return_value=""):
                    with patch.object(adapter, "_call_task", return_value=task_result):
                        delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    stored_figures = delta["models"][0]["stored_figure_artifacts"]
    assert stored_figures[0]["artifact_id"] == "stored-figure-confusion_matrix.png"
    assert stored_figures[0]["figure_name"] == "confusion_matrix"


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
            "stored_artifact_id": "stored-model-1",
            "stored_artifact_uri": "file:///stored/model.json",
            "mlflow_run_id": "mlflow-run-123",
        }],
        "models": [{
            "model_id": "activation_sparsity_ab12cd34",
            "proposal_name": "activation_sparsity",
            "stored_artifact_id": "stored-model-1",
            "stored_artifact_uri": "file:///stored/model.json",
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
    assert "neuralsignal_model" in kinds
    assert "neuralsignal_experiment" in kinds

    experiment_record = next(record for record in records if record["kind"] == "neuralsignal_experiment")
    dataset_record = next(record for record in records if record["kind"] == "neuralsignal_dataset")
    featureset_record = next(record for record in records if record["kind"] == "neuralsignal_featureset")
    model_record = next(record for record in records if record["kind"] == "neuralsignal_model")

    assert experiment_record["object_type"] == "experiment_result"
    assert experiment_record["metadata"]["dataset"] == "HaluBench"
    assert experiment_record["metadata"]["assessment"] == "strong"
    assert experiment_record["metadata"]["dataset_config_fingerprint"]
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

    assert model_record["object_type"] == "model"
    assert model_record["metadata"]["model_config_fingerprint"]


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
            "stored_artifact_uri": "file:///stored/model.json",
            "test_auc": 0.72,
        },
    }

    artifact = adapter.memory_record_to_artifact(_profile(), record, {})

    assert artifact["source_type"] == "neuralsignal_experiment"
    assert "dataset=HaluBench" in artifact["title"]
    assert "detector=hallucination" in artifact["title"]
    assert "Feature set class: ActivationSparsity" in artifact["summary"]
    assert "test_auc: 0.72" in artifact["summary"]
    assert "file:///stored/model.json" in artifact["summary"]


def test_execute_experiment_records_not_ready_dataset_error():
    adapter = NeuralSignalPlugin()
    artifact = {
        "artifact_type": "dataset",
        "status": "missing_file",
        "proposal_name": "activation_sparsity",
    }

    with patch("core.plugins.neuralsignal.adapter._write_incremental_state_snapshot") as write_snapshot:
        delta = adapter.execute_experiment(_profile(), {"experiment_artifacts": [artifact]})

    assert delta["experiment_results"] == []
    assert delta["models"] == []
    assert any("dataset artifact is not ready" in error for error in delta["errors"])
    write_snapshot.assert_called_once()


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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_runner", return_value=runner):
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
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=_cfg(tmp_path)):
        with patch("core.plugins.neuralsignal.adapter.get_artifact_store", return_value=_mock_artifact_store()):
            with patch("core.plugins.neuralsignal.adapter.get_runner", return_value=runner):
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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("core.plugins.neuralsignal.adapter.subprocess.Popen", return_value=FakeProc()) as popen:
            result = NeuralSignalPlugin()._call_task("some.module.task", {"x": 1}, cwd=str(tmp_path))

    assert result == {"ok": True}
    env = popen.call_args.kwargs["env"]
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == cfg.neuralsignal_src_path
    assert str((Path(os.getcwd()) / "core").resolve()) in pythonpath
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

    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("core.plugins.neuralsignal.adapter.subprocess.Popen", return_value=FakeProc()):
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
    with patch("core.plugins.neuralsignal.adapter.get_config", return_value=cfg):
        with patch("core.plugins.neuralsignal.adapter.subprocess.Popen", return_value=FakeProc()) as popen:
            result = NeuralSignalPlugin()._call_task("some.module.task", {"x": 1})

    assert result == {"ok": True}
    pythonpath = popen.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(package_dir.parent) in pythonpath
    assert str(package_dir) in pythonpath
    assert str((Path(os.getcwd()) / "core").resolve()) in pythonpath
    assert popen.call_args.args[0][-3:] == ["-m", "plugins.task_runner", "some.module.task"]
