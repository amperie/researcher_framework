"""Tests for graph/nodes/store_results.py."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.graph.nodes.store_results import store_results_node


PROFILE = {
    "name": "test",
    "storage": {
        "mlflow_experiment": "test_exp",
        "mongodb_results_db": "test_db",
        "mongodb_results_collection": "test_experiments",
        "chroma_collection": "test_col",
    },
}

MOCK_CFG = SimpleNamespace(
    mlflow_uri="http://localhost:5000",
    mongo_url="mongodb://localhost:27017",
    chroma_host="localhost",
    chroma_port=8000,
    chroma_ssl=False,
    chroma_auth_token=None,
    chroma_collection="test_col",
)

RESULT = {
    "experiment_id": "exp-001",
    "proposal_name": "my_proposal",
    "proposal": {"description": "Test proposal"},
    "metrics": {"test_auc": 0.75, "test_f1": 0.68},
}


class TestStoreResultsNodeEmpty:
    def test_no_results_returns_empty_ids(self):
        result = store_results_node({}, PROFILE)
        assert result == {"stored_result_ids": []}

    def test_empty_results_list_returns_empty_ids(self):
        result = store_results_node({"experiment_results": []}, PROFILE)
        assert result == {"stored_result_ids": []}


class TestStoreResultsMLflow:
    def test_mlflow_run_logged(self):
        mock_service = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "mlflow-run-123"

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile", return_value=mock_service):
                                        with patch("pymongo.MongoClient"):
                                            result = store_results_node(
                                                {"experiment_results": [RESULT], "research_direction": "test"},
                                                PROFILE,
                                            )

        assert "exp-001" in result["stored_result_ids"]

    def test_mlflow_failure_is_non_fatal(self):
        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("MLflow down")):
                with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                    with patch("pymongo.MongoClient"):
                        result = store_results_node({"experiment_results": [RESULT]}, PROFILE)

        assert "exp-001" in result["stored_result_ids"]
        assert any("MLflow failed" in e for e in result["errors"])

    def test_only_numeric_metrics_logged_to_mlflow(self):
        result_with_mixed = {
            **RESULT,
            "metrics": {"test_auc": 0.75, "name": "string_val", "flag": True, "count": 10},
        }
        logged_metrics = {}

        def capture_metrics(metrics):
            logged_metrics.update(metrics)

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics", side_effect=capture_metrics):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node({"experiment_results": [result_with_mixed]}, PROFILE)

        assert "test_auc" in logged_metrics
        assert "count" in logged_metrics
        assert "name" not in logged_metrics
        assert "flag" not in logged_metrics

    def test_reuses_existing_mlflow_run_id(self):
        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri") as set_tracking_uri:
                with patch("mlflow.set_experiment") as set_experiment:
                    with patch("mlflow.start_run") as start_run:
                        with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                            with patch("pymongo.MongoClient"):
                                result = store_results_node(
                                    {"experiment_results": [{**RESULT, "mlflow_run_id": "run-existing"}]},
                                    PROFILE,
                                )

        assert "exp-001" in result["stored_result_ids"]
        set_tracking_uri.assert_not_called()
        set_experiment.assert_not_called()
        start_run.assert_not_called()

    def test_blank_experiment_id_falls_back_to_proposal_name_and_persists_mlflow_id(self):
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-fallback"
        result_row = {
            **RESULT,
            "experiment_id": "",
        }

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                                        with patch("pymongo.MongoClient"):
                                            result = store_results_node(
                                                {"experiment_results": [result_row], "research_direction": "test"},
                                                PROFILE,
                                            )

        assert "my_proposal" in result["stored_result_ids"]
        assert result["experiment_results"][0]["experiment_id"] == "my_proposal"
        assert result["experiment_results"][0]["mlflow_run_id"] == "run-fallback"


class TestStoreResultsMemory:
    def test_memory_service_persist_records_called(self):
        mock_service = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile", return_value=mock_service):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node(
                                                {"experiment_results": [RESULT], "research_direction": "test dir"},
                                                PROFILE,
                                            )

        mock_service.persist_records.assert_called_once()
        records = mock_service.persist_records.call_args.args[0]
        assert len(records) == 1
        assert records[0]["record_id"] == "exp-001"

    def test_memory_record_contains_structured_metadata(self):
        mock_service = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"
        state = {
            "experiment_results": [RESULT],
            "research_direction": "test dir",
            "evaluation_summary": {
                "llm_analysis": {
                    "per_proposal": [
                        {
                            "proposal_name": "my_proposal",
                            "assessment": "moderate",
                            "interpretation": "Signal mostly came from mid-layer activations.",
                            "key_features": ["layer_8_ffn_norm"],
                            "hypothesis_supported": True,
                        }
                    ]
                }
            },
        }

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile", return_value=mock_service):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node(state, PROFILE)

        record = mock_service.persist_records.call_args.args[0][0]
        assert "Assessment: moderate" in record["summary"]
        assert record["kind"] == "prior_experiment"
        assert record["metadata"]["research_direction"] == "test dir"
        assert record["metadata"]["assessment"] == "moderate"
        assert record["metadata"]["hypothesis_supported"] is True
        assert any("mid-layer activations" in lesson for lesson in record["metadata"]["lessons"])

    def test_memory_persistence_failure_is_non_fatal(self):
        mock_service = MagicMock()
        mock_service.persist_records.side_effect = Exception("memory down")

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("mlflow off")):
                with patch("core.graph.nodes.store_results.MemoryService.for_profile", return_value=mock_service):
                    with patch("pymongo.MongoClient"):
                        result = store_results_node({"experiment_results": [RESULT]}, PROFILE)

        assert "exp-001" in result["stored_result_ids"]
        assert any("memory persistence failed" in e for e in result["errors"])


class TestStoreResultsMongo:
    def test_mongodb_insert_called(self):
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                                        with patch("pymongo.MongoClient", return_value=mock_client):
                                            store_results_node({"experiment_results": [RESULT]}, PROFILE)

        mock_client["test_db"]["test_experiments"].insert_one.assert_called_once()
        mock_client.close.assert_called_once()

    def test_mongodb_insert_includes_execution_metadata(self):
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"
        result = {
            **RESULT,
            "artifacts": {"runtime_config_path": "dev/experiments/trading/runtime_configs/Algo.yaml"},
            "execution_config": {"mode": "backtest", "hpo": {"search_space": {"stop_pct": {"type": "uniform", "low": 1.0, "high": 8.0}}}},
            "variant_results": [{"variant_name": "base", "raw_output": {"best_config": {"stop_pct": 3.5}}}],
            "report": "summary",
        }

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                                        with patch("pymongo.MongoClient", return_value=mock_client):
                                            store_results_node({"experiment_results": [result]}, PROFILE)

        inserted = mock_client["test_db"]["test_experiments"].insert_one.call_args.args[0]
        assert inserted["proposal"]["description"] == "Test proposal"
        assert inserted["artifacts"]["runtime_config_path"].endswith("Algo.yaml")
        assert inserted["execution_config"]["hpo"]["search_space"]["stop_pct"]["high"] == 8.0
        assert inserted["variant_results"][0]["raw_output"]["best_config"]["stop_pct"] == 3.5
        assert inserted["report"] == "summary"

    def test_mongodb_insert_persists_inserted_id_on_result(self):
        mock_client = MagicMock()
        mock_client["test_db"]["test_experiments"].insert_one.return_value.inserted_id = "mongo-123"
        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics"):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                                        with patch("pymongo.MongoClient", return_value=mock_client):
                                            result = store_results_node(
                                                {"experiment_results": [RESULT]},
                                                PROFILE,
                                            )

        assert result["experiment_results"][0]["mongo_document_id"] == "mongo-123"

    def test_mongodb_failure_is_non_fatal(self):
        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("off")):
                with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                    with patch("pymongo.MongoClient", side_effect=Exception("MongoDB down")):
                        result = store_results_node({"experiment_results": [RESULT]}, PROFILE)

        assert "exp-001" in result["stored_result_ids"]
        assert any("MongoDB failed" in e for e in result["errors"])

    def test_model_metrics_logged_with_prefix(self):
        model = {
            "experiment_id": "exp-001",
            "metrics": {"model_auc": 0.80},
        }
        logged_metrics = {}

        def capture_metrics(metrics):
            logged_metrics.update(metrics)

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-1"

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch("mlflow.start_run", return_value=mock_run):
                        with patch("mlflow.log_params"):
                            with patch("mlflow.log_metrics", side_effect=capture_metrics):
                                with patch("mlflow.set_tags"):
                                    with patch("core.graph.nodes.store_results.MemoryService.for_profile"):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node(
                                                {"experiment_results": [RESULT], "models": [model]},
                                                PROFILE,
                                            )

        if len(logged_metrics) > 1:
            assert any(k.startswith("model_") for k in logged_metrics)
