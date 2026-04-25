"""Tests for graph/nodes/store_results.py."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.graph.nodes.store_results import store_results_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFILE = {
    "name": "test",
    "storage": {
        "mlflow_experiment": "test_exp",
        "mongodb_results_db": "test_db",
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


# ---------------------------------------------------------------------------
# No results to store
# ---------------------------------------------------------------------------

class TestStoreResultsNodeEmpty:
    def test_no_results_returns_empty_ids(self):
        result = store_results_node({}, PROFILE)
        assert result == {"stored_result_ids": []}

    def test_empty_results_list_returns_empty_ids(self):
        result = store_results_node({"experiment_results": []}, PROFILE)
        assert result == {"stored_result_ids": []}


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

class TestStoreResultsMLflow:
    def test_mlflow_run_logged(self):
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
                                    with patch("core.graph.nodes.store_results.ChromaStore"):
                                        with patch("pymongo.MongoClient"):
                                            result = store_results_node(
                                                {"experiment_results": [RESULT],
                                                 "research_direction": "test"},
                                                PROFILE,
                                            )

        assert "exp-001" in result["stored_result_ids"]

    def test_mlflow_failure_is_non_fatal(self):
        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("MLflow down")):
                with patch("core.graph.nodes.store_results.ChromaStore"):
                    with patch("pymongo.MongoClient"):
                        result = store_results_node(
                            {"experiment_results": [RESULT]},
                            PROFILE,
                        )

        # Still stored the ID and recorded the error
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
                                    with patch("core.graph.nodes.store_results.ChromaStore"):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node(
                                                {"experiment_results": [result_with_mixed]},
                                                PROFILE,
                                            )

        # bool should be excluded; strings excluded; int and float included
        assert "test_auc" in logged_metrics
        assert "count" in logged_metrics
        assert "name" not in logged_metrics
        assert "flag" not in logged_metrics

    def test_reuses_existing_mlflow_run_id(self):
        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri") as set_tracking_uri:
                with patch("mlflow.set_experiment") as set_experiment:
                    with patch("mlflow.start_run") as start_run:
                        with patch("core.graph.nodes.store_results.ChromaStore"):
                            with patch("pymongo.MongoClient"):
                                result = store_results_node(
                                    {"experiment_results": [{**RESULT, "mlflow_run_id": "run-existing"}]},
                                    PROFILE,
                                )

        assert "exp-001" in result["stored_result_ids"]
        set_tracking_uri.assert_not_called()
        set_experiment.assert_not_called()
        start_run.assert_not_called()


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------

class TestStoreResultsChroma:
    def test_chromadb_upsert_called(self):
        mock_store = MagicMock()
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
                                    with patch("core.graph.nodes.store_results.ChromaStore",
                                               return_value=mock_store):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node(
                                                {"experiment_results": [RESULT],
                                                 "research_direction": "test dir"},
                                                PROFILE,
                                            )

        mock_store.upsert.assert_called_once()
        call_args = mock_store.upsert.call_args
        assert call_args[0][0] == "exp-001"  # record_id

    def test_chromadb_failure_is_non_fatal(self):
        mock_store = MagicMock()
        mock_store.upsert.side_effect = Exception("ChromaDB down")

        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("mlflow off")):
                with patch("core.graph.nodes.store_results.ChromaStore", return_value=mock_store):
                    with patch("pymongo.MongoClient"):
                        result = store_results_node(
                            {"experiment_results": [RESULT]},
                            PROFILE,
                        )

        assert "exp-001" in result["stored_result_ids"]


# ---------------------------------------------------------------------------
# MongoDB storage
# ---------------------------------------------------------------------------

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
                                    with patch("core.graph.nodes.store_results.ChromaStore"):
                                        with patch("pymongo.MongoClient",
                                                   return_value=mock_client):
                                            store_results_node(
                                                {"experiment_results": [RESULT]},
                                                PROFILE,
                                            )

        mock_client["test_db"]["experiments"].insert_one.assert_called_once()
        mock_client.close.assert_called_once()

    def test_mongodb_failure_is_non_fatal(self):
        with patch("core.graph.nodes.store_results.get_config", return_value=MOCK_CFG):
            with patch("mlflow.set_tracking_uri", side_effect=Exception("off")):
                with patch("core.graph.nodes.store_results.ChromaStore"):
                    with patch("pymongo.MongoClient",
                               side_effect=Exception("MongoDB down")):
                        result = store_results_node(
                            {"experiment_results": [RESULT]},
                            PROFILE,
                        )

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
                                    with patch("core.graph.nodes.store_results.ChromaStore"):
                                        with patch("pymongo.MongoClient"):
                                            store_results_node(
                                                {
                                                    "experiment_results": [RESULT],
                                                    "models": [model],
                                                },
                                                PROFILE,
                                            )

        # Second call should have model_ prefix
        if len(logged_metrics) > 1:
            assert any(k.startswith("model_") for k in logged_metrics)

