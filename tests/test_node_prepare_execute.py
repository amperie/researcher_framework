"""Tests for graph/nodes/prepare_experiment.py and execute_experiment.py.

New API: adapter receives (profile, state) and returns a delta dict.
Legacy fallback: per-proposal create_dataset / run_experiment calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.graph.nodes.prepare_experiment import prepare_experiment_node, _normalize_delta as prep_normalize
from core.graph.nodes.execute_experiment import execute_experiment_node


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PROFILE = {
    "name": "test",
    "experiment_adapter": "plugins.test_adapter",
}

PROPOSAL = {"name": "prop1", "description": "Test proposal"}
IMPL = {"class_name": "TestClass", "proposal_name": "prop1", "script_path": "/tmp/test.py"}


def _spec_adapter(*method_names):
    """Return a MagicMock that only has the listed methods as callable."""
    adapter = MagicMock(spec=list(method_names))
    return adapter


# ---------------------------------------------------------------------------
# _normalize_delta (prepare)
# ---------------------------------------------------------------------------

class TestNormalizeDeltaPrepare:
    def test_none_delta_returns_empty(self):
        result = prep_normalize(None, {})
        assert result["experiment_artifacts"] == []
        assert result["errors"] == []

    def test_empty_dict_returns_empty(self):
        result = prep_normalize({}, {})
        assert result["experiment_artifacts"] == []

    def test_preserves_artifacts(self):
        delta = {"experiment_artifacts": [{"artifact_id": "a1"}]}
        result = prep_normalize(delta, {})
        assert len(result["experiment_artifacts"]) == 1

    def test_adds_datasets_alias_for_dataset_artifacts(self):
        delta = {"experiment_artifacts": [{"artifact_id": "a1", "dataset_id": "d1"}]}
        result = prep_normalize(delta, {})
        assert "datasets" in result
        assert len(result["datasets"]) == 1

    def test_preserves_existing_errors_from_state(self):
        result = prep_normalize({}, {"errors": ["prior error"]})
        assert "prior error" in result["errors"]


# ---------------------------------------------------------------------------
# prepare_experiment_node — adapter load failure
# ---------------------------------------------------------------------------

class TestPrepareExperimentAdapterLoad:
    def test_missing_adapter_key_returns_error(self):
        result = prepare_experiment_node(
            {"proposals": [PROPOSAL]},
            {"name": "test"},  # no experiment_adapter
        )
        assert result["experiment_artifacts"] == []
        assert any("adapter load failed" in e for e in result["errors"])

    def test_import_error_returns_error(self):
        with patch("graph.nodes.prepare_experiment.load_adapter",
                   side_effect=ImportError("no module")):
            result = prepare_experiment_node(
                {"proposals": [PROPOSAL]},
                PROFILE,
            )
        assert result["experiment_artifacts"] == []
        assert any("adapter load failed" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# prepare_experiment_node — new API (adapter.prepare_experiment(profile, state))
# ---------------------------------------------------------------------------

class TestPrepareExperimentNewApi:
    def test_delegates_to_adapter_prepare_experiment(self):
        delta = {"experiment_artifacts": [{"artifact_id": "a1", "proposal_name": "prop1"}]}
        adapter = MagicMock()
        adapter.prepare_experiment.return_value = delta

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node(
                {"proposals": [PROPOSAL]},
                PROFILE,
            )

        adapter.prepare_experiment.assert_called_once_with(PROFILE, {"proposals": [PROPOSAL]})
        assert len(result["experiment_artifacts"]) == 1

    def test_adapter_returns_none_gives_empty_artifacts(self):
        adapter = MagicMock()
        adapter.prepare_experiment.return_value = None

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_artifacts"] == []

    def test_adapter_exception_returns_error(self):
        adapter = MagicMock()
        adapter.prepare_experiment.side_effect = Exception("adapter crashed")

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_artifacts"] == []
        assert any("adapter failed" in e for e in result["errors"])

    def test_datasets_alias_propagated_from_delta(self):
        delta = {
            "experiment_artifacts": [{"artifact_id": "a1", "artifact_type": "dataset"}],
        }
        adapter = MagicMock()
        adapter.prepare_experiment.return_value = delta

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert "datasets" in result


# ---------------------------------------------------------------------------
# prepare_experiment_node — legacy fallback (create_dataset per proposal)
# ---------------------------------------------------------------------------

class TestPrepareExperimentLegacy:
    def test_no_proposals_returns_empty(self):
        adapter = _spec_adapter("create_dataset")
        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({}, PROFILE)
        assert result["experiment_artifacts"] == []

    def test_calls_create_dataset_per_proposal(self):
        artifact = {"dataset_id": "d1", "proposal_name": "prop1"}
        adapter = _spec_adapter("create_dataset")
        adapter.create_dataset.return_value = artifact

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node(
                {"proposals": [PROPOSAL], "implementations": [IMPL]},
                PROFILE,
            )

        adapter.create_dataset.assert_called_once()
        assert len(result["experiment_artifacts"]) == 1

    def test_none_artifact_recorded_as_error(self):
        adapter = _spec_adapter("create_dataset")
        adapter.create_dataset.return_value = None

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_artifacts"] == []
        assert any("no artifact returned" in e for e in result["errors"])

    def test_exception_recorded_as_error(self):
        adapter = _spec_adapter("create_dataset")
        adapter.create_dataset.side_effect = Exception("create_dataset crashed")

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_artifacts"] == []
        assert any("failed" in e for e in result["errors"])

    def test_multiple_proposals_processed(self):
        proposals = [
            {"name": "p1", "description": "d1"},
            {"name": "p2", "description": "d2"},
        ]
        adapter = _spec_adapter("create_dataset")
        adapter.create_dataset.side_effect = [
            {"artifact_id": "a1", "proposal_name": "p1"},
            {"artifact_id": "a2", "proposal_name": "p2"},
        ]

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": proposals}, PROFILE)

        assert len(result["experiment_artifacts"]) == 2

    def test_adapter_with_neither_method_creates_none_artifact(self):
        """Adapter with no prepare_experiment or create_dataset creates a 'none' artifact."""
        adapter = _spec_adapter()  # no methods

        with patch("graph.nodes.prepare_experiment.load_adapter", return_value=adapter):
            result = prepare_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert len(result["experiment_artifacts"]) == 1
        assert result["experiment_artifacts"][0]["artifact_type"] == "none"


# ---------------------------------------------------------------------------
# execute_experiment_node — adapter load failure
# ---------------------------------------------------------------------------

class TestExecuteExperimentAdapterLoad:
    def test_missing_adapter_key_returns_error(self):
        result = execute_experiment_node(
            {"proposals": [PROPOSAL]},
            {"name": "test"},
        )
        assert result["experiment_results"] == []
        assert any("adapter load failed" in e for e in result["errors"])

    def test_import_error_returns_error(self):
        with patch("graph.nodes.execute_experiment.load_adapter",
                   side_effect=ImportError("no module")):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)
        assert result["experiment_results"] == []


# ---------------------------------------------------------------------------
# execute_experiment_node — new API (adapter.execute_experiment(profile, state))
# ---------------------------------------------------------------------------

class TestExecuteExperimentNewApi:
    def test_delegates_to_adapter_execute_experiment(self):
        exp_result = {"metrics": {"test_auc": 0.75}, "proposal_name": "prop1",
                      "experiment_id": "e1"}
        delta = {"experiment_results": [exp_result]}
        adapter = MagicMock()
        adapter.execute_experiment.return_value = delta

        state = {"proposals": [PROPOSAL], "implementations": [IMPL]}

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node(state, PROFILE)

        adapter.execute_experiment.assert_called_once_with(PROFILE, state)
        assert len(result["experiment_results"]) == 1

    def test_adapter_returns_none_gives_empty_results(self):
        adapter = MagicMock()
        adapter.execute_experiment.return_value = None

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_results"] == []

    def test_adapter_exception_returns_error(self):
        adapter = MagicMock()
        adapter.execute_experiment.side_effect = Exception("execution failed")

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_results"] == []
        assert any("adapter failed" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# execute_experiment_node — legacy fallback (run_experiment per proposal)
# ---------------------------------------------------------------------------

class TestExecuteExperimentLegacy:
    def test_no_proposals_returns_empty(self):
        adapter = _spec_adapter("run_experiment")
        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({}, PROFILE)
        assert result["experiment_results"] == []

    def test_calls_run_experiment_per_proposal(self):
        run_result = {"metrics": {"test_auc": 0.72}}
        adapter = _spec_adapter("run_experiment")
        adapter.run_experiment.return_value = run_result

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node(
                {"proposals": [PROPOSAL], "implementations": [IMPL]},
                PROFILE,
            )

        adapter.run_experiment.assert_called_once()
        assert len(result["experiment_results"]) == 1

    def test_train_model_called_when_available(self):
        run_result = {"metrics": {"test_auc": 0.72}}
        model_result = {"metrics": {"model_auc": 0.80}}
        adapter = _spec_adapter("run_experiment", "train_model")
        adapter.run_experiment.return_value = run_result
        adapter.train_model.return_value = model_result

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        adapter.train_model.assert_called_once()
        assert "models" in result
        assert len(result["models"]) == 1

    def test_none_result_recorded_as_error(self):
        adapter = _spec_adapter("run_experiment")
        adapter.run_experiment.return_value = None

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_results"] == []
        assert any("no result returned" in e for e in result["errors"])

    def test_exception_recorded_as_error(self):
        adapter = _spec_adapter("run_experiment")
        adapter.run_experiment.side_effect = Exception("run crashed")

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_results"] == []
        assert any("failed" in e for e in result["errors"])

    def test_adapter_with_no_run_method_records_error(self):
        adapter = _spec_adapter()  # no run_experiment

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert any("cannot execute" in e for e in result["errors"])

    def test_experiment_id_set_on_result(self):
        run_result = {"metrics": {"test_auc": 0.75}}
        adapter = _spec_adapter("run_experiment")
        adapter.run_experiment.return_value = run_result

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node({"proposals": [PROPOSAL]}, PROFILE)

        assert result["experiment_results"][0]["experiment_id"] != ""
        assert result["experiment_results"][0]["proposal_name"] == "prop1"

    def test_existing_errors_preserved(self):
        adapter = _spec_adapter("run_experiment")
        adapter.run_experiment.side_effect = Exception("crash")

        with patch("graph.nodes.execute_experiment.load_adapter", return_value=adapter):
            result = execute_experiment_node(
                {"proposals": [PROPOSAL], "errors": ["prior error"]},
                PROFILE,
            )

        assert "prior error" in result["errors"]
