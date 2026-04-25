"""Tests for async experiment job graph nodes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.graph.nodes.check_experiment_jobs import check_experiment_jobs_node
from core.graph.nodes.submit_experiment_jobs import submit_experiment_jobs_node


PROFILE = {"name": "test", "experiment_adapter": "plugins.test_adapter"}


def test_submit_experiment_jobs_delegates_to_adapter():
    adapter = MagicMock()
    adapter.submit_experiment_jobs.return_value = {"experiment_jobs": [{"job_id": "j1"}]}
    state = {"proposals": [{"name": "p1"}]}

    with patch("graph.nodes.submit_experiment_jobs.load_adapter", return_value=adapter):
        result = submit_experiment_jobs_node(state, PROFILE)

    adapter.submit_experiment_jobs.assert_called_once_with(PROFILE, state)
    assert result["experiment_jobs"] == [{"job_id": "j1"}]


def test_check_experiment_jobs_delegates_to_adapter():
    adapter = MagicMock()
    adapter.check_experiment_jobs.return_value = {
        "experiment_jobs": [{"job_id": "j1", "status": "succeeded"}],
        "experiment_results": [{"experiment_id": "e1"}],
    }
    state = {"experiment_jobs": [{"job_id": "j1"}]}

    with patch("graph.nodes.check_experiment_jobs.load_adapter", return_value=adapter):
        result = check_experiment_jobs_node(state, PROFILE)

    adapter.check_experiment_jobs.assert_called_once_with(PROFILE, state)
    assert result["experiment_jobs"][0]["status"] == "succeeded"
    assert result["experiment_results"] == [{"experiment_id": "e1"}]


def test_submit_missing_adapter_method_records_error():
    adapter = MagicMock(spec=[])

    with patch("graph.nodes.submit_experiment_jobs.load_adapter", return_value=adapter):
        result = submit_experiment_jobs_node({"experiment_jobs": []}, PROFILE)

    assert result["experiment_jobs"] == []
    assert any("does not implement" in error for error in result["errors"])


def test_check_missing_adapter_method_records_error():
    adapter = MagicMock(spec=[])

    with patch("graph.nodes.check_experiment_jobs.load_adapter", return_value=adapter):
        result = check_experiment_jobs_node({"experiment_jobs": []}, PROFILE)

    assert result["experiment_jobs"] == []
    assert any("does not implement" in error for error in result["errors"])
