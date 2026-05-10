"""Tests for memory retrieval helpers in core.tools.research_tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.tools.research_tools import collect_memory, collect_prior_experiments


PROFILE = {
    "name": "test",
    "storage": {
        "chroma_collection": "test_col",
    },
    "experiment_adapter": "plugins.test_adapter",
}


def _record() -> dict:
    return {
        "record": {
            "record_id": "exp-001",
            "domain": "test",
            "kind": "prior_experiment",
            "title": "ffn_dispersion",
            "summary": "Moderate signal. test_auc=0.69.",
            "metadata": {
                "proposal_name": "ffn_dispersion",
                "assessment": "moderate",
                "research_direction": "hallucination probing",
                "lessons": ["Avoid assuming outputs[0] shape."],
            },
        },
        "distance": 0.12,
    }


class TestCollectMemory:
    def test_collect_memory_returns_memory_artifacts(self):
        mock_service = MagicMock()
        mock_service.search.return_value = [_record()]

        with patch("core.tools.research_tools.MemoryService.for_profile", return_value=mock_service):
            with patch("core.tools.research_tools.load_adapter", return_value=object()):
                artifacts = collect_memory(
                    "hallucination probing",
                    PROFILE,
                    {"name": "memory", "n_results": 3},
                    {},
                )

        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact["artifact_id"] == "memory:exp-001"
        assert artifact["source"] == "memory"
        assert artifact["source_type"] == "prior_experiment"
        assert artifact["title"] == "ffn_dispersion [moderate]"
        assert "Moderate signal. test_auc=0.69." in artifact["summary"]
        assert "Avoid assuming outputs[0] shape." in artifact["summary"]
        assert artifact["metadata"]["distance"] == 0.12

    def test_collect_prior_experiments_aliases_to_memory(self):
        mock_service = MagicMock()
        mock_service.search.return_value = [_record()]

        with patch("core.tools.research_tools.MemoryService.for_profile", return_value=mock_service):
            with patch("core.tools.research_tools.load_adapter", return_value=object()):
                artifacts = collect_prior_experiments(
                    "hallucination probing",
                    PROFILE,
                    {"name": "prior_experiments", "n_results": 3},
                    {},
                )

        assert len(artifacts) == 1
        assert artifacts[0]["artifact_id"] == "memory:exp-001"
        assert artifacts[0]["source"] == "prior_experiments"

    def test_collect_memory_falls_back_when_adapter_returns_none(self):
        mock_service = MagicMock()
        mock_service.search.return_value = [_record()]
        mock_adapter = MagicMock()
        mock_adapter.memory_record_to_artifact.return_value = None

        with patch("core.tools.research_tools.MemoryService.for_profile", return_value=mock_service):
            with patch("core.tools.research_tools.load_adapter", return_value=mock_adapter):
                artifacts = collect_memory(
                    "hallucination probing",
                    PROFILE,
                    {"name": "memory", "n_results": 3},
                    {},
                )

        assert len(artifacts) == 1
        assert artifacts[0]["artifact_id"] == "memory:exp-001"
        assert artifacts[0]["source"] == "memory"
        assert artifacts[0]["metadata"]["distance"] == 0.12
