"""Tests for graph/nodes/ideate.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graph.nodes.ideate import ideate_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFILE = {
    "name": "test",
    "prompts": {"ideate": {"system": "Brainstorm ideas."}},
}

IDEAS_JSON = '[{"name": "idea1", "description": "desc1"}, {"name": "idea2", "description": "desc2"}]'


def _make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestIdeateNodeHappy:
    def test_returns_ideas_list(self):
        llm = _make_llm(IDEAS_JSON)
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(
                {"research_direction": "test", "research_summary": "Summary."},
                PROFILE,
            )
        assert "ideas" in result
        assert len(result["ideas"]) == 2

    def test_ideas_have_expected_keys(self):
        llm = _make_llm(IDEAS_JSON)
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(
                {"research_direction": "test", "research_summary": "Summary."},
                PROFILE,
            )
        assert result["ideas"][0]["name"] == "idea1"

    def test_uses_paper_digests_when_available(self):
        llm = _make_llm(IDEAS_JSON)
        state = {
            "research_direction": "test",
            "research_summary": "Summary.",
            "paper_digests": [{"title": "Paper A", "digest": "Detailed digest content."}],
        }
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(state, PROFILE)

        # Verify digest content was included in the LLM call
        human_message = llm.invoke.call_args[0][0][1]
        assert "Paper A" in human_message.content

    def test_falls_back_to_paper_abstracts_when_no_digests(self):
        llm = _make_llm(IDEAS_JSON)
        state = {
            "research_direction": "test",
            "research_summary": "Summary.",
            "paper_digests": [],
            "research_papers": [
                {"title": "Paper B", "abstract": "Abstract text.", "relevance_score": 8}
            ],
        }
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(state, PROFILE)

        human_message = llm.invoke.call_args[0][0][1]
        assert "Paper B" in human_message.content

    def test_includes_artifacts_in_context(self):
        llm = _make_llm(IDEAS_JSON)
        state = {
            "research_direction": "test",
            "research_summary": "Summary.",
            "research_artifacts": [
                {"source_type": "paper", "source": "arxiv", "title": "Artifact Title",
                 "summary": "Art summary.", "relevance_score": 7, "usefulness": "high", "risks": "low"}
            ],
        }
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(state, PROFILE)

        human_message = llm.invoke.call_args[0][0][1]
        assert "Artifact Title" in human_message.content

    def test_empty_state_still_calls_llm(self):
        llm = _make_llm(IDEAS_JSON)
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node({}, PROFILE)

        llm.invoke.assert_called_once()
        assert "ideas" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestIdeateNodeErrors:
    def test_llm_failure_returns_empty_ideas(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM timeout")
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(
                {"research_direction": "test"},
                PROFILE,
            )
        assert result["ideas"] == []
        assert any("ideate failed" in e for e in result["errors"])

    def test_bad_json_response_returns_empty_ideas(self):
        llm = _make_llm("This is not JSON at all.")
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(
                {"research_direction": "test"},
                PROFILE,
            )
        assert result["ideas"] == []
        assert "errors" in result

    def test_existing_errors_preserved(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("fail")
        with patch("graph.nodes.ideate.get_llm", return_value=llm):
            result = ideate_node(
                {"research_direction": "test", "errors": ["previous error"]},
                PROFILE,
            )
        assert "previous error" in result["errors"]
