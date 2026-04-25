"""Tests for graph/nodes/research.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.graph.nodes.research import research_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TOOLS = [
    {
        "name": "arxiv",
        "tool": "tools.research_tools.collect_arxiv",
        "max_results": 5,
        "relevance_score_threshold": 6,
        "max_papers_to_digest": 0,
    }
]


def _make_profile(tools=None, prompts_override=None):
    """tools=None uses default arxiv tool; tools=[] uses no tools."""
    resolved_tools = _DEFAULT_TOOLS if tools is None else tools
    prompts = {
        "research": {
            "system": "You are a researcher.",
            "artifact_score_system": "Score this.",
            "score_system": "Score this.",
            "artifact_summary_system": "Summarise.",
            "summary_system": "Summarise.",
            "digest_system": "Digest.",
        }
    }
    if prompts_override:
        prompts["research"].update(prompts_override)
    return {
        "name": "test",
        "research": {
            "tools": resolved_tools,
            "domain_context": "test context",
        },
        "prompts": prompts,
    }


def _make_artifact(artifact_id="a1", title="Paper 1", summary="Abstract.", score=7):
    return {
        "artifact_id": artifact_id,
        "source": "arxiv",
        "source_type": "paper",
        "title": title,
        "summary": summary,
        "url": "http://arxiv.org/abs/2301.00001",
        "published": "2023-01-01",
        "metadata": {"arxiv_id": "2301.00001"},
        "raw": {},
        "score_threshold": 6,
        "relevance_score": score,
        "relevance_reason": "Relevant.",
    }


def _mock_llm_for_scoring(score=7):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=f'{{"score": {score}, "reason": "Good paper"}}'
    )
    return llm


# ---------------------------------------------------------------------------
# No research direction
# ---------------------------------------------------------------------------

class TestResearchNodeNoDirection:
    def test_returns_error_when_no_direction(self):
        result = research_node({}, _make_profile())
        assert "errors" in result
        assert any("no research_direction" in e for e in result["errors"])

    def test_empty_string_direction_returns_error(self):
        result = research_node({"research_direction": ""}, _make_profile())
        assert "errors" in result


# ---------------------------------------------------------------------------
# No tools configured
# ---------------------------------------------------------------------------

class TestResearchNodeNoTools:
    def test_empty_tools_returns_empty_artifacts(self):
        profile = _make_profile(tools=[])
        result = research_node({"research_direction": "test"}, profile)
        assert result["research_artifacts"] == []
        assert result["research_papers"] == []
        assert result["research_summary"] == ""
        assert result["paper_digests"] == []


# ---------------------------------------------------------------------------
# Tool collection and artifact scoring
# ---------------------------------------------------------------------------

class TestResearchNodeCollection:
    def test_collects_and_scores_artifacts(self):
        artifact = _make_artifact()
        mock_collector = MagicMock(return_value=[
            {
                "title": artifact["title"],
                "summary": artifact["summary"],
                "url": artifact["url"],
                "source_type": "paper",
                "metadata": artifact["metadata"],
            }
        ])
        llm = _mock_llm_for_scoring(score=8)

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                result = research_node(
                    {"research_direction": "attention mechanisms"},
                    _make_profile(),
                )

        assert len(result["research_artifacts"]) == 1
        assert result["research_artifacts"][0]["relevance_score"] == 8

    def test_low_score_artifact_filtered_out(self):
        mock_collector = MagicMock(return_value=[
            {"title": "Low", "summary": "s", "url": "u", "source_type": "paper", "metadata": {}}
        ])
        llm = _mock_llm_for_scoring(score=3)  # below threshold of 6

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                result = research_node(
                    {"research_direction": "test"},
                    _make_profile(),
                )

        assert result["research_artifacts"] == []

    def test_scoring_failure_sets_score_zero(self):
        mock_collector = MagicMock(return_value=[
            {"title": "T", "summary": "s", "url": "u", "source_type": "paper", "metadata": {}}
        ])
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM failure")

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                result = research_node(
                    {"research_direction": "test"},
                    _make_profile(),
                )

        # Should not crash; low-score artifacts filtered out
        assert isinstance(result, dict)
        assert result["research_artifacts"] == []

    def test_tool_failure_recorded_as_error(self):
        mock_collector = MagicMock(side_effect=Exception("tool crashed"))
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"score": 8}')

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                result = research_node(
                    {"research_direction": "test"},
                    _make_profile(),
                )

        assert any("failed" in e for e in result.get("errors", []))

    def test_deduplication_by_artifact_id(self):
        artifact = {"title": "Dup", "summary": "s", "url": "u",
                    "artifact_id": "same_id", "source_type": "paper", "metadata": {}}
        mock_collector = MagicMock(return_value=[artifact, artifact])
        llm = _mock_llm_for_scoring(score=7)

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                result = research_node(
                    {"research_direction": "test"},
                    _make_profile(),
                )

        assert len(result["research_artifacts"]) == 1

    def test_summary_generated_from_selected_artifacts(self):
        mock_collector = MagicMock(return_value=[
            {"title": "T", "summary": "s", "url": "u", "source_type": "paper", "metadata": {}}
        ])
        llm = MagicMock()
        # First call = scoring, second call = summary
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 8}'),
            MagicMock(content="This is the research summary."),
        ]

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                result = research_node(
                    {"research_direction": "test"},
                    _make_profile(),
                )

        assert result["research_summary"] == "This is the research summary."


# ---------------------------------------------------------------------------
# Paper digests
# ---------------------------------------------------------------------------

class TestResearchNodeDigests:
    def test_uses_cached_digest(self):
        cached = {"arxiv_id": "2301.00001", "title": "T", "digest": "cached content"}
        mock_collector = MagicMock(return_value=[
            {"title": "T", "summary": "s", "url": "u", "source_type": "paper",
             "metadata": {"arxiv_id": "2301.00001"}}
        ])
        llm = _mock_llm_for_scoring(score=8)
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 8}'),
            MagicMock(content="Summary."),
        ]

        profile = _make_profile(tools=[{
            "name": "arxiv",
            "tool": "tools.research_tools.collect_arxiv",
            "max_results": 5,
            "relevance_score_threshold": 6,
            "max_papers_to_digest": 2,
        }])

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                with patch("graph.nodes.research.load_cached_digest", return_value=cached):
                    result = research_node(
                        {"research_direction": "test"},
                        profile,
                    )

        assert len(result["paper_digests"]) == 1
        assert result["paper_digests"][0]["digest"] == "cached content"

    def test_skips_digest_when_no_full_text(self):
        mock_collector = MagicMock(return_value=[
            {"title": "T", "summary": "s", "url": "u", "source_type": "paper",
             "metadata": {"arxiv_id": "2301.00001"}}
        ])
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 8}'),
            MagicMock(content="Summary."),
        ]

        profile = _make_profile(tools=[{
            "name": "arxiv",
            "tool": "tools.research_tools.collect_arxiv",
            "max_results": 5,
            "relevance_score_threshold": 6,
            "max_papers_to_digest": 2,
        }])

        with patch("graph.nodes.research.load_research_tool", return_value=mock_collector):
            with patch("graph.nodes.research.get_llm", return_value=llm):
                with patch("graph.nodes.research.load_cached_digest", return_value=None):
                    with patch("graph.nodes.research.download_paper_text", return_value=None):
                        result = research_node(
                            {"research_direction": "test"},
                            profile,
                        )

        assert result["paper_digests"] == []
