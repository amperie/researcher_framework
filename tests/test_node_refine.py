"""Tests for graph/nodes/refine.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.graph.nodes.refine import refine_node


PROFILE = {
    "name": "test",
    "datasets": [],
    "base_classes": [],
    "prompts": {"refine": {"system": "Refine ideas."}},
}


def test_refine_persists_memory_after_success():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='[{"name":"idea1","description":"desc","hypothesis":"h","rationale":"r"}]'
    )
    state = {
        "research_direction": "test direction",
        "ideas": [{"name": "idea1", "description": "desc"}],
    }

    with patch("core.graph.nodes.refine.get_llm", return_value=llm):
        with patch("core.graph.nodes.refine.persist_memory_records_for_state") as persist_memory:
            result = refine_node(state, PROFILE)

    assert result["refined_ideas"][0]["name"] == "idea1"
    persist_memory.assert_called_once()


def test_refine_memory_failure_is_non_fatal():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='[{"name":"idea1","description":"desc","hypothesis":"h","rationale":"r"}]'
    )
    state = {
        "research_direction": "test direction",
        "ideas": [{"name": "idea1", "description": "desc"}],
    }

    with patch("core.graph.nodes.refine.get_llm", return_value=llm):
        with patch(
            "core.graph.nodes.refine.persist_memory_records_for_state",
            side_effect=Exception("memory down"),
        ):
            result = refine_node(state, PROFILE)

    assert result["refined_ideas"][0]["name"] == "idea1"
    assert any("memory persistence failed" in error for error in result["errors"])
