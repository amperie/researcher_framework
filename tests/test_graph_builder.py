"""Tests for graph/builder.py — build_graph."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graph.builder import build_graph, _wrap_node
from graph.nodes import STEP_REGISTRY


# ---------------------------------------------------------------------------
# _wrap_node
# ---------------------------------------------------------------------------

class TestWrapNode:
    def test_wrapped_fn_injects_profile(self):
        profile = {"name": "test"}
        calls = []

        def node_fn(state, prof):
            calls.append((state, prof))
            return {}

        wrapped = _wrap_node(node_fn, profile)
        state = {"research_direction": "test"}
        wrapped(state)

        assert len(calls) == 1
        assert calls[0][0] == state
        assert calls[0][1] is profile

    def test_wrapper_preserves_function_name(self):
        def my_node(state, profile):
            return {}

        wrapped = _wrap_node(my_node, {})
        assert wrapped.__name__ == "my_node"


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def _minimal_profile(self, steps: list[str]) -> dict:
        return {
            "name": "test",
            "pipeline": {"steps": steps},
        }

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="no pipeline.steps"):
            build_graph(self._minimal_profile([]))

    def test_none_steps_raises(self):
        profile = {"name": "test", "pipeline": {}}
        with pytest.raises(ValueError, match="no pipeline.steps"):
            build_graph(profile)

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError, match="Unknown step"):
            build_graph(self._minimal_profile(["research", "nonexistent_step"]))

    def test_single_step_compiles(self):
        compiled = build_graph(self._minimal_profile(["research"]))
        assert compiled is not None

    def test_multi_step_compiles(self):
        compiled = build_graph(self._minimal_profile(["research", "ideate", "refine"]))
        assert compiled is not None

    def test_all_registry_steps_compile(self):
        all_steps = list(STEP_REGISTRY.keys())
        compiled = build_graph(self._minimal_profile(all_steps))
        assert compiled is not None

    def test_registry_contains_expected_steps(self):
        expected = {
            "research", "ideate", "refine", "propose_experiments",
            "plan_implementation", "implement", "validate",
            "prepare_experiment", "execute_experiment",
            "evaluate", "store_results", "propose_next_steps",
        }
        assert expected.issubset(set(STEP_REGISTRY.keys()))
