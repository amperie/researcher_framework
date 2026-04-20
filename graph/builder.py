"""LangGraph pipeline builder.

Reads the step list from a research profile and assembles a linear StateGraph.
Each step is wrapped so it receives both state and the loaded profile dict.
"""
from __future__ import annotations

import functools
from typing import Callable

from langgraph.graph import StateGraph, END

from graph.nodes import STEP_REGISTRY
from graph.state import ResearchState
from utils.logger import get_logger

log = get_logger(__name__)


def _wrap_node(fn: Callable, profile: dict) -> Callable:
    """Wrap a node function to inject the profile dict as a second argument."""
    @functools.wraps(fn)
    def wrapper(state: ResearchState) -> dict:
        return fn(state, profile)
    return wrapper


def build_graph(profile: dict):
    """Build and compile a LangGraph StateGraph from the profile's step list.

    Args:
        profile: Loaded research profile dict (from utils.profile_loader.load_profile).

    Returns:
        Compiled LangGraph that accepts a ResearchState dict and returns one.

    Raises:
        ValueError: If a step name in the profile is not in STEP_REGISTRY.
    """
    steps: list[str] = (profile.get("pipeline") or {}).get("steps") or []
    if not steps:
        raise ValueError(f"Profile {profile.get('name')!r} has no pipeline.steps defined")

    # Validate all step names up front
    unknown = [s for s in steps if s not in STEP_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown step(s) in profile {profile.get('name')!r}: {unknown}. "
            f"Available: {sorted(STEP_REGISTRY.keys())}"
        )

    log.info("builder | Assembling pipeline for profile=%r — steps=%s", profile.get("name"), steps)

    graph = StateGraph(ResearchState)

    # Add all nodes
    for step_name in steps:
        node_fn = STEP_REGISTRY[step_name]
        wrapped = _wrap_node(node_fn, profile)
        graph.add_node(step_name, wrapped)
        log.debug("builder | Added node: %s", step_name)

    # Wire linear edges
    graph.set_entry_point(steps[0])
    for i in range(len(steps) - 1):
        graph.add_edge(steps[i], steps[i + 1])
    graph.add_edge(steps[-1], END)

    compiled = graph.compile()
    log.info("builder | Pipeline compiled — %d steps", len(steps))
    return compiled
