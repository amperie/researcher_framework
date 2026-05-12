"""LangGraph pipeline builder.

Reads the step list from a research profile and assembles a linear StateGraph.
Each step is wrapped so it receives both state and the loaded profile dict.
"""
from __future__ import annotations

import functools
import json
from typing import Callable

from langgraph.graph import StateGraph, END

from configs.config import dev_path
from core.graph.nodes import STEP_REGISTRY
from core.graph.state import ResearchState
from core.utils.logger import get_logger

log = get_logger(__name__)


def _canonical_step_name(node_name: str) -> str:
    raw = str(node_name or "").strip().lower()
    if not raw:
        return "unknown"
    return raw.replace(" ", "_").replace("-", "_")


def _wrap_node(fn: Callable, profile: dict, step_name: str) -> Callable:
    """Wrap a node function to inject the profile dict as a second argument."""
    @functools.wraps(fn)
    def wrapper(state: ResearchState) -> dict:
        delta = fn(state, profile)
        merged_state = {**state, **(delta or {})}
        _write_state_snapshot(str(profile.get("name") or state.get("profile_name") or "default"), step_name, merged_state)
        return delta
    return wrapper


def _write_state_snapshot(profile_name: str, node_name: str, state: dict) -> None:
    step_name = _canonical_step_name(node_name)
    serialisable: dict = {}
    for key, value in state.items():
        try:
            json.dumps(value, default=str)
            serialisable[key] = value
        except Exception:
            log.debug("builder | Skipping non-serialisable key %r for snapshot %s", key, step_name)
    out_path = dev_path("state", profile_name, f"after_{step_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")


def pipeline_steps(profile: dict) -> list[str]:
    """Return the validated linear step list for the given profile."""
    steps: list[str] = (profile.get("pipeline") or {}).get("steps") or []
    if not steps:
        raise ValueError(f"Profile {profile.get('name')!r} has no pipeline.steps defined")

    unknown = [s for s in steps if s not in STEP_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown step(s) in profile {profile.get('name')!r}: {unknown}. "
            f"Available: {sorted(STEP_REGISTRY.keys())}"
        )
    return steps


def build_graph(profile: dict, *, start_node: str | None = None):
    """Build and compile a LangGraph StateGraph from the profile's step list.

    Args:
        profile: Loaded research profile dict (from utils.profile_loader.load_profile).

    Returns:
        Compiled LangGraph that accepts a ResearchState dict and returns one.

    Raises:
        ValueError: If a step name in the profile is not in STEP_REGISTRY.
    """
    steps = pipeline_steps(profile)
    if start_node:
        if start_node not in steps:
            raise ValueError(
                f"Start node {start_node!r} is not in profile {profile.get('name')!r} pipeline: {steps}"
            )
        steps = steps[steps.index(start_node):]

    log.info("builder | Assembling pipeline for profile=%r — steps=%s", profile.get("name"), steps)

    graph = StateGraph(ResearchState)

    # Add all nodes
    for step_name in steps:
        node_fn = STEP_REGISTRY[step_name]
        wrapped = _wrap_node(node_fn, profile, step_name)
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
