"""Refine step — filter ideas for feasibility and improve them.

Reads:
    state['ideas']
    state['research_direction']

Writes:
    state['refined_ideas']
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ResearchState
from llm.factory import get_llm
from utils import extract_json_array
from utils.logger import get_logger
from utils.profile_loader import get_prompt, get_step_datasets

log = get_logger(__name__)


def refine_node(state: ResearchState, profile: dict) -> dict:
    ideas = state.get("ideas") or []
    direction = state.get("research_direction", "")

    if not ideas:
        log.warning("refine_node | No ideas to refine")
        return {"refined_ideas": []}

    system_prompt = get_prompt(profile, "refine")
    llm = get_llm("refine", profile)

    # Build dataset constraints block from profile — inject into prompt context
    datasets = get_step_datasets(profile)
    dataset_context = ""
    if datasets:
        scan_field_parts = []
        for ds in datasets:
            asf = ds.get("available_scan_fields") or {}
            scan_field_parts.append(
                f"Dataset: {ds['name']}\n"
                f"  Guaranteed fields: {asf.get('guaranteed', [])}\n"
                f"  Not available: {asf.get('not_available', [])}"
            )
        dataset_context = "\nDataset constraints:\n" + "\n".join(scan_field_parts)

    base_classes = profile.get("base_classes") or []
    base_class_context = ""
    if base_classes:
        base_class_context = "\nAvailable base classes:\n" + "\n".join(
            f"  {bc['name']}: {bc.get('description', '')}" for bc in base_classes
        )

    user_content = (
        f"Research direction: {direction}\n"
        f"{dataset_context}\n"
        f"{base_class_context}\n\n"
        f"Ideas to refine:\n{json.dumps(ideas, indent=2)}"
    )

    log.info("refine_node | Refining %d ideas", len(ideas))
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        refined = extract_json_array(resp.content)
        log.info("refine_node | %d ideas remain after refinement", len(refined))
    except Exception as exc:
        log.error("refine_node | Failed: %s", exc, exc_info=True)
        return {
            "refined_ideas": ideas,  # fall back to unrefined
            "errors": (state.get("errors") or []) + [f"refine failed: {exc}"],
        }

    return {"refined_ideas": refined}
