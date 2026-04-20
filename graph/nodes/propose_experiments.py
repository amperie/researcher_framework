"""Propose experiments step — turn refined ideas into fully-specified proposals.

Reads:
    state['refined_ideas']

Writes:
    state['proposals']  — experiments with hyperparameters and success criteria
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


def propose_experiments_node(state: ResearchState, profile: dict) -> dict:
    refined_ideas = state.get("refined_ideas") or []
    direction = state.get("research_direction", "")

    if not refined_ideas:
        log.warning("propose_experiments_node | No refined ideas in state")
        return {"proposals": []}

    system_prompt = get_prompt(profile, "propose_experiments")
    llm = get_llm("propose_experiments", profile)

    # Summarise available datasets for the LLM to choose from
    datasets = get_step_datasets(profile)
    datasets_summary = json.dumps(
        [
            {
                "name": d["name"],
                "description": d.get("description", ""),
                "available_detectors": d.get("available_detectors", []),
            }
            for d in datasets
        ],
        indent=2,
    )

    evaluation_cfg = profile.get("evaluation") or {}
    metrics_info = (
        f"Primary metric: {evaluation_cfg.get('primary_metric', 'n/a')}\n"
        f"Available metrics: {evaluation_cfg.get('metrics', [])}\n"
        f"Thresholds: {evaluation_cfg.get('thresholds', {})}"
    )

    user_content = (
        f"Research direction: {direction}\n\n"
        f"Available datasets:\n{datasets_summary}\n\n"
        f"Evaluation metrics:\n{metrics_info}\n\n"
        f"Refined ideas to turn into experiments:\n{json.dumps(refined_ideas, indent=2)}"
    )

    log.info("propose_experiments_node | Proposing experiments for %d ideas", len(refined_ideas))
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        proposals = extract_json_array(resp.content)
        log.info("propose_experiments_node | Generated %d proposals", len(proposals))
        for i, p in enumerate(proposals):
            log.debug("  Proposal %d: %s", i + 1, p.get("name"))
    except Exception as exc:
        log.error("propose_experiments_node | Failed: %s", exc, exc_info=True)
        return {
            "proposals": [],
            "errors": (state.get("errors") or []) + [f"propose_experiments failed: {exc}"],
        }

    return {"proposals": proposals}
