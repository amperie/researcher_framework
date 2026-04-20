"""Propose next steps — suggest follow-up research directions based on results.

Reads:
    state['evaluation_summary']
    state['research_direction']
    state['experiment_results']

Writes:
    state['next_steps']
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ResearchState
from llm.factory import get_llm
from utils import extract_json_array
from utils.logger import get_logger
from utils.profile_loader import get_prompt

log = get_logger(__name__)


def propose_next_steps_node(state: ResearchState, profile: dict) -> dict:
    evaluation_summary = state.get("evaluation_summary") or {}
    direction = state.get("research_direction", "")
    experiment_results = state.get("experiment_results") or []

    system_prompt = get_prompt(profile, "propose_next_steps")
    llm = get_llm("propose_next_steps", profile)

    # Build compact results summary for the LLM
    results_summary = [
        {
            "proposal_name": r.get("proposal_name"),
            "metrics": r.get("metrics", {}),
        }
        for r in experiment_results
    ]

    user_content = (
        f"Research direction: {direction}\n\n"
        f"Evaluation summary:\n{json.dumps(evaluation_summary, indent=2, default=str)}\n\n"
        f"Experiment results overview:\n{json.dumps(results_summary, indent=2, default=str)}"
    )

    log.info("propose_next_steps_node | Generating next steps from %d results", len(experiment_results))
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        next_steps = extract_json_array(resp.content)
        log.info("propose_next_steps_node | Generated %d next steps", len(next_steps))
        for i, s in enumerate(next_steps):
            log.debug("  Step %d: %s", i + 1, s.get("title"))
    except Exception as exc:
        log.error("propose_next_steps_node | Failed: %s", exc, exc_info=True)
        return {
            "next_steps": [],
            "errors": (state.get("errors") or []) + [f"propose_next_steps failed: {exc}"],
        }

    return {"next_steps": next_steps}
