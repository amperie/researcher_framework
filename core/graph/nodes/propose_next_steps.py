"""Propose next steps - suggest follow-up research directions based on results.

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

from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils import extract_json_array
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)


def propose_next_steps_node(state: ResearchState, profile: dict) -> dict:
    evaluation_summary = state.get("evaluation_summary") or {}
    direction = state.get("research_direction", "")
    experiment_results = state.get("experiment_results") or []

    system_prompt = get_prompt(profile, "propose_next_steps")
    llm = get_llm("propose_next_steps", profile)

    results_summary = [
        {
            "proposal_name": result.get("proposal_name"),
            "metrics": result.get("metrics", {}),
        }
        for result in experiment_results
    ]

    user_content = (
        f"Research direction: {direction}\n\n"
        f"Evaluation summary:\n{_truncate_text(json.dumps(evaluation_summary, indent=2, default=str), 2500)}\n\n"
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
        for i, step in enumerate(next_steps):
            log.debug("  Step %d: %s", i + 1, step.get("title"))
    except Exception as exc:
        log.error("propose_next_steps_node | Failed: %s", exc, exc_info=True)
        return {
            "next_steps": [],
            "errors": (state.get("errors") or []) + [f"propose_next_steps failed: {exc}"],
        }

    return {"next_steps": next_steps}


def _truncate_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
