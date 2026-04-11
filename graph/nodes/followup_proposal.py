"""Follow-up proposal node — propose next experiments based on history."""
from __future__ import annotations

from graph.state import ResearchState
from logger import get_logger

log = get_logger(__name__)


def followup_proposal_node(state: ResearchState) -> dict:
    """Generate follow-up experiment proposals from experiment history.

    Reads:
        state['research_direction']
        state['raw_results']
        state['analysis_summary']
        state['generalization_results']

    Writes:
        state['followup_proposals']  — 3 proposals, each:
            {title, rationale, suggested_direction, priority}
        state['continue_loop']       — set to False (user controls looping via CLI)

    Steps:
        1. Retrieve last 10 experiment records: chroma.list_recent(10).
        2. Build LLM prompt with current experiment results + recent history.
        3. Ask LLM to identify: best-performing feature combos, failure modes,
           unexplored directions adjacent to the current research.
        4. Return 3 ranked follow-up proposals as structured JSON.

    Note on continue_loop:
        Always returns continue_loop=False. The CLI can set it to True to
        drive automatic multi-iteration sessions (--loop flag).

    TODO: implement steps 1–4.
    TODO: parse LLM JSON response into followup_proposals list.
    TODO: sort proposals by priority field before returning.
    """
    direction = state.get("research_direction", "")
    log.info("followup_proposal_node | Generating follow-up proposals — direction=%r", direction)

    # TODO: chroma.list_recent(10)
    # log.debug("followup_proposal_node | Fetched %d recent experiment records", len(history))

    # TODO: call LLM
    # log.debug("followup_proposal_node | Requesting proposals from LLM")

    # TODO: parse and sort
    # log.info("followup_proposal_node | Generated %d proposals: %s",
    #          len(proposals), [p["title"] for p in proposals])
    # for p in proposals:
    #     log.debug("followup_proposal_node |   priority=%s, direction=%r",
    #               p.get("priority"), p.get("suggested_direction"))

    # log.info("followup_proposal_node | continue_loop=False (user controls looping)")
    raise NotImplementedError("followup_proposal_node is not yet implemented")
