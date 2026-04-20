"""Result analysis node — compare experiment results against historical experiments."""
from __future__ import annotations

from graph.state import ResearchState
from utils.logger import get_logger

log = get_logger(__name__)


def result_analysis_node(state: ResearchState) -> dict:
    """Analyse current results in the context of similar past experiments.

    Reads:
        state['research_direction']
        state['raw_results']
        state['feature_proposals']   — proposals[0] for context

    Writes:
        state['similar_experiments'] — top-5 past experiments from ChromaDB
        state['analysis_summary']    — LLM comparative analysis narrative

    TODO: instantiate ChromaStore from config.
    TODO: call chroma.query_similar(state['research_direction'], n_results=5).
    TODO: build LLM prompt with current results + similar_experiments context.
    TODO: ask LLM to analyse: what worked, what didn't, metric deltas vs prior art.
    TODO: return analysis_summary as a markdown-formatted string.
    """
    direction = state.get("research_direction", "")
    raw = state.get("raw_results") or {}
    log.info("result_analysis_node | Analysing results — direction=%r, result_keys=%s",
             direction, list(raw.keys()))

    # TODO: instantiate ChromaStore
    # log.debug("result_analysis_node | Querying ChromaDB for similar experiments")

    # log.info("result_analysis_node | Found %d similar past experiments", len(similar))
    # for s in similar:
    #     log.debug("result_analysis_node |   past experiment: id=%r, distance=%s",
    #               s["id"], s.get("distance"))

    # TODO: call LLM for analysis
    # log.debug("result_analysis_node | Requesting comparative analysis from LLM")
    # log.info("result_analysis_node | Analysis complete — %d chars", len(summary))

    raise NotImplementedError("result_analysis_node is not yet implemented")
