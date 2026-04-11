"""Feature proposal node — translate research findings into neuralsignal experiments."""
from __future__ import annotations

from graph.state import ResearchState
from logger import get_logger

log = get_logger(__name__)


def feature_proposal_node(state: ResearchState) -> dict:
    """Propose neuralsignal experiments grounded in the research summary.

    Reads:
        state['research_direction']
        state['research_summary']
        state['arxiv_papers']

    Writes:
        state['feature_proposals'] — list of experiment proposals, each with:
            {name, description, neuralsignal_classes, feature_sets,
             zone_config, dataset, rationale}

    System prompt context includes the neuralsignal SDK API surface:
        - Feature sets: zones, logit-lens, T-F-diff, layer_distribution
        - Collector config: zone_size, zone_size_by_layer, layer_names_to_include
        - Detector types: normal, scan_delta
        - Training pipeline: DatasetRunner → DatasetCreator → S1Trainer
        - Available HuggingFace datasets for data collection

    TODO: build system prompt with neuralsignal API reference.
    TODO: pass research_summary + top-3 abstracts in user message.
    TODO: ask LLM to produce 3–5 proposals as a JSON list.
    TODO: parse and validate the JSON response.

    Note: on subsequent loop iterations, state['research_direction'] contains
    the top follow-up proposal from the previous iteration, not the original
    user input. The research_summary may still reflect the original research.
    """
    direction = state.get("research_direction", "")
    papers = state.get("arxiv_papers") or []
    log.info("feature_proposal_node | Generating proposals — direction=%r", direction)
    log.debug("feature_proposal_node | Context: %d papers, summary_len=%d",
              len(papers), len(state.get("research_summary") or ""))

    # TODO: build system prompt
    # log.debug("feature_proposal_node | System prompt length: %d chars", len(system_prompt))

    # TODO: call LLM
    # log.debug("feature_proposal_node | Sending request to LLM")

    # TODO: parse response
    # log.info("feature_proposal_node | Generated %d proposals: %s",
    #          len(proposals), [p["name"] for p in proposals])
    # for i, p in enumerate(proposals):
    #     log.debug("feature_proposal_node | Proposal %d — name=%r, classes=%s",
    #               i + 1, p["name"], p.get("neuralsignal_classes"))

    raise NotImplementedError("feature_proposal_node is not yet implemented")
