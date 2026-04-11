"""Code generation node — produce a runnable experiment script."""
from __future__ import annotations

from graph.state import ResearchState
from logger import get_logger

log = get_logger(__name__)


def code_generation_node(state: ResearchState) -> dict:
    """Generate a runnable experiment script for the top feature proposal.

    Reads:
        state['feature_proposals']   — uses proposals[0] (highest priority)
        state['research_summary']

    Writes:
        state['generated_code']      — complete Python script (string)
        state['experiment_config']   — structured params dict (mirrors script inputs)

    Generated script contract:
        - Imports from the `neuralsignal` package (on PYTHONPATH via subprocess env).
        - Uses DatasetRunner, DatasetCreator, S1Trainer or SDK.evaluate_indirect
          as appropriate for the proposal.
        - On success: prints results as JSON on the LAST line of stdout.
          e.g. print(json.dumps({"auc": 0.87, "accuracy": 0.91}))
        - On failure: prints {"error": "<message>"} and exits with code 1.

    TODO: build system prompt with full neuralsignal class signatures.
    TODO: pass experiment_config as structured input context for the LLM.
    TODO: ask LLM to generate the script, enforcing the JSON-last-line contract.
    TODO: strip markdown code fences if the LLM wraps output in ```python.
    """
    proposals = state.get("feature_proposals") or []
    top = proposals[0] if proposals else {}
    log.info("code_generation_node | Generating script for proposal=%r", top.get("name"))
    log.debug("code_generation_node | Proposal detail: %s", top)

    # TODO: derive experiment_config from top proposal
    # log.debug("code_generation_node | Experiment config: %s", experiment_config)

    # TODO: call LLM
    # log.debug("code_generation_node | Requesting script from LLM")

    # TODO: strip fences, validate
    # log.info("code_generation_node | Script generated — %d lines", len(code.splitlines()))
    # log.debug("code_generation_node | First 200 chars: %s", code[:200])

    raise NotImplementedError("code_generation_node is not yet implemented")
