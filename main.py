"""NeuralSignalResearcher — CLI entry point.

Usage:
    uv run python main.py --direction "activation patching in transformer FFN layers"
    uv run python main.py               # prompts interactively for direction
    uv run python main.py --loop        # automatically loop using top follow-up
    uv run python main.py --log-config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys

from utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent LangGraph orchestrator for NeuralSignal research."
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Research direction to investigate (e.g. 'layer-wise activation probing').",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Automatically loop: promote top follow-up proposal as the next direction.",
    )
    parser.add_argument(
        "--log-config",
        type=str,
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML config file for logging (default: config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Logging must be configured before any other module emits records ---
    setup_logging(args.log_config)
    log.debug("CLI args parsed: direction=%r, loop=%s, log_config=%r",
              args.direction, args.loop, args.log_config)

    # --- Resolve research direction ---
    direction = args.direction
    if not direction:
        direction = input("Research direction: ").strip()
    if not direction:
        log.error("No research direction provided — aborting.")
        print("Error: a research direction is required.", file=sys.stderr)
        sys.exit(1)

    log.info("Starting NeuralSignalResearcher — direction: %r", direction)

    # --- Build graph ---
    log.debug("Importing and compiling LangGraph pipeline")
    from graph.graph import build_graph

    graph = build_graph()
    log.info("Pipeline compiled — %d nodes", len(graph.nodes))

    # --- Initial state ---
    initial_state = {
        "research_direction": direction,
        "continue_loop": args.loop,
        "errors": [],
    }
    log.debug("Initial state: %s", initial_state)

    # --- Run pipeline ---
    print(f"\n[NeuralSignalResearcher] Starting research: {direction!r}\n")
    log.info("Invoking pipeline")

    try:
        # TODO: replace invoke with stream() to log per-node progress
        final_state = graph.invoke(initial_state)
    except Exception:
        log.critical("Pipeline raised an unhandled exception", exc_info=True)
        raise

    log.info("Pipeline finished — experiment_id=%r, success=%s",
             final_state.get("experiment_id"), final_state.get("execution_success"))

    errors = final_state.get("errors") or []
    if errors:
        log.warning("Pipeline completed with %d error(s): %s", len(errors), errors)

    # --- Print results ---
    _print_results(final_state)


def _print_results(state: dict) -> None:
    """Print a human-readable summary of the pipeline results to stdout."""
    log.debug("Rendering results summary")

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)

    if state.get("mlflow_run_id"):
        print(f"MLflow run:              {state['mlflow_run_id']}")
        log.info("MLflow primary run: %s", state["mlflow_run_id"])
    if state.get("mlflow_generalization_run_id"):
        print(f"MLflow gen. eval run:    {state['mlflow_generalization_run_id']}")
        log.info("MLflow generalization run: %s", state["mlflow_generalization_run_id"])
    if state.get("chroma_record_id"):
        print(f"ChromaDB record:         {state['chroma_record_id']}")
        log.info("ChromaDB record: %s", state["chroma_record_id"])

    if state.get("raw_results"):
        print(f"\nExperiment results:\n{json.dumps(state['raw_results'], indent=2)}")
        log.debug("Raw results: %s", state["raw_results"])

    if state.get("analysis_summary"):
        print(f"\nAnalysis:\n{state['analysis_summary']}")

    proposals = state.get("followup_proposals") or []
    if proposals:
        print(f"\nSuggested follow-up experiments ({len(proposals)}):")
        log.info("Follow-up proposals (%d):", len(proposals))
        for i, p in enumerate(proposals, 1):
            title = p.get("title", "(no title)")
            rationale = p.get("rationale", "")
            direction = p.get("suggested_direction", "")
            print(f"  {i}. {title}")
            if rationale:
                print(f"     Rationale: {rationale}")
            if direction:
                print(f"     Direction:  {direction}")
            log.info("  %d. %s — %s", i, title, direction)

    errors = state.get("errors") or []
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
            log.warning("Pipeline error: %s", e)

    print("=" * 72 + "\n")
    log.debug("Results summary complete")


if __name__ == "__main__":
    main()
