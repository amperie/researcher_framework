"""Dev harness for manually invoking individual pipeline nodes.

CLI usage:
    uv run python run_node.py research --direction "sparse autoencoders in LLMs"
    uv run python run_node.py feature_proposal --state-file dev/state_after_research.json
    uv run python run_node.py research --direction "test" --no-save
    uv run python run_node.py <node> --out /tmp/my_state.json

IDE / debug usage:
    Run with no arguments — the script will prompt interactively for node,
    state file, and research direction.

State is loaded from --state-file (if given), then --direction is applied on top.
The returned delta is merged into state and saved to dev/state_after_<node>.json
(or --out if specified) unless --no-save is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.logger import setup_logging
from utils import fmt_value

setup_logging("configs/config.yaml")

from graph.nodes import (  # noqa: E402  (after logging setup)
    code_generation_node,
    db_logger_node,
    experiment_runner_node,
    feature_proposal_node,
    followup_proposal_node,
    generalization_eval_node,
    mlflow_logger_node,
    research_node,
    result_analysis_node,
)

NODES: dict[str, callable] = {
    "research":            research_node,
    "feature_proposal":    feature_proposal_node,
    "code_generation":     code_generation_node,
    "experiment_runner":   experiment_runner_node,
    "result_analysis":     result_analysis_node,
    "mlflow_logger":       mlflow_logger_node,
    "generalization_eval": generalization_eval_node,
    "db_logger":           db_logger_node,
    "followup_proposal":   followup_proposal_node,
}

_SEP = "-" * 60


def _prompt_node() -> str:
    """Interactively pick a node by name or number."""
    node_names = list(NODES)
    print("\nAvailable nodes:")
    for i, name in enumerate(node_names, 1):
        print(f"  {i}. {name}")
    while True:
        raw = input("\nNode to run (name or number): ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(node_names):
                return node_names[idx]
        elif raw in NODES:
            return raw
        print(f"  Invalid choice: {raw!r}. Enter a name or 1-{len(node_names)}.")


def _prompt_state_file() -> str | None:
    """Offer available dev/ fixtures and let the user pick one, or skip."""
    dev_dir = Path("dev")
    fixtures = sorted(dev_dir.glob("state_after_*.json")) if dev_dir.exists() else []
    if fixtures:
        print("\nAvailable state files:")
        for i, f in enumerate(fixtures, 1):
            print(f"  {i}. {f}")
        print("  0. Start with empty state")
        raw = input("State file (name, number, or Enter to skip): ").strip()
        if not raw or raw == "0":
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(fixtures):
                return str(fixtures[idx])
        return raw  # treat as a path
    else:
        raw = input("\nState file path (or Enter to skip): ").strip()
        return raw or None


def _prompt_direction(state: dict) -> str | None:
    """Ask for a research direction if one isn't already in state."""
    if state.get("research_direction"):
        print(f"\nresearch_direction already in state: {state['research_direction']!r}")
        override = input("Override? (new direction or Enter to keep): ").strip()
        return override or None
    raw = input("\nResearch direction (or Enter to skip): ").strip()
    return raw or None


def _interactive_prompt(args: argparse.Namespace) -> None:
    """Fill in any missing args by prompting the user on stdin."""
    print("\n[run_node] No arguments detected -- running in interactive mode.")

    if not args.node:
        args.node = _prompt_node()

    if not args.state_file:
        args.state_file = _prompt_state_file()

    # Direction prompt deferred until after state is loaded (see main)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single pipeline node for manual testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "node",
        nargs="?",           # optional so IDE run (no args) doesn't immediately error
        choices=list(NODES),
        metavar="node",
        help=f"Node to run. Choices: {', '.join(NODES)}",
    )
    parser.add_argument(
        "--state-file",
        metavar="PATH",
        help="JSON file to load as the initial state.",
    )
    parser.add_argument(
        "--direction",
        metavar="TEXT",
        help="Set state['research_direction'] (applied after --state-file).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Where to save the merged state JSON (default: dev/state_after_<node>.json).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print output but do not write a state file.",
    )
    args = parser.parse_args()

    # If no positional node given, assume IDE/debug run and prompt for everything
    interactive = args.node is None
    if interactive:
        _interactive_prompt(args)

    # ------------------------------------------------------------------
    # Load state
    # ------------------------------------------------------------------
    state: dict = {}
    state_source = "empty state"

    if args.state_file:
        path = Path(args.state_file)
        if not path.exists():
            print(f"[run_node] ERROR: state file not found: {path}", file=sys.stderr)
            sys.exit(1)
        state = json.loads(path.read_text(encoding="utf-8"))
        state_source = str(path)

    # In interactive mode, offer direction prompt now that state is loaded
    if interactive and not args.direction:
        direction = _prompt_direction(state)
        if direction:
            args.direction = direction

    if args.direction:
        state["research_direction"] = args.direction

    # ------------------------------------------------------------------
    # Run node
    # ------------------------------------------------------------------
    node_fn = NODES[args.node]
    print(f"\n[run_node] Running: {args.node}")
    print(f"[run_node] State loaded from: {state_source}")
    if state:
        present = [k for k in state if state[k] not in (None, [], {}, "")]
        print(f"[run_node] State keys present: {present}")
    print(_SEP)

    try:
        delta = node_fn(state)
    except NotImplementedError as exc:
        msg = str(exc) or f"{args.node} is not yet implemented"
        print(f"\n[run_node] NOT IMPLEMENTED: {msg}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[run_node] ERROR: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # Display delta
    # ------------------------------------------------------------------
    print("Node output (keys written to state):")
    for key, val in delta.items():
        print(f"  {key:<24} -> {fmt_value(val)}")
        if isinstance(val, str) and len(val) < 300:
            for line in val.splitlines()[:5]:
                print(f"    {line}")
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            for item in val[:2]:
                preview = {k: v for k, v in list(item.items())[:4]}
                print(f"    {preview}")
    print(_SEP)

    # ------------------------------------------------------------------
    # Save merged state
    # ------------------------------------------------------------------
    if args.no_save:
        print("[run_node] --no-save: state not written.\n")
        return

    merged = {**state, **delta}
    out_path = Path(args.out) if args.out else Path("dev") / f"state_after_{args.node}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"[run_node] Full state saved -> {out_path}\n")


if __name__ == "__main__":
    main()
