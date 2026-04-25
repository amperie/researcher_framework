"""Research Pipeline — CLI entry point.

Usage:
    uv run python main.py --profile neuralsignal --direction "attention head specialization"
    uv run python main.py --profile neuralsignal          # prompts for direction
    uv run python main.py --profile neuralsignal --loop   # loop using top next_step
    uv run python main.py --list-profiles
"""
from __future__ import annotations

import argparse
import sys

from core.utils import setup_logging, get_logger

setup_logging()
log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configuration-driven agentic research pipeline."
    )
    parser.add_argument("--profile", type=str, default=None,
                        help="Research profile name (e.g. 'neuralsignal', 'trading').")
    parser.add_argument("--direction", type=str, default=None,
                        help="Research direction / question to investigate.")
    parser.add_argument("--loop", action="store_true", default=False,
                        help="Auto-loop: use top next_step as the next direction.")
    parser.add_argument("--list-profiles", action="store_true",
                        help="List available profiles and exit.")
    return parser.parse_args()


def _add_plugin_to_path(profile: dict) -> None:
    """Add any plugin-specific source paths to sys.path."""
    from pathlib import Path
    from configs.config import get_config
    cfg = get_config()

    if profile.get("name") == "neuralsignal":
        ns_path = Path(cfg.neuralsignal_src_path).resolve()
        if ns_path.exists() and str(ns_path) not in sys.path:
            sys.path.insert(0, str(ns_path))
            log.debug("Added neuralsignal to sys.path: %s", ns_path)
        elif not ns_path.exists():
            log.warning("neuralsignal_src_path not found: %s", ns_path)


def main() -> None:
    args = parse_args()

    if args.list_profiles:
        from core.utils.profile_loader import list_profiles
        profiles = list_profiles()
        if profiles:
            print("Available profiles:")
            for p in profiles:
                print(f"  {p}")
        else:
            print("No profiles found in configs/profiles/")
        return

    # Resolve profile
    profile_name = args.profile
    if not profile_name:
        from core.utils.profile_loader import list_profiles
        available = list_profiles()
        if not available:
            print("Error: no profiles found in configs/profiles/", file=sys.stderr)
            sys.exit(1)
        if len(available) == 1:
            profile_name = available[0]
            log.info("Auto-selected only available profile: %r", profile_name)
        else:
            print(f"Available profiles: {available}")
            profile_name = input("Profile: ").strip()

    from core.utils.profile_loader import load_profile
    try:
        profile = load_profile(profile_name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading profile {profile_name!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    _add_plugin_to_path(profile)

    # Resolve direction
    direction = args.direction
    if not direction:
        direction = input(f"[{profile_name}] Research direction: ").strip()
    if not direction:
        print("Error: a research direction is required.", file=sys.stderr)
        sys.exit(1)

    log.info("Starting pipeline — profile=%r, direction=%r", profile_name, direction)

    from core.graph.builder import build_graph
    graph = build_graph(profile)

    initial_state = {
        "profile_name": profile_name,
        "research_direction": direction,
        "continue_loop": args.loop,
        "errors": [],
    }

    print(f"\n[{profile_name}] Researching: {direction!r}\n")

    try:
        final_state = graph.invoke(initial_state)
    except Exception:
        log.critical("Pipeline raised an unhandled exception", exc_info=True)
        raise

    _print_results(final_state, profile_name)


def _print_results(state: dict, profile_name: str) -> None:
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETE  [{profile_name}]")
    print("=" * 72)

    stored = state.get("stored_result_ids") or []
    if stored:
        print(f"Stored results ({len(stored)}): {stored}")

    eval_summary = state.get("evaluation_summary") or {}
    if eval_summary:
        best = eval_summary.get("best_proposal")
        best_val = eval_summary.get("best_metric_value")
        metric = eval_summary.get("best_metric_name", "")
        if best:
            print(f"Best result: {best} — {metric}={best_val:.4f}" if isinstance(best_val, float) else f"Best: {best}")

    next_steps = state.get("next_steps") or []
    if next_steps:
        print(f"\nProposed next steps ({len(next_steps)}):")
        for i, s in enumerate(next_steps, 1):
            print(f"  {i}. [{s.get('priority', '?')}] {s.get('title', '(no title)')}")
            if s.get("rationale"):
                print(f"     {s['rationale']}")
            if s.get("suggested_direction"):
                print(f"     → {s['suggested_direction']}")

    errors = state.get("errors") or []
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
