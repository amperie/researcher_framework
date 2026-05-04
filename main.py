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
from typing import Any
from uuid import uuid4

from core.handoffs import resolve_next_step_seed, resolve_proposal_seed, resolve_run_handoff_seed
from core.graph.builder import pipeline_steps
from core.maintenance.dev_cleanup import run_periodic_dev_cleanup
from core.pipeline_resume import build_resume_state, ensure_resume_state_for_node
from core.utils.logger import setup_logging, get_logger

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
    parser.add_argument(
        "--source-experiment",
        type=str,
        default=None,
        help="Source experiment record id to seed from the latest saved UI handoff.",
    )
    parser.add_argument(
        "--handoff",
        type=str,
        default=None,
        help="Explicit saved handoff record id to launch from.",
    )
    parser.add_argument(
        "--proposal-seed",
        type=str,
        default=None,
        help="Explicit saved proposal seed record id to launch from proposal stage.",
    )
    parser.add_argument(
        "--next-step",
        type=str,
        default=None,
        help="Explicit persisted next_step record id to launch as the next direction.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Persisted memory record id to hydrate state from before restarting at a later node.",
    )
    parser.add_argument(
        "--start-node",
        type=str,
        default=None,
        help="Pipeline node to start from, then continue through all remaining nodes.",
    )
    parser.add_argument("--loop", action="store_true", default=False,
                        help="Auto-loop: use top next_step as the next direction.")
    parser.add_argument(
        "--run-next-steps-once",
        action="store_true",
        default=False,
        help="Run the initial direction once, then run each proposed next step once and stop.",
    )
    parser.add_argument("--list-profiles", action="store_true",
                        help="List available profiles and exit.")
    parser.add_argument("--list-nodes", action="store_true",
                        help="List pipeline nodes for the selected profile and exit.")
    return parser.parse_args()


def _choose_run_mode(args: argparse.Namespace) -> str:
    if args.loop:
        return "loop"
    if args.run_next_steps_once:
        return "next_steps_once"

    print("\nRun mode:")
    print("  1. Run once with the given direction")
    print("  2. Run once, then run all proposed next steps once")
    print("  3. Continuous loop using the top proposed next step")
    choice = input("Choose mode [1/2/3, default 1]: ").strip()
    if choice == "2":
        return "next_steps_once"
    if choice == "3":
        return "loop"
    return "single"


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
    cleanup = run_periodic_dev_cleanup()
    if not cleanup.skipped and (cleanup.deleted_files or cleanup.deleted_dirs or cleanup.errors):
        log.info(
            "dev_cleanup | files=%d dirs=%d errors=%d",
            len(cleanup.deleted_files),
            len(cleanup.deleted_dirs),
            len(cleanup.errors),
        )
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

    if args.list_nodes:
        print("Pipeline nodes:")
        for step in pipeline_steps(profile):
            print(f"  {step}")
        return

    _add_plugin_to_path(profile)

    if args.resume_from and (args.direction or args.source_experiment or args.handoff or args.proposal_seed or args.next_step):
        print(
            "Error: --resume-from cannot be used together with --direction, --source-experiment, --handoff, --proposal-seed, or --next-step.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.direction and (args.source_experiment or args.handoff or args.proposal_seed or args.next_step):
        print("Error: --direction cannot be used together with --source-experiment, --handoff, --proposal-seed, or --next-step.", file=sys.stderr)
        sys.exit(1)

    start_node = _resolve_start_node(profile_name, profile, args)
    profile_steps = pipeline_steps(profile)
    if not args.resume_from and start_node != profile_steps[0]:
        print(
            f"Error: --start-node {start_node!r} requires --resume-from because it is not the pipeline entry node.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve direction
    if args.resume_from:
        try:
            seed = build_resume_state(profile, str(args.resume_from or ""))
            ensure_resume_state_for_node(start_node, seed)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        log.info(
            "Resolved resume state profile=%r source_record=%r start_node=%r direction=%r",
            profile_name,
            args.resume_from,
            start_node,
            seed.get("research_direction"),
        )
    else:
        seed = _resolve_initial_seed(profile, profile_name, args)
    direction = str(seed["research_direction"])

    if args.loop and args.run_next_steps_once:
        print("Error: --loop and --run-next-steps-once cannot be used together.", file=sys.stderr)
        sys.exit(1)

    run_mode = _choose_run_mode(args)

    log.info("Starting pipeline — profile=%r, direction=%r mode=%s", profile_name, direction, run_mode)

    from core.graph.builder import build_graph
    graph = build_graph(profile, start_node=start_node)

    seen_directions: set[str] = set()
    current_direction = direction
    pending_next_step_runs: list[dict[str, str]] = []
    seeded_followup_runs = False
    total_planned_runs: int | None = 1 if run_mode == "single" else None
    completed_runs = 0
    root_run_family_id = str(seed.get("root_run_family_id") or f"{profile_name}:{uuid4()}")
    root_research_direction = str(seed.get("root_research_direction") or direction)
    initial_state: dict[str, Any] = {
        "profile_name": profile_name,
        "research_direction": current_direction,
        "continue_loop": run_mode == "loop",
        "root_run_family_id": root_run_family_id,
        "root_research_direction": root_research_direction,
        "errors": [],
    }
    if seed.get("source_next_step_record_id"):
        initial_state["source_next_step_record_id"] = seed["source_next_step_record_id"]
    if seed.get("source_next_step_title"):
        initial_state["source_next_step_title"] = seed["source_next_step_title"]
    if seed.get("source_proposal_seed_record_id"):
        initial_state["source_proposal_seed_record_id"] = seed["source_proposal_seed_record_id"]
    if seed.get("source_proposal_seed_title"):
        initial_state["source_proposal_seed_title"] = seed["source_proposal_seed_title"]
    if seed.get("proposal_seed_planning_notes"):
        initial_state["proposal_seed_planning_notes"] = seed["proposal_seed_planning_notes"]
    if seed.get("proposals"):
        initial_state["proposals"] = list(seed["proposals"])

    while True:
        current_run_number = completed_runs + 1
        if total_planned_runs is not None:
            log.info(
                "Top-level run %d/%d starting direction=%r mode=%s",
                current_run_number,
                total_planned_runs,
                current_direction,
                run_mode,
            )
            print(f"\n[{profile_name}] Top-level run {current_run_number}/{total_planned_runs}")
        else:
            log.info(
                "Top-level run %d starting direction=%r mode=%s",
                current_run_number,
                current_direction,
                run_mode,
            )
            print(f"\n[{profile_name}] Top-level run {current_run_number}")
        print(f"\n[{profile_name}] Researching: {current_direction!r}\n")
        try:
            final_state = graph.invoke(initial_state)
        except Exception:
            log.critical("Pipeline raised an unhandled exception", exc_info=True)
            raise

        _print_results(final_state, profile_name)
        seen_directions.add(current_direction)
        completed_runs += 1

        if run_mode == "single":
            break

        if run_mode == "next_steps_once":
            if not seeded_followup_runs:
                pending_next_step_runs = _next_step_seeds(profile_name, current_direction, final_state)
                seeded_followup_runs = True
                if not pending_next_step_runs:
                    log.info("Stopping because no proposed next steps were returned.")
                    break
                pending_next_step_runs = [
                    seed for seed in pending_next_step_runs
                    if str(seed.get("research_direction") or "").strip() not in seen_directions
                ]
                if not pending_next_step_runs:
                    log.info("Stopping because all proposed next steps have already been seen.")
                    break
                total_planned_runs = 1 + len(pending_next_step_runs)
                log.info("Running %d proposed next step(s) once before stopping.", len(pending_next_step_runs))
            elif not pending_next_step_runs:
                log.info("Completed one-time follow-up run set; stopping without recursing into later next steps.")
                break
            next_seed = pending_next_step_runs.pop(0)
            next_direction = str(next_seed["research_direction"])
            if next_direction in seen_directions:
                log.info("Skipping one-time follow-up because direction was already seen: %r", next_direction)
                if pending_next_step_runs:
                    continue
                break
            log.info(
                "Running one-time follow-up next_step=%r next_direction=%r remaining=%d",
                next_seed.get("source_next_step_title"),
                next_direction,
                len(pending_next_step_runs),
            )
            current_direction = next_direction
            initial_state = {
                "profile_name": profile_name,
                "research_direction": current_direction,
                "continue_loop": False,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "errors": [],
                "source_next_step_record_id": next_seed["source_next_step_record_id"],
                "source_next_step_title": next_seed["source_next_step_title"],
            }
            continue

        next_seed = _next_loop_seed(profile_name, current_direction, final_state)
        if not next_seed:
            break
        next_direction = str(next_seed["research_direction"])
        if not next_direction or next_direction in seen_directions:
            log.info("Loop stopping because next direction is empty or already seen: %r", next_direction)
            break
        log.info(
            "Loop continuing with next_step=%r next_direction=%r",
            next_seed.get("source_next_step_title"),
            next_direction,
        )
        current_direction = next_direction
        initial_state = {
            "profile_name": profile_name,
            "research_direction": current_direction,
            "continue_loop": True,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "errors": [],
            "source_next_step_record_id": next_seed["source_next_step_record_id"],
            "source_next_step_title": next_seed["source_next_step_title"],
        }


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
    selection = state.get("next_step_selection") or {}
    dropped = selection.get("dropped") or []
    if selection:
        print(
            f"\nNext-step selection: mode={selection.get('ranking_mode', '?')} "
            f"selected={len(selection.get('selected') or [])} dropped={len(dropped)}"
        )
        for item in dropped[:5]:
            title = item.get("title") or item.get("suggested_direction") or "(untitled)"
            print(f"  drop: {title} [{item.get('drop_reason', '?')}]")
        if len(dropped) > 5:
            print(f"  ... {len(dropped) - 5} more dropped candidate(s)")

    errors = state.get("errors") or []
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    print("=" * 72 + "\n")


def _resolve_start_node(profile_name: str, profile: dict[str, Any], args: argparse.Namespace) -> str:
    steps = pipeline_steps(profile)
    if args.start_node:
        if args.start_node not in steps:
            print(
                f"Error: start node {args.start_node!r} is not in pipeline: {steps}",
                file=sys.stderr,
            )
            sys.exit(1)
        return str(args.start_node)

    if args.resume_from:
        print("\nStart node options:")
        for index, step in enumerate(steps, 1):
            print(f"  {index}. {step}")
        choice = input(f"[{profile_name}] Start at node [default {steps[0]}]: ").strip()
        if not choice:
            return steps[0]
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(steps):
                return steps[selected - 1]
        if choice in steps:
            return choice
        print(f"Error: invalid start node selection {choice!r}.", file=sys.stderr)
        sys.exit(1)

    return steps[0]


def _resolve_initial_seed(profile: dict[str, Any], profile_name: str, args: argparse.Namespace) -> dict[str, Any]:
    selected_seed_args = [
        bool(args.handoff),
        bool(args.proposal_seed),
        bool(args.next_step),
    ]
    if sum(1 for item in selected_seed_args if item) > 1:
        print("Error: --handoff, --proposal-seed, and --next-step cannot be used together.", file=sys.stderr)
        sys.exit(1)

    if args.next_step:
        try:
            seed = resolve_next_step_seed(
                profile,
                next_step_record_id=str(args.next_step or ""),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        log.info(
            "Resolved next step seed profile=%r next_step=%r direction=%r",
            profile_name,
            args.next_step,
            seed.get("research_direction"),
        )
        return seed

    if args.proposal_seed:
        try:
            seed = resolve_proposal_seed(
                profile,
                source_experiment_record_id=str(args.source_experiment or ""),
                proposal_seed_record_id=str(args.proposal_seed or ""),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        log.info(
            "Resolved proposal seed profile=%r source_experiment=%r proposal_seed=%r",
            profile_name,
            args.source_experiment,
            args.proposal_seed,
        )
        return seed

    if args.handoff or args.source_experiment:
        try:
            seed = resolve_run_handoff_seed(
                profile,
                source_experiment_record_id=str(args.source_experiment or ""),
                handoff_record_id=str(args.handoff or ""),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        log.info(
            "Resolved saved handoff seed profile=%r source_experiment=%r handoff=%r direction=%r",
            profile_name,
            args.source_experiment,
            args.handoff,
            seed.get("research_direction"),
        )
        return seed

    direction = str(args.direction or "").strip()
    if not direction:
        direction = input(f"[{profile_name}] Research direction: ").strip()
    if not direction:
        print("Error: a research direction is required.", file=sys.stderr)
        sys.exit(1)
    return {"research_direction": direction}


def _next_loop_seed(profile_name: str, current_direction: str, state: dict[str, Any]) -> dict[str, str] | None:
    seeds = _next_step_seeds(profile_name, current_direction, state)
    return seeds[0] if seeds else None


def _next_step_seeds(profile_name: str, current_direction: str, state: dict[str, Any]) -> list[dict[str, str]]:
    from core.memory.defaults import next_step_record_id

    next_steps = state.get("next_steps") or []
    seeds: list[dict[str, str]] = []
    for step in next_steps:
        next_direction = str(step.get("suggested_direction") or step.get("title") or "").strip()
        if not next_direction:
            continue
        title = str(step.get("title") or step.get("suggested_direction") or "next_step").strip()
        seeds.append({
            "research_direction": next_direction,
            "source_next_step_record_id": next_step_record_id(profile_name, current_direction, step),
            "source_next_step_title": title,
        })
    return seeds


if __name__ == "__main__":
    main()
