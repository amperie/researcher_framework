"""Research pipeline CLI entry point.

Usage:
    uv run python main.py help
    uv run python main.py --list-profiles
    uv run python main.py --profile trading_researcher --direction "attention head specialization"
    uv run python main.py --mode brainstorm --profile trading --direction "SPY regime filters"
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from configs.config import dev_path
from core.handoffs import resolve_next_step_seed, resolve_proposal_seed, resolve_run_handoff_seed
from core.graph.builder import pipeline_steps
from core.maintenance.dev_cleanup import run_periodic_dev_cleanup
from core.pipeline_resume import build_resume_state, ensure_resume_state_for_node
from core.utils.logger import setup_logging, get_logger, temporarily_raise_console_log_level
from core.utils import terminal_progress

setup_logging()
log = get_logger(__name__)

_CLI_EPILOG = """Examples:
  Help and discovery:
    uv run python main.py help
    uv run python main.py --list-profiles
    uv run python main.py --profile trading_researcher --list-nodes

  Pipeline from a direction:
    uv run python main.py --profile trading_researcher --direction "attention head specialization"
    uv run python main.py --profile trading --direction "SPY regime filters" --run-next-steps-once
    uv run python main.py --profile trading_researcher --direction "residual entropy features" --loop

  Pipeline from saved work:
    uv run python main.py --profile trading_researcher --next-step "next_step:1"
    uv run python main.py --profile trading_researcher --source-experiment "exp-123" --proposal-seed "proposal_seed:1"
    uv run python main.py --profile trading_researcher --source-experiment "exp-123" --handoff "run_handoff:1"
    uv run python main.py --profile trading_researcher --resume-from "experiment_result:123" --start-node implement
    uv run python main.py --profile trading_researcher --start-node check_experiment_jobs --resume-snapshot
    uv run python main.py --profile trading_researcher --start-node submit_experiment_jobs --resume-snapshot --force-dataset-refresh

  Brainstorm mode:
    uv run python main.py --mode brainstorm --profile trading_researcher --direction "new detector direction"
    uv run python main.py --mode brainstorm --profile trading --config configs/brainstorm/default.trading.brainstorm.yaml
    uv run python main.py --mode brainstorm --profile trading --resume-brainstorm "<session-id>"
    uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123"
    uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --proposal-seed "proposal_seed:123"
    uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --handoff "run_handoff:123"
    uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --next-step "next_step:123"

  Campaign metadata:
    uv run python main.py --profile trading --direction "factor rotation" --campaign-id "camp-1" --campaign-title "Macro rotation" --campaign-variant-id "v1" --campaign-variant-title "Baseline" --campaign-variant-index 1 --campaign-size 3

Notes:
  --mode defaults to pipeline.
  Omit --direction in pipeline mode to be prompted.
  --config is an alias for --brainstorm-config.
  Use --resume-snapshot with no value to load dev/state/<profile>/after_<previous-node>.json.
  Use --force-dataset-refresh only for TradingResearcher dataset regeneration.

Interactive brainstorm commands after pause:
  help, continue, summary, research, plan, feedback <text>, approve_plan, execute, exit
"""


class _BrainstormSpinner:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._message = ""
        self._stream = sys.stdout

    def start(self, message: str) -> None:
        self.stop()
        self._message = str(message or "").strip() or "thinking"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="brainstorm-spinner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=0.5)
        self._thread = None
        self._clear_line()

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        index = 0
        while not self._stop_event.is_set():
            frame = frames[index % len(frames)]
            print(f"\r[{frame}] {self._message}...", end="", file=self._stream, flush=True)
            index += 1
            if self._stop_event.wait(0.1):
                break

    def _clear_line(self) -> None:
        print("\r" + (" " * 80) + "\r", end="", file=self._stream, flush=True)


_BRAINSTORM_ROLE_COLORS = {
    "facilitator": "\x1b[38;5;81m",
    "skeptic": "\x1b[38;5;179m",
    "researcher": "\x1b[38;5;114m",
    "seed": "\x1b[38;5;141m",
}
_ANSI_RESET = "\x1b[0m"
_ANSI_CHECKPOINT = "\x1b[1;38;5;82m"    # bold bright green
_ANSI_INTERRUPTED = "\x1b[1;38;5;214m"  # bold orange
_ANSI_MARKER = "\x1b[38;5;67m"          # steel blue — subtle section markers
_ANSI_LABEL = "\x1b[38;5;244m"          # medium gray — summary/plan labels

_BRAINSTORM_BANNERS: dict[str, tuple[str, str]] = {
    "[checkpoint]":      (_ANSI_CHECKPOINT, "CHECKPOINT"),
    "[interrupted]":     (_ANSI_INTERRUPTED, "INTERRUPTED"),
    "[current thinking]": (_ANSI_MARKER,    "Current thinking"),
    "[plan]":            (_ANSI_MARKER,      "Plan"),
}

_SUMMARY_LABELS = {
    "goal", "agreed", "ideas", "risks", "evidence", "questions", "next",
    "direction", "refined ideas", "proposals", "implementation plans",
    "constraints", "exclusions", "success criteria", "unresolved questions",
}


def _brainstorm_banner(label: str, color: str, width: int = 68) -> str:
    inner = f" {label} "
    pad = max(2, (width - len(inner)) // 2)
    right_pad = width - pad - len(inner)
    return f"{color}{'─' * pad}{inner}{'─' * right_pad}{_ANSI_RESET}"


def _write_state_snapshot(profile_name: str, step_name: str, state: dict[str, Any]) -> None:
    serialisable: dict[str, Any] = {}
    for key, value in state.items():
        try:
            json.dumps(value, default=str)
            serialisable[key] = value
        except Exception:
            log.debug("main | Skipping non-serialisable key %r", key)
    out_path = dev_path("state", profile_name, f"after_{step_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")


def _format_brainstorm_cli_output(text: str) -> str:
    raw = str(text or "")
    lines = raw.splitlines()
    if not lines:
        return raw
    formatted: list[str] = []
    for line in lines:
        formatted.append(_colorize_brainstorm_role_line(line))
    suffix = "\n" if raw.endswith("\n") else ""
    return "\n".join(formatted) + suffix


def _colorize_brainstorm_role_line(line: str) -> str:
    raw = str(line or "")
    key = raw.strip().lower()

    # Checkpoint / section banners
    if key in _BRAINSTORM_BANNERS:
        color, label = _BRAINSTORM_BANNERS[key]
        return _brainstorm_banner(label, color)

    # Role-name tags: [facilitator] ..., [skeptic] ..., etc.
    if raw.lstrip().startswith("["):
        stripped = raw.lstrip()
        closing = stripped.find("]")
        if closing > 1:
            role_name = stripped[1:closing].strip().lower()
            color = _BRAINSTORM_ROLE_COLORS.get(role_name)
            if color:
                leading = raw[: len(raw) - len(stripped)]
                prefix = stripped[: closing + 1]
                rest = stripped[closing + 1:]
                return f"{leading}{color}{prefix}{_ANSI_RESET}{rest}"

    # Summary / plan section labels: "Goal: text", "Agreed:", "Risks:", etc.
    colon = raw.find(":")
    if colon > 0:
        label_text = raw[:colon].strip().lower()
        if label_text in _SUMMARY_LABELS:
            return f"{_ANSI_LABEL}{raw[:colon + 1]}{_ANSI_RESET}{raw[colon + 1:]}"

    return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configuration-driven research pipeline and interactive brainstorm runner.",
        epilog=_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", type=str, default=None,
                        help="Profile name from configs/profiles, e.g. trading_researcher or trading. If omitted, the CLI prompts when multiple profiles exist.")
    parser.add_argument(
        "--mode",
        type=str,
        default="pipeline",
        choices=["pipeline", "brainstorm"],
        help="Runner mode. pipeline executes the graph; brainstorm starts an interactive planning session.",
    )
    parser.add_argument("--direction", type=str, default=None,
                        help="Fresh research direction or question. In pipeline mode, omit it to be prompted.")
    parser.add_argument(
        "--brainstorm-config",
        "--config",
        type=str,
        default=None,
        help="Brainstorm YAML config path. --config is the short alias.",
    )
    parser.add_argument(
        "--resume-brainstorm",
        type=str,
        default=None,
        help="Existing brainstorm session id to resume.",
    )
    parser.add_argument(
        "--source-experiment",
        type=str,
        default=None,
        help="Prior experiment_result id. Used with seeds in pipeline mode or imported as context in brainstorm mode.",
    )
    parser.add_argument(
        "--handoff",
        type=str,
        default=None,
        help="Saved run_handoff id. Pipeline launches from it; brainstorm imports it as editable context.",
    )
    parser.add_argument(
        "--proposal-seed",
        type=str,
        default=None,
        help="Saved proposal_seed id. Pipeline starts from proposed experiments; brainstorm imports it.",
    )
    parser.add_argument(
        "--next-step",
        type=str,
        default=None,
        help="Persisted next_step id. Pipeline runs the recommendation; brainstorm imports it.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Memory record id used to rebuild pipeline state for resume.",
    )
    parser.add_argument(
        "--resume-snapshot",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Load resume state from a JSON snapshot. With no value, loads "
            "dev/state/<profile>/after_<previous-node>.json for --start-node."
        ),
    )
    parser.add_argument(
        "--start-node",
        type=str,
        default=None,
        help="Pipeline node to start at, then continue through remaining profile steps.",
    )
    parser.add_argument(
        "--force-dataset-refresh",
        action="store_true",
        default=False,
        help="TradingResearcher only: regenerate datasets instead of reusing memory records or local CSVs.",
    )
    parser.add_argument("--campaign-id", type=str, default=None, help="Campaign id to attach to stored results.")
    parser.add_argument("--campaign-title", type=str, default=None, help="Human-readable campaign title.")
    parser.add_argument("--campaign-variant-id", type=str, default=None, help="Variant id within the campaign.")
    parser.add_argument("--campaign-variant-title", type=str, default=None, help="Human-readable variant title.")
    parser.add_argument("--campaign-variant-index", type=int, default=0, help="1-based variant index within the campaign.")
    parser.add_argument("--campaign-size", type=int, default=0, help="Total planned variants in the campaign.")
    parser.add_argument("--loop", action="store_true", default=False,
                        help="After each pipeline run, continue with the top proposed next_step until no new step remains.")
    parser.add_argument(
        "--run-next-steps-once",
        action="store_true",
        default=False,
        help="Run the initial pipeline once, then run each proposed next_step once and stop.",
    )
    parser.add_argument("--list-profiles", action="store_true",
                        help="List available profiles and exit.")
    parser.add_argument("--list-nodes", action="store_true",
                        help="List pipeline nodes for the selected profile and exit.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if len(args_list) == 1 and args_list[0].strip().lower() == "help":
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args(args_list)


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

    if profile.get("name") == "trading_researcher":
        ns_path = Path(cfg.trading_researcher_src_path).resolve()
        if ns_path.exists() and str(ns_path) not in sys.path:
            sys.path.insert(0, str(ns_path))
            log.debug("Added trading_researcher to sys.path: %s", ns_path)
        elif not ns_path.exists():
            log.warning("trading_researcher_src_path not found: %s", ns_path)


def build_initial_state(
    profile_name: str,
    direction: str,
    seed: dict[str, Any],
    *,
    continue_loop: bool,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_run_family_id = str(seed.get("root_run_family_id") or f"{profile_name}:{uuid4()}")
    root_research_direction = str(seed.get("root_research_direction") or direction)
    initial_state: dict[str, Any] = {
        "profile_name": profile_name,
        "research_direction": direction,
        "continue_loop": continue_loop,
        "root_run_family_id": root_run_family_id,
        "root_research_direction": root_research_direction,
        "errors": [],
    }
    for key, value in seed.items():
        if key in {"profile_name", "research_direction", "continue_loop", "root_run_family_id", "root_research_direction"}:
            continue
        if value not in (None, "", [], {}):
            initial_state[key] = value
    for key in (
        "source_next_step_record_id",
        "source_next_step_title",
        "source_proposal_seed_record_id",
        "source_proposal_seed_title",
        "proposal_seed_planning_notes",
        "campaign_id",
        "campaign_title",
        "campaign_variant_id",
        "campaign_variant_title",
        "campaign_variant_index",
        "campaign_size",
    ):
        if seed.get(key) not in (None, "", 0):
            initial_state[key] = seed[key]
    if extra_state:
        for key, value in extra_state.items():
            if value not in (None, "", 0):
                initial_state[key] = value
    return initial_state


def run_pipeline_graph(
    profile_name: str,
    profile: dict[str, Any],
    *,
    initial_state: dict[str, Any],
    start_node: str,
    print_results: bool = True,
) -> dict[str, Any]:
    from core.graph.builder import build_graph

    graph = build_graph(profile, start_node=start_node)
    steps = pipeline_steps(profile)
    if start_node in steps:
        steps = steps[steps.index(start_node):]
    final_step = steps[-1]
    terminal_progress.configure_pipeline(steps)
    log.info(
        "Invoking pipeline graph profile=%r start_node=%r direction=%r family=%r campaign=%r",
        profile_name,
        start_node,
        initial_state.get("research_direction"),
        initial_state.get("root_run_family_id"),
        initial_state.get("campaign_id"),
    )
    try:
        final_state = graph.invoke(initial_state)
        terminal_progress.finish_stage(final_step)
        _write_state_snapshot(profile_name, final_step, final_state)
        return final_state
    finally:
        terminal_progress.clear()
        if "final_state" in locals() and print_results:
            _print_results(final_state, profile_name)


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
    if args.force_dataset_refresh:
        _force_dataset_refresh(profile)

    if args.mode == "brainstorm":
        _run_brainstorm_mode(args, profile_name, profile)
        return

    if args.list_nodes:
        print("Pipeline nodes:")
        for step in pipeline_steps(profile):
            print(f"  {step}")
        return

    _add_plugin_to_path(profile)

    if args.resume_from and args.resume_snapshot:
        print("Error: --resume-from and --resume-snapshot cannot be used together.", file=sys.stderr)
        sys.exit(1)

    if (args.resume_from or args.resume_snapshot) and (
        args.direction or args.source_experiment or args.handoff or args.proposal_seed or args.next_step
    ):
        print(
            "Error: resume options cannot be used together with --direction, --source-experiment, --handoff, --proposal-seed, or --next-step.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.direction and (args.source_experiment or args.handoff or args.proposal_seed or args.next_step):
        print("Error: --direction cannot be used together with --source-experiment, --handoff, --proposal-seed, or --next-step.", file=sys.stderr)
        sys.exit(1)

    start_node = _resolve_start_node(profile_name, profile, args)
    profile_steps = pipeline_steps(profile)
    if not (args.resume_from or args.resume_snapshot) and start_node != profile_steps[0]:
        print(
            f"Error: --start-node {start_node!r} requires --resume-from or --resume-snapshot because it is not the pipeline entry node.",
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
    elif args.resume_snapshot:
        try:
            seed = _load_resume_snapshot(profile_name, profile, start_node, str(args.resume_snapshot or "auto"))
            ensure_resume_state_for_node(start_node, seed)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        log.info(
            "Resolved snapshot resume state profile=%r snapshot=%r start_node=%r direction=%r",
            profile_name,
            args.resume_snapshot,
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

    seen_directions: set[str] = set()
    current_direction = direction
    pending_next_step_runs: list[dict[str, str]] = []
    seeded_followup_runs = False
    total_planned_runs: int | None = 1 if run_mode == "single" else None
    completed_runs = 0
    root_run_family_id = str(seed.get("root_run_family_id") or f"{profile_name}:{uuid4()}")
    root_research_direction = str(seed.get("root_research_direction") or direction)
    seed["root_run_family_id"] = root_run_family_id
    seed["root_research_direction"] = root_research_direction
    campaign_state = {
        "campaign_id": str(args.campaign_id or seed.get("campaign_id") or ""),
        "campaign_title": str(args.campaign_title or seed.get("campaign_title") or ""),
        "campaign_variant_id": str(args.campaign_variant_id or seed.get("campaign_variant_id") or ""),
        "campaign_variant_title": str(args.campaign_variant_title or seed.get("campaign_variant_title") or ""),
        "campaign_variant_index": int(args.campaign_variant_index or seed.get("campaign_variant_index") or 0),
        "campaign_size": int(args.campaign_size or seed.get("campaign_size") or 0),
    }
    initial_state = build_initial_state(
        profile_name,
        current_direction,
        seed,
        continue_loop=(run_mode == "loop"),
        extra_state=campaign_state,
    )

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
            final_state = run_pipeline_graph(
                profile_name,
                profile,
                initial_state=initial_state,
                start_node=start_node,
                print_results=False,
            )
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
            next_run_seed = {
                "research_direction": current_direction,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "source_next_step_record_id": next_seed["source_next_step_record_id"],
                "source_next_step_title": next_seed["source_next_step_title"],
                **campaign_state,
            }
            initial_state = build_initial_state(
                profile_name,
                current_direction,
                next_run_seed,
                continue_loop=False,
            )
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
        next_run_seed = {
            "research_direction": current_direction,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "source_next_step_record_id": next_seed["source_next_step_record_id"],
            "source_next_step_title": next_seed["source_next_step_title"],
            **campaign_state,
        }
        initial_state = build_initial_state(
            profile_name,
            current_direction,
            next_run_seed,
            continue_loop=True,
        )


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

    if args.resume_from or args.resume_snapshot:
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


def _load_resume_snapshot(
    profile_name: str,
    profile: dict[str, Any],
    start_node: str,
    snapshot: str,
) -> dict[str, Any]:
    path = _resume_snapshot_path(profile_name, profile, start_node, snapshot)
    if not path.exists():
        raise ValueError(f"Resume snapshot not found: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Resume snapshot is not valid JSON: {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Resume snapshot must contain a JSON object: {path}")
    state.setdefault("profile_name", profile_name)
    state.setdefault("errors", [])
    return state


def _resume_snapshot_path(profile_name: str, profile: dict[str, Any], start_node: str, snapshot: str) -> Path:
    if snapshot and snapshot != "auto":
        return Path(snapshot).expanduser().resolve()
    steps = pipeline_steps(profile)
    if start_node not in steps:
        raise ValueError(f"Start node {start_node!r} is not in pipeline: {steps}")
    index = steps.index(start_node)
    if index <= 0:
        raise ValueError("--resume-snapshot auto requires --start-node to be after the first pipeline node")
    previous = _canonical_step_name(steps[index - 1])
    return dev_path("state", profile_name, f"after_{previous}.json")


def _canonical_step_name(node_name: str) -> str:
    raw = str(node_name or "").strip().lower()
    return raw.replace(" ", "_").replace("-", "_") if raw else "unknown"


def _force_dataset_refresh(profile: dict[str, Any]) -> None:
    for dataset in profile.get("datasets") or []:
        if isinstance(dataset, dict):
            dataset["overwrite_existing_dataset"] = True


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


def _edit_brainstorm_plan(state: dict[str, Any], brainstorm_cfg: dict[str, Any]) -> dict[str, Any]:
    import os
    import shlex
    import subprocess
    import tempfile
    from pathlib import Path as _Path
    from core.brainstorm.handoff import build_execution_handoff
    from core.brainstorm.summaries import render_consensus_summary

    plan = dict(state.get("plan_draft") or {})
    plan_text = json.dumps(plan, indent=2, ensure_ascii=False)

    editor = (
        os.environ.get("EDITOR")
        or os.environ.get("VISUAL")
        or ("notepad" if sys.platform == "win32" else "vi")
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="brainstorm_plan_",
        delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write(plan_text)
        tmp_path = tmp.name

    print(f"Opening plan in {editor!r} — save and close to apply changes.")
    try:
        subprocess.run(shlex.split(editor) + [tmp_path], check=False)
        edited_text = _Path(tmp_path).read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Editor error: {exc}", file=sys.stderr)
        return state
    finally:
        try:
            _Path(tmp_path).unlink()
        except OSError:
            pass

    if edited_text.strip() == plan_text.strip():
        print("No changes.")
        return state

    try:
        edited = json.loads(edited_text)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON — changes discarded: {exc}", file=sys.stderr)
        return state

    if not isinstance(edited, dict):
        print("Plan must be a JSON object — changes discarded.", file=sys.stderr)
        return state

    state["plan_draft"] = edited
    state["execution_handoff"] = build_execution_handoff(state, brainstorm_cfg)
    state["last_summary"] = render_consensus_summary(state)
    print("Plan updated.")
    return state


def _run_brainstorm_mode(args: argparse.Namespace, profile_name: str, profile: dict[str, Any]) -> None:
    from core.brainstorm import (
        BrainstormEngine,
        HELP_TEXT,
        BrainstormConfigError,
        create_brainstorm_state,
        execute_brainstorm_handoff,
        list_brainstorm_configs,
        load_brainstorm_config,
        load_brainstorm_session,
        persist_brainstorm_session,
        resolve_brainstorm_seed,
    )

    brainstorm_cfg = _load_brainstorm_config_for_cli(
        profile_name=profile_name,
        path=args.brainstorm_config,
        load_brainstorm_config_fn=load_brainstorm_config,
        list_brainstorm_configs_fn=list_brainstorm_configs,
        error_type=BrainstormConfigError,
    )
    engine = BrainstormEngine(profile, brainstorm_cfg)

    if args.resume_brainstorm:
        state = load_brainstorm_session(profile, str(args.resume_brainstorm))
    else:
        try:
            seed = resolve_brainstorm_seed(
                profile,
                source_experiment_record_id=str(args.source_experiment or ""),
                handoff_record_id=str(args.handoff or ""),
                proposal_seed_record_id=str(args.proposal_seed or ""),
                next_step_record_id=str(args.next_step or ""),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        direction = str(seed.get("research_direction") or args.direction or "").strip()
        if not direction:
            direction = input(f"[{profile_name}] Brainstorm direction: ").strip()
        if not direction:
            print("Error: a brainstorm direction is required.", file=sys.stderr)
            sys.exit(1)
        state = create_brainstorm_state(
            profile_name=profile_name,
            direction=direction,
            brainstorm_cfg=brainstorm_cfg,
            seed=seed,
        )

    def _emit(text: str) -> None:
        if text:
            print(_format_brainstorm_cli_output(text).rstrip())

    spinner = _BrainstormSpinner()

    def _on_role_start(role: dict[str, Any], _round_index: int) -> None:
        role_name = str(role.get("name") or role.get("persona_type") or "role").strip()
        spinner.start(f"{role_name} thinking")

    def _on_role_end(_role: dict[str, Any], _round_index: int) -> None:
        spinner.stop()

    try:
        with temporarily_raise_console_log_level("WARNING"):
            state = engine.run_until_pause(
                state,
                emit=_emit,
                on_role_start=_on_role_start,
                on_role_end=_on_role_end,
            )
        persist_brainstorm_session(profile, brainstorm_cfg, state)

        while True:
            if state.get("status") == "cancelled":
                print("Brainstorm session exited.")
                return
            if state.get("status") == "approved_for_execution":
                start_node, _result = execute_brainstorm_handoff(
                    state,
                    brainstorm_cfg,
                    build_initial_state_fn=build_initial_state,
                    run_pipeline_graph_fn=run_pipeline_graph,
                    profile_name=profile_name,
                    profile=profile,
                )
                persist_brainstorm_session(profile, brainstorm_cfg, state)
                print(f"Brainstorm handoff executed from node: {start_node}")
                return

            command = input("brainstorm> ").strip()
            if not command:
                command = "continue"
            if command == "help":
                print(HELP_TEXT.rstrip())
                continue
            if command == "edit_plan":
                state = _edit_brainstorm_plan(state, brainstorm_cfg)
                persist_brainstorm_session(profile, brainstorm_cfg, state)
                continue
            if command == "execute":
                state = engine.apply_command(
                    state,
                    command,
                    emit=_emit,
                    on_role_start=_on_role_start,
                    on_role_end=_on_role_end,
                )
            else:
                with temporarily_raise_console_log_level("WARNING"):
                    state = engine.apply_command(
                        state,
                        command,
                        emit=_emit,
                        on_role_start=_on_role_start,
                        on_role_end=_on_role_end,
                    )
            persist_brainstorm_session(profile, brainstorm_cfg, state)
    finally:
        spinner.stop()


def _load_brainstorm_config_for_cli(
    *,
    profile_name: str,
    path: str | None,
    load_brainstorm_config_fn,
    list_brainstorm_configs_fn,
    error_type,
) -> dict[str, Any]:
    if not path:
        config_paths = list_brainstorm_configs_fn()
        if config_paths:
            selected = _select_brainstorm_config_path(profile_name=profile_name, config_paths=config_paths)
            return load_brainstorm_config_fn(str(selected))
    try:
        return load_brainstorm_config_fn(path)
    except error_type as exc:
        if path:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        config_paths = list_brainstorm_configs_fn()
        if not config_paths:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        selected = _select_brainstorm_config_path(profile_name=profile_name, config_paths=config_paths)
        return load_brainstorm_config_fn(str(selected))


def _select_brainstorm_config_path(*, profile_name: str, config_paths: list[Any]) -> Any:
    normalized_paths = [str(item) for item in config_paths]
    if len(normalized_paths) == 1:
        print(f"[{profile_name}] Brainstorm config: {normalized_paths[0]}")
        return config_paths[0]
    default_index = _default_brainstorm_config_index(profile_name=profile_name, config_paths=normalized_paths)
    print(f"\n[{profile_name}] Brainstorm config:")
    for index, config_path in enumerate(normalized_paths, 1):
        marker = " (default)" if index == default_index else ""
        print(f"  {index}. {config_path.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}{marker}")
    choice = input(f"Choose brainstorm config [default {default_index}]: ").strip()
    selected_index = default_index
    if choice:
        if not choice.isdigit():
            print(f"Error: invalid brainstorm config selection {choice!r}.", file=sys.stderr)
            sys.exit(1)
        selected_index = int(choice)
    if not (1 <= selected_index <= len(normalized_paths)):
        print(f"Error: invalid brainstorm config selection {selected_index!r}.", file=sys.stderr)
        sys.exit(1)
    return config_paths[selected_index - 1]


def _default_brainstorm_config_index(*, profile_name: str, config_paths: list[str]) -> int:
    profile_token = str(profile_name or "").strip().lower()
    if profile_token:
        for index, path in enumerate(config_paths, 1):
            filename = path.rsplit("/", 1)[-1].rsplit(chr(92), 1)[-1].lower()
            if profile_token in filename:
                return index
    return 1


if __name__ == "__main__":
    main()
