"""Dev harness for manually running individual pipeline steps.

Usage:
    uv run python run_node.py research --profile neuralsignal --direction "sparse autoencoders"
    uv run python run_node.py implement --profile neuralsignal --state-file dev/state/after_plan.json
    uv run python run_node.py --list
    uv run python run_node.py           # interactive mode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.utils import setup_logging, get_logger
from core.utils import fmt_value

setup_logging()
log = get_logger(__name__)

from core.graph.nodes import STEP_REGISTRY  # noqa: E402

_SEP = "-" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single pipeline step for manual testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("step", nargs="?", choices=list(STEP_REGISTRY),
                        metavar="step", help=f"Step to run. Choices: {', '.join(STEP_REGISTRY)}")
    parser.add_argument("--profile", metavar="NAME", default=None,
                        help="Research profile name (e.g. 'neuralsignal').")
    parser.add_argument("--state-file", metavar="PATH",
                        help="JSON file to load as initial state.")
    parser.add_argument("--direction", metavar="TEXT",
                        help="Override state['research_direction'].")
    parser.add_argument("--out", metavar="PATH",
                        help="Where to save merged state (default: dev/state/after_<step>.json).")
    parser.add_argument("--no-save", action="store_true",
                        help="Print output but do not write state file.")
    parser.add_argument("--list", action="store_true",
                        help="List available steps and exit.")
    return parser.parse_args()


def _load_profile(profile_name: str | None, state: dict) -> dict:
    """Load the profile by name, falling back to state['profile_name'], then first available."""
    from core.utils.profile_loader import load_profile, list_profiles
    name = profile_name or state.get("profile_name")
    if not name:
        available = list_profiles()
        if not available:
            raise RuntimeError("No profiles found in configs/profiles/")
        name = available[0]
        log.info("run_node | Auto-selected profile: %r", name)
    return load_profile(name)


def _add_plugin_to_path(profile: dict) -> None:
    from pathlib import Path
    from configs.config import get_config
    cfg = get_config()
    if profile.get("name") == "neuralsignal":
        ns_path = Path(cfg.neuralsignal_src_path).resolve()
        if ns_path.exists() and str(ns_path) not in sys.path:
            sys.path.insert(0, str(ns_path))


def _run_once(step_name: str, args: argparse.Namespace) -> None:
    state: dict = {}

    if args.state_file:
        path = Path(args.state_file)
        if not path.exists():
            print(f"[run_node] ERROR: state file not found: {path}", file=sys.stderr)
            sys.exit(1)
        state = json.loads(path.read_text(encoding="utf-8"))
        print(f"[run_node] State loaded from: {path}")
    else:
        print("[run_node] Starting with empty state")

    if args.direction:
        state["research_direction"] = args.direction
    if args.profile:
        state["profile_name"] = args.profile

    profile = _load_profile(args.profile, state)
    _add_plugin_to_path(profile)

    node_fn = STEP_REGISTRY[step_name]
    present = [k for k, v in state.items() if v not in (None, [], {}, "")]
    print(f"[run_node] Running step: {step_name}  (profile={profile.get('name')!r})")
    print(f"[run_node] State keys present: {present}")
    print(_SEP)

    try:
        delta = node_fn(state, profile)
    except NotImplementedError as exc:
        print(f"\n[run_node] NOT IMPLEMENTED: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[run_node] ERROR: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        raise

    print("Step output (keys written to state):")
    for key, val in delta.items():
        print(f"  {key:<30} -> {fmt_value(val)}")
        if isinstance(val, str) and len(val) < 300:
            for line in val.splitlines()[:5]:
                print(f"    {line}")
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            for item in val[:2]:
                preview = {k: v for k, v in list(item.items())[:4]}
                print(f"    {preview}")
    print(_SEP)

    if args.no_save:
        print("[run_node] --no-save: state not written.\n")
        return

    merged = {**state, **delta}
    # Strip non-serialisable keys (DataFrames, etc.)
    serialisable = {}
    for k, v in merged.items():
        try:
            json.dumps(v, default=str)
            serialisable[k] = v
        except Exception:
            log.debug("run_node | Skipping non-serialisable key %r", k)

    out_path = Path(args.out) if args.out else Path("dev/state") / f"after_{step_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
    print(f"[run_node] State saved → {out_path}\n")


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available pipeline steps:")
        for name in STEP_REGISTRY:
            print(f"  {name}")
        return

    if args.step is not None:
        _run_once(args.step, args)
        return

    # Interactive mode
    print("\n[run_node] Interactive mode — no step specified.\n")
    steps = list(STEP_REGISTRY)
    print("Available steps:")
    for i, name in enumerate(steps, 1):
        print(f"  {i}. {name}")

    while True:
        raw = input("\nStep to run (name or number): ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(steps):
                args.step = steps[idx]
                break
        elif raw in STEP_REGISTRY:
            args.step = raw
            break
        print(f"  Invalid: {raw!r}")

    if not args.state_file:
        dev_dir = Path("dev/state")
        fixtures = sorted(dev_dir.glob("after_*.json")) if dev_dir.exists() else []
        if fixtures:
            print("\nAvailable state files:")
            for i, f in enumerate(fixtures, 1):
                print(f"  {i}. {f}")
            print("  0. Empty state")
            raw = input("State file (number or path, Enter for empty): ").strip()
            if raw and raw != "0":
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(fixtures):
                        args.state_file = str(fixtures[idx])
                else:
                    args.state_file = raw

    if not args.direction:
        raw = input("\nResearch direction (or Enter to skip): ").strip()
        if raw:
            args.direction = raw

    if not args.profile:
        raw = input("Profile name (or Enter to auto-select): ").strip()
        if raw:
            args.profile = raw

    _run_once(args.step, args)

    while input("Run another step? (y/N): ").strip().lower() in ("y", "yes"):
        main()


if __name__ == "__main__":
    main()
