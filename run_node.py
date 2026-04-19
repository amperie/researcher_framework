"""Dev harness for manually invoking individual pipeline nodes.

CLI usage:
    uv run python run_node.py research --direction "sparse autoencoders in LLMs"
    uv run python run_node.py feature_proposal --state-file dev/state_after_research.json
    uv run python run_node.py ns_experiment_runner --state-file dev/state_after_feature_proposal.json
    uv run python run_node.py research --direction "test" --no-save
    uv run python run_node.py <node> --out /tmp/my_state.json

IDE / debug usage:
    Run with no arguments — the script will prompt interactively for node,
    state file, and research direction, then ask whether to run again after
    each iteration.

State is loaded from --state-file (if given), then --direction is applied on top.
The returned delta is merged into state and saved to dev/state_after_<node>.json
(or --out if specified) unless --no-save is set.

Node list is built dynamically from graph.nodes.__all__ so it always reflects
the current set of exported nodes. Nodes wired into the active graph are shown
with a [*] marker in interactive mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.logger import setup_logging, get_logger
from utils import fmt_value

setup_logging("configs/config.yaml")

# Inject neuralsignal into sys.path before importing nodes — ns_experiment_runner
# attempts the neuralsignal import at module load time.
def _add_neuralsignal_to_path() -> None:
    import sys
    from pathlib import Path
    from configs.config import get_config
    _log = get_logger(__name__)
    ns_path = Path(get_config().neuralsignal_src_path).resolve()
    if ns_path.exists() and str(ns_path) not in sys.path:
        sys.path.insert(0, str(ns_path))
        _log.debug("Added neuralsignal to sys.path: %s", ns_path)
    elif not ns_path.exists():
        _log.warning("neuralsignal_src_path does not exist: %s", ns_path)

_add_neuralsignal_to_path()

import graph.nodes as _nodes_module  # noqa: E402
from graph.nodes import __all__ as _node_exports  # noqa: E402

# ---------------------------------------------------------------------------
# Build NODES dict dynamically from graph.nodes.__all__
# Convention: exported names end in "_node"; strip suffix for the CLI key.
# ---------------------------------------------------------------------------
NODES: dict[str, object] = {
    name.removesuffix("_node"): getattr(_nodes_module, name)
    for name in _node_exports
    if name.endswith("_node") and hasattr(_nodes_module, name)
}

# ---------------------------------------------------------------------------
# Discover which nodes are wired into the active graph (for display only)
# ---------------------------------------------------------------------------
_ACTIVE_NODES: set[str] = set()
try:
    from graph.graph import build_graph as _build_graph  # noqa: E402
    _ACTIVE_NODES = set(_build_graph().nodes) - {"__start__"}
except Exception:
    pass  # non-fatal — just means no [*] markers shown

_SEP = "-" * 60


def _prompt_node() -> str:
    """Interactively pick a node by name or number."""
    node_names = list(NODES)
    print("\nAvailable nodes  ([*] = wired into active graph):")
    for i, name in enumerate(node_names, 1):
        marker = "[*]" if name in _ACTIVE_NODES else "   "
        print(f"  {i}. {marker} {name}")
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


def _run_once(args: argparse.Namespace) -> None:
    """Load state, run the node once, display output, and save."""
    state: dict = {}
    state_source = "empty state"

    if args.state_file:
        path = Path(args.state_file)
        if not path.exists():
            print(f"[run_node] ERROR: state file not found: {path}", file=sys.stderr)
            sys.exit(1)
        state = json.loads(path.read_text(encoding="utf-8"))
        state_source = str(path)

    if args.direction:
        state["research_direction"] = args.direction

    node_fn = NODES[args.node]
    in_graph = "[*] active" if args.node in _ACTIVE_NODES else "not in active graph"
    print(f"\n[run_node] Running: {args.node}  ({in_graph})")
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

    if args.no_save:
        print("[run_node] --no-save: state not written.\n")
        return

    merged = {**state, **delta}
    out_path = Path(args.out) if args.out else Path("dev") / f"state_after_{args.node}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"[run_node] Full state saved -> {out_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single pipeline node for manual testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "node",
        nargs="?",
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
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available nodes and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable nodes  ([*] = wired into active graph):")
        for name in NODES:
            marker = "[*]" if name in _ACTIVE_NODES else "   "
            print(f"  {marker} {name}")
        return

    # Non-interactive (CLI args provided): run once and exit.
    if args.node is not None:
        _run_once(args)
        return

    # Interactive mode: prompt, run, then ask whether to go again.
    print("\n[run_node] No arguments detected -- running in interactive mode.")
    while True:
        args.node = _prompt_node()
        args.state_file = _prompt_state_file()

        # Load state now so we can offer a direction override
        state: dict = {}
        if args.state_file:
            path = Path(args.state_file)
            if not path.exists():
                print(f"[run_node] ERROR: state file not found: {path}", file=sys.stderr)
                continue
            state = json.loads(path.read_text(encoding="utf-8"))

        direction = _prompt_direction(state)
        if direction:
            args.direction = direction
        else:
            args.direction = None

        _run_once(args)

        again = input("Run again? (y/N): ").strip().lower()
        if again not in ("y", "yes"):
            break


if __name__ == "__main__":
    main()
