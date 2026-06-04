"""Persist memory records from a saved pipeline state JSON."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph.nodes.memory import build_memory_records_for_state
from core.memory import MemoryService
from core.utils.profile_loader import load_profile


def main() -> int:
    args = _parse_args()
    profile = load_profile(args.profile)
    state_path = args.state_path or ROOT / "dev" / "state" / args.profile / "after_propose_next_steps.json"
    state = _load_state(state_path)
    records = build_memory_records_for_state(profile, state)
    if args.object_type:
        wanted = set(args.object_type)
        records = [record for record in records if str(record.get("object_type") or "") in wanted]

    print(f"State: {state_path}")
    print(f"Records: {len(records)} {dict(Counter(str(r.get('object_type') or 'unknown') for r in records))}")
    for record in records[: args.show]:
        print(f"- {record.get('record_id')} | {record.get('object_type')} | {record.get('title')}")
    if len(records) > args.show:
        print(f"... {len(records) - args.show} more")

    if args.dry_run:
        print("Dry run: no memory writes performed.")
        return 0
    if not records:
        print("No records to persist.")
        return 1

    MemoryService.for_profile(profile).persist_records(records)
    print(f"Persisted {len(records)} memory record(s).")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="trading", help="Profile name. Defaults to trading.")
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Saved state JSON. Defaults to dev/state/<profile>/after_propose_next_steps.json.",
    )
    parser.add_argument(
        "--object-type",
        action="append",
        help="Only persist this memory object_type. Repeatable, e.g. --object-type next_step.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print records without writing memory.")
    parser.add_argument("--show", type=int, default=20, help="Number of record IDs to print.")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        state = json.load(fh)
    if not isinstance(state, dict):
        raise ValueError(f"State file must contain a JSON object: {path}")
    return state


if __name__ == "__main__":
    raise SystemExit(main())
