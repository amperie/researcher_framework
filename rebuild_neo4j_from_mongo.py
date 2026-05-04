"""Reset the configured Neo4j research graph and rebuild it from Mongo records."""
from __future__ import annotations

import argparse
import sys

from core.memory import MemoryService
from core.utils.profile_loader import list_profiles, load_profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete current Neo4j research graph data for one or more profiles and rebuild it from Mongo memory records.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Profile name to rebuild. Repeat to rebuild multiple profiles. Defaults to all profiles.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1_000_000,
        help="Maximum number of Mongo memory records to replay per profile.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not clear Neo4j first; append/re-merge instead.",
    )
    return parser.parse_args()


def _selected_profiles(names: list[str] | None) -> list[str]:
    if names:
        return names
    available = list_profiles()
    if not available:
        raise SystemExit("No profiles found under configs/profiles.")
    return available


def main() -> int:
    args = _parse_args()
    profile_names = _selected_profiles(args.profiles)
    total = 0
    for profile_name in profile_names:
        profile = load_profile(profile_name)
        service = MemoryService.for_profile(profile)
        count = service.rebuild_graph_from_documents(
            {"domain": profile_name},
            limit=max(int(args.limit), 1),
            reset_first=not bool(args.no_reset),
        )
        total += count
        print(f"{profile_name}: rebuilt graph from {count} Mongo memory record(s)")
    print(f"total: rebuilt graph from {total} Mongo memory record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
