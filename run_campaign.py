from __future__ import annotations

import argparse
import json

from core.campaigns import run_campaign
from core.utils.logger import setup_logging

setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batched research campaign from a YAML config.")
    parser.add_argument("config", type=str, help="Path to campaign YAML config.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Validate and print the materialized campaign without executing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_campaign(args.config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
