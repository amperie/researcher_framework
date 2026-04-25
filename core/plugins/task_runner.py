"""Generic subprocess task runner.

This script runs any importable Python callable in a separate process. It is
domain-agnostic: domain-specific code belongs in task modules such as
``plugins.neuralsignal.tasks``.

Usage:
    python plugins/task_runner.py package.module.function

The runner reads JSON from stdin and passes it as the single argument to the
callable. The callable must return a JSON-serializable value. The final stdout
line is JSON so parent processes can parse it even if libraries print logs.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_callable(dotted_path: str) -> Callable[[dict[str, Any]], Any]:
    """Load a callable from ``package.module.function``."""
    module_name, _, attr_name = dotted_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid task path {dotted_path!r}; expected package.module.function")

    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"Task {dotted_path!r} is not callable")
    return fn


def main() -> None:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: task_runner.py <package.module.function>"}))
        sys.exit(1)

    task_path = sys.argv[1]
    log.info("task_runner started - task=%s", task_path)

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON on stdin: {exc}"}))
        sys.exit(1)

    try:
        task = load_callable(task_path)
        result = task(payload)
    except Exception as exc:
        log.exception("task failed")
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()

