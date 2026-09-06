"""Generic subprocess task runner.

This script runs any importable Python callable in a separate process. It is
domain-agnostic: domain-specific code belongs in task modules such as
``plugins.trading_researcher.tasks``.

Usage:
    python plugins/task_runner.py package.module.function

The runner reads JSON from stdin and passes it as the single argument to the
callable. The callable must return a JSON-serializable value. The final stdout
line is JSON so parent processes can parse it even if libraries print logs.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from core.utils.logger import get_logger, setup_logging, setup_plugin_file_logging

log = get_logger("core.plugins.task_runner")


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
    log_config_path = str(os.environ.get("RESEARCH_LOG_CONFIG") or "configs/config.yaml").strip() or "configs/config.yaml"
    setup_logging(log_config_path)
    plugin_name = str(os.environ.get("RESEARCH_PLUGIN_LOG") or "").strip()
    plugin_loggers = [
        item.strip()
        for item in str(os.environ.get("RESEARCH_PLUGIN_LOGGERS") or "").split(",")
        if item.strip()
    ]
    if plugin_name and plugin_loggers:
        setup_plugin_file_logging(plugin_name, logger_prefixes=plugin_loggers, config_path=log_config_path)

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
