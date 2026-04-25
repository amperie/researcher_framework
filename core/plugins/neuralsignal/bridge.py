"""Compatibility wrapper for NeuralSignal subprocess tasks.

The generic runner is ``plugins/task_runner.py``. This file remains only so old
commands such as ``plugins/neuralsignal/bridge.py create_dataset`` continue to
work while callers migrate to dotted task paths.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_TASKS = {
    "create_dataset": "plugins.neuralsignal.tasks.create_dataset",
    "create_s1_model": "plugins.neuralsignal.tasks.create_s1_model",
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _TASKS:
        print(json.dumps({"error": f"Usage: bridge.py <action>  known: {list(_TASKS)}"}))
        sys.exit(1)

    runner = Path(__file__).resolve().parents[1] / "task_runner.py"
    payload = sys.stdin.read()
    proc = subprocess.run(
        [sys.executable, "-u", str(runner), _TASKS[sys.argv[1]]],
        input=payload,
        text=True,
        capture_output=True,
    )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.stdout:
        print(proc.stdout, end="")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

