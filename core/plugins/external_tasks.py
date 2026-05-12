"""Shared subprocess bridge for plugin task execution.

Adapters provide the task path, payload, interpreter, cwd, and any extra
PYTHONPATH/env requirements. This module owns the generic mechanics of invoking
``core.plugins.task_runner`` and decoding its JSON response.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any


def _default_log_config_path() -> str:
    return str((Path(__file__).resolve().parents[2] / "configs" / "config.yaml").resolve())


def build_pythonpath(
    entries: list[str | os.PathLike[str]],
    *,
    existing_pythonpath: str = "",
) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        value = str(Path(entry).resolve())
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    if existing_pythonpath:
        for value in existing_pythonpath.split(os.pathsep):
            stripped = value.strip()
            if stripped and stripped not in seen:
                values.append(stripped)
                seen.add(stripped)
    return os.pathsep.join(values)


def call_external_task(
    task_path: str,
    payload: dict[str, Any],
    *,
    python: str,
    timeout: int,
    plugin_name: str,
    logger_prefixes: list[str],
    cwd: str | None = None,
    pythonpath_entries: list[str | os.PathLike[str]] | None = None,
    env_overrides: dict[str, str] | None = None,
    log=None,
) -> dict[str, Any]:
    env = os.environ.copy()
    pythonpath = build_pythonpath(
        list(pythonpath_entries or []),
        existing_pythonpath=env.get("PYTHONPATH", ""),
    )
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    env["RESEARCH_PLUGIN_LOG"] = plugin_name
    env["RESEARCH_PLUGIN_LOGGERS"] = ",".join(logger_prefixes)
    env.setdefault("RESEARCH_LOG_CONFIG", _default_log_config_path())
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    for key, value in (env_overrides or {}).items():
        env[str(key)] = str(value)

    cmd = str(python).split() + ["-u", "-m", "core.plugins.task_runner", task_path]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=cwd,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _read_stdout() -> None:
        if not proc.stdout:
            return
        for line in iter(proc.stdout.readline, ""):
            stdout_chunks.append(line)
            stripped = line.rstrip("\n")
            if stripped and log is not None and not _looks_like_json_line(stripped):
                log.info("[%s bridge] %s", plugin_name, stripped)

    def _relay_stderr() -> None:
        if not proc.stderr:
            return
        for line in iter(proc.stderr.readline, ""):
            stripped = line.rstrip("\n")
            stderr_chunks.append(line)
            if stripped and log is not None:
                log.info("[%s bridge] %s", plugin_name, stripped)

    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err = threading.Thread(target=_relay_stderr, daemon=True)
    t_out.start()
    t_err.start()

    if proc.stdin:
        try:
            proc.stdin.write(json.dumps(payload, default=str))
            proc.stdin.close()
        except BrokenPipeError:
            pass

    start = time.monotonic()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        raise RuntimeError(
            f"Task {task_path!r} timed out after {timeout} seconds"
        ) from exc

    elapsed = time.monotonic() - start
    remaining = max(0.0, timeout - elapsed)
    t_out.join(timeout=remaining or 2)
    t_err.join(timeout=2)

    stdout_data = "".join(stdout_chunks)
    stderr_data = "".join(stderr_chunks)
    if proc.returncode != 0:
        json_error = _extract_json_error(stdout_data)
        detail = json_error or _tail_text(stdout_data, limit=1200) or _tail_text(stderr_data, limit=1200)
        raise RuntimeError(f"Task {task_path!r} exited {proc.returncode}: {detail}")

    json_line = next(
        (line for line in reversed(stdout_data.splitlines()) if line.lstrip().startswith(("{", "["))),
        None,
    )
    if json_line is None:
        raise RuntimeError(f"Task {task_path!r} produced no JSON output")

    result = json.loads(json_line)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Task {task_path!r} error: {result['error']}")
    return result


def _extract_json_error(stdout_data: str) -> str:
    for line in reversed(stdout_data.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    return ""


def _tail_text(value: str, *, limit: int) -> str:
    text = value.strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _looks_like_json_line(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith(("{", "[")) and stripped.endswith(("}", "]"))
