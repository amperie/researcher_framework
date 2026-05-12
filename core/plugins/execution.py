"""Framework-owned external task execution API.

Adapters provide task specs. The framework decides whether to run them inline,
through the durable local-process runner, or through Ray.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from configs.config import get_config
from core.plugins.external_tasks import build_pythonpath, call_external_task
from core.plugins.job_runner import TERMINAL_STATUSES, get_runner
from core.utils.logger import get_logger


log = get_logger(__name__)


def runner_name(profile: dict[str, Any], purpose: str, default: str = "sync") -> str:
    execution = profile.get("execution") or {}
    raw = str(
        execution.get(f"{purpose}_runner")
        or execution.get("runner")
        or default
    ).strip().lower()
    alias_map = {
        "sync": "sync",
        "inline": "sync",
        "subprocess": "sync",
        "async": "local_process",
        "local_process": "local_process",
        "local-process": "local_process",
        "ray": "ray",
    }
    return alias_map.get(raw, default)


def run_task(task_spec: dict[str, Any], profile: dict[str, Any], purpose: str, *, default_runner: str = "sync") -> dict[str, Any]:
    mode = runner_name(profile, purpose, default=default_runner)
    if mode == "sync":
        return call_external_task(
            str(task_spec["task_path"]),
            dict(task_spec.get("payload") or {}),
            python=str(task_spec["python"]),
            timeout=int(task_spec.get("timeout") or get_config().experiment_timeout_seconds),
            plugin_name=str(task_spec.get("plugin_name") or profile.get("name") or "framework"),
            logger_prefixes=list(task_spec.get("logger_prefixes") or []),
            cwd=task_spec.get("cwd"),
            pythonpath_entries=list(task_spec.get("pythonpath_entries") or []),
            env_overrides=dict(task_spec.get("env") or {}),
            log=log,
        )

    job = submit_task(task_spec, profile, purpose, default_runner=default_runner)
    timeout = int(task_spec.get("timeout") or get_config().experiment_timeout_seconds)
    poll_seconds = float(((profile.get("execution") or {}).get("poll_interval_seconds")) or 1)
    deadline = time.monotonic() + timeout + 5
    while True:
        checked = check_task(job, profile, purpose, default_runner=default_runner)
        if checked.get("status") in TERMINAL_STATUSES:
            if checked.get("status") != "succeeded":
                raise RuntimeError(str(checked.get("error") or f"Task failed with status {checked.get('status')}"))
            return read_task_result(checked)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Task {task_spec.get('task_path')!r} timed out after {timeout} seconds")
        time.sleep(poll_seconds)


def submit_task(task_spec: dict[str, Any], profile: dict[str, Any], purpose: str, *, default_runner: str = "local_process") -> dict[str, Any]:
    mode = runner_name(profile, purpose, default=default_runner)
    job_spec = _job_spec(task_spec, mode)
    if mode == "sync":
        return _run_sync_job(job_spec)
    runner = get_runner(mode)
    return runner.submit(job_spec)


def check_task(job: dict[str, Any], profile: dict[str, Any], purpose: str, *, default_runner: str = "local_process") -> dict[str, Any]:
    mode = str(job.get("runner") or runner_name(profile, purpose, default=default_runner))
    if mode == "sync":
        return job
    runner = get_runner(mode)
    return runner.check(job)


def read_task_result(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job["result_path"]))
    return json.loads(path.read_text(encoding="utf-8"))


def _job_spec(task_spec: dict[str, Any], mode: str) -> dict[str, Any]:
    spec = dict(task_spec)
    env = dict(spec.get("env") or {})
    pythonpath = build_pythonpath(
        list(spec.get("pythonpath_entries") or []),
        existing_pythonpath=str(env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")),
    )
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    if spec.get("plugin_name"):
        env.setdefault("RESEARCH_PLUGIN_LOG", str(spec["plugin_name"]))
    if spec.get("logger_prefixes"):
        env.setdefault("RESEARCH_PLUGIN_LOGGERS", ",".join(spec["logger_prefixes"]))
    spec["env"] = env
    spec["runner"] = mode
    return spec


def _run_sync_job(job_spec: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(str(job_spec["job_dir"])).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(job_spec.get("payload") or {})
    job = {
        "job_id": job_spec["job_id"],
        "job_dir": str(job_dir),
        "runner": "sync",
        "status": "running",
        "task_path": job_spec.get("task_path"),
        "proposal_name": job_spec.get("proposal_name"),
        "stage": job_spec.get("stage"),
    }
    _write_json(job_dir / "job.json", {**job_spec, "payload": None})
    _write_json(job_dir / "payload.json", payload)
    _write_json(job_dir / "status.json", job)
    try:
        result = call_external_task(
            str(job_spec["task_path"]),
            payload,
            python=str(job_spec["python"]),
            timeout=int(job_spec.get("timeout") or get_config().experiment_timeout_seconds),
            plugin_name=str(job_spec.get("plugin_name") or "framework"),
            logger_prefixes=list(job_spec.get("logger_prefixes") or []),
            cwd=job_spec.get("cwd"),
            pythonpath_entries=list(job_spec.get("pythonpath_entries") or []),
            env_overrides=dict(job_spec.get("env") or {}),
            log=log,
        )
        result_path = job_dir / "result.json"
        _write_json(result_path, result)
        job.update({
            "status": "succeeded",
            "result_path": str(result_path),
            "stdout_path": str(job_dir / "stdout.log"),
            "stderr_path": str(job_dir / "stderr.log"),
        })
    except Exception as exc:
        job.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_path": str(job_dir / "stdout.log"),
            "stderr_path": str(job_dir / "stderr.log"),
        })
    _write_json(job_dir / "status.json", job)
    return job


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
