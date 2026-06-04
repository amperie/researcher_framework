"""Durable local job runner for long-running plugin tasks.

The runner stores each job in a directory:

    job.json      immutable task spec
    payload.json  callable payload
    status.json   submitted/running/succeeded/failed metadata
    result.json   task return value on success
    stdout.log    worker stdout
    stderr.log    worker stderr and exception traces

The public shape is small so different backends can implement the same
submit/check methods. This module currently provides local-process and Ray
runners.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from configs.config import get_config
from core.plugins.external_tasks import call_external_task
from core.plugins.task_runner import load_callable
from core.utils.logger import get_logger, setup_plugin_file_logging, setup_logging


TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}
log = get_logger("core.plugins.job_runner")
_JOB_STORE_ACTOR_NAME = "researcher_framework_job_store"
# NeuralSignalResearcher project root — always use this as the worker cwd so that
# configs/config.yaml is resolvable and the framework's Python packages are importable.
_NSR_ROOT = Path(__file__).resolve().parents[2]
_LOG_CONFIG_PATH = str((_NSR_ROOT / "configs" / "config.yaml").resolve())


class JobRunner(Protocol):
    def submit(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Submit a task and return durable job metadata."""
        ...

    def check(self, job: dict[str, Any]) -> dict[str, Any]:
        """Return the latest durable job metadata."""
        ...


class LocalProcessRunner:
    """Submit dotted Python callables to detached local worker processes."""

    runner_name = "local_process"

    def submit(self, spec: dict[str, Any]) -> dict[str, Any]:
        job_id = spec.get("job_id") or str(uuid4())
        job_dir = Path(spec["job_dir"]).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        job_spec = {**spec, "job_id": job_id, "job_dir": str(job_dir), "runner": self.runner_name}
        payload = job_spec.pop("payload")

        _write_json(job_dir / "job.json", job_spec)
        _write_json(job_dir / "payload.json", payload)
        _write_json(job_dir / "status.json", _status(job_id, "submitted", job_spec))

        cmd = _worker_command(job_spec, job_dir)
        env = os.environ.copy()
        env.update(job_spec.get("env") or {})
        env.setdefault("RESEARCH_LOG_CONFIG", _LOG_CONFIG_PATH)
        stdout = (job_dir / "stdout.log").open("a", encoding="utf-8")
        stderr = (job_dir / "stderr.log").open("a", encoding="utf-8")
        subprocess.Popen(
            cmd,
            cwd=str(_NSR_ROOT),  # NSR root so configs/config.yaml and core.* modules resolve
            env=env,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            close_fds=False,
        )
        stdout.close()
        stderr.close()

        return self.check({"job_id": job_id, "job_dir": str(job_dir)})

    def check(self, job: dict[str, Any]) -> dict[str, Any]:
        return _read_durable_status(job)


class RayRunner:
    """Submit jobs to Ray tasks while keeping the same durable on-disk contract."""

    runner_name = "ray"

    def __init__(self) -> None:
        self._cfg = get_config()

    def submit(self, spec: dict[str, Any]) -> dict[str, Any]:
        ray = _ensure_ray(self._cfg)
        job_id = spec.get("job_id") or str(uuid4())
        job_dir = Path(spec["job_dir"]).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        job_spec = {
            **spec,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "runner": self.runner_name,
            "ray_namespace": getattr(self._cfg, "ray_namespace", None),
        }
        payload = job_spec.pop("payload")

        _write_json(job_dir / "job.json", job_spec)
        _write_json(job_dir / "payload.json", payload)
        submitted = _status(job_id, "submitted", job_spec)
        _write_json(job_dir / "status.json", submitted)

        store = _get_or_create_job_store(ray, self._cfg)
        ray.get(store.upsert.remote(job_id, submitted))
        task = ray.remote(_ray_execute_task).options(runtime_env=_ray_runtime_env(job_spec, self._cfg))
        task.remote(job_spec, payload, _job_store_actor_name(self._cfg))
        return self.check({"job_id": job_id, "job_dir": str(job_dir)})

    def check(self, job: dict[str, Any]) -> dict[str, Any]:
        status = _read_durable_status(job)
        try:
            ray = _ensure_ray(self._cfg)
            store = _get_existing_job_store(ray, self._cfg)
            if store is None:
                return status
            remote = ray.get(store.get.remote(str(job.get("job_id") or "")))
        except Exception:
            return status
        if not isinstance(remote, dict) or not remote:
            return status
        _sync_ray_status(Path(job["job_dir"]), remote)
        return _read_durable_status(job)


def get_runner(name: str | None = None) -> JobRunner:
    runner = name or "local_process"
    if runner == "local_process":
        return LocalProcessRunner()
    if runner == "ray":
        return RayRunner()
    raise ValueError(f"Unknown job runner {runner!r}")


def run_job(job_dir: str) -> None:
    """Worker entry point. Runs one job and writes durable status/result files."""
    setup_logging(os.environ.get("RESEARCH_LOG_CONFIG") or _LOG_CONFIG_PATH)
    root = Path(job_dir).resolve()
    spec = _read_json(root / "job.json")
    payload = _read_json(root / "payload.json")
    job_id = spec["job_id"]
    plugin_name = str(spec.get("plugin_name") or "")
    if plugin_name:
        setup_plugin_file_logging(
            plugin_name,
            logger_prefixes=[
                "core.plugins.neuralsignal",
                "core.plugins.task_runner",
                "core.plugins.job_runner",
            ],
        )

    _write_json(root / "status.json", _status(job_id, "running", spec))
    try:
        log.info("Running plugin job id=%s task=%s plugin=%s", job_id, spec.get("task_path"), plugin_name or "unknown")
        task = load_callable(spec["task_path"])
        if spec.get("cwd"):
            os.chdir(str(spec["cwd"]))
        result = task(payload)
        _write_json(root / "result.json", result)
        _write_json(root / "status.json", _status(job_id, "succeeded", spec))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with (root / "stderr.log").open("a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
            fh.write("\n")
        _write_json(root / "status.json", _status(job_id, "failed", spec, error=error))
        raise


def _ray_run_job(spec: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(spec["job_dir"]).resolve()
    cmd = _worker_command(spec, job_dir)
    env = os.environ.copy()
    env.update(spec.get("env") or {})
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"

    try:
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            proc = subprocess.run(
                cmd,
                cwd=spec.get("cwd") or None,
                env=env,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                text=True,
                check=False,
            )
    except Exception as exc:
        failed = _status(spec["job_id"], "failed", spec, error=f"{type(exc).__name__}: {exc}")
        _write_json(job_dir / "status.json", failed)
        return failed

    return {
        "job_id": spec["job_id"],
        "job_dir": str(job_dir),
        "returncode": proc.returncode,
    }


class _ResearchJobStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(self, job_id: str, record: dict[str, Any]) -> bool:
        self._records[str(job_id)] = dict(record)
        return True

    def get(self, job_id: str) -> dict[str, Any] | None:
        record = self._records.get(str(job_id))
        return dict(record) if isinstance(record, dict) else None


def _ray_execute_task(spec: dict[str, Any], payload: dict[str, Any], actor_name: str) -> dict[str, Any]:
    ray = importlib.import_module("ray")
    actor = ray.get_actor(actor_name, namespace=spec.get("ray_namespace") or None)
    running = _status(spec["job_id"], "running", spec)
    ray.get(actor.upsert.remote(spec["job_id"], running))
    try:
        repo_root = str(Path(__file__).resolve().parents[2])
        result = call_external_task(
            str(spec["task_path"]),
            dict(payload or {}),
            python=str(spec.get("python") or sys.executable),
            timeout=int(spec.get("timeout") or get_config().experiment_timeout_seconds),
            plugin_name=str(spec.get("plugin_name") or "framework"),
            logger_prefixes=list(spec.get("logger_prefixes") or []),
            cwd=spec.get("cwd"),
            pythonpath_entries=[repo_root, *list(spec.get("pythonpath_entries") or [])],
            env_overrides=dict(spec.get("env") or {}),
            log=log,
        )
        succeeded = _status(spec["job_id"], "succeeded", spec)
        succeeded["result"] = result
        ray.get(actor.upsert.remote(spec["job_id"], succeeded))
        return succeeded
    except Exception as exc:
        failed = _status(spec["job_id"], "failed", spec, error=f"{type(exc).__name__}: {exc}")
        ray.get(actor.upsert.remote(spec["job_id"], failed))
        return failed


def _worker_command(spec: dict[str, Any], job_dir: Path) -> list[str]:
    python = spec.get("python") or sys.executable
    parts = str(python).split()
    # When using `uv run` and the spec carries a plugin cwd, pin the project so the
    # correct venv is used regardless of the subprocess cwd (which is _NSR_ROOT).
    if len(parts) >= 2 and parts[0] == "uv" and parts[1] == "run" and spec.get("cwd"):
        parts = ["uv", "run", "--project", str(spec["cwd"])] + parts[2:]
    return parts + ["-u", "-m", "core.plugins.job_runner", "run", str(job_dir)]


def _read_durable_status(job: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(job["job_dir"])
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return {
            **job,
            "status": "unknown",
            "error": f"Missing status file: {status_path}",
        }
    status = _read_json(status_path)
    result_path = job_dir / "result.json"
    if result_path.exists():
        status["result_path"] = str(result_path)
    status["stdout_path"] = str(job_dir / "stdout.log")
    status["stderr_path"] = str(job_dir / "stderr.log")
    return status


def _sync_ray_status(job_dir: Path, remote_status: dict[str, Any]) -> None:
    _write_json(job_dir / "status.json", remote_status)
    if remote_status.get("status") == "succeeded" and "result" in remote_status:
        _write_json(job_dir / "result.json", remote_status["result"])


def _job_store_actor_name(cfg: Any) -> str:
    namespace = str(getattr(cfg, "ray_namespace", "") or "").strip()
    return f"{_JOB_STORE_ACTOR_NAME}::{namespace or 'default'}"


def _get_existing_job_store(ray: Any, cfg: Any) -> Any | None:
    try:
        return ray.get_actor(
            _job_store_actor_name(cfg),
            namespace=getattr(cfg, "ray_namespace", None) or None,
        )
    except Exception:
        return None


def _get_or_create_job_store(ray: Any, cfg: Any) -> Any:
    actor = _get_existing_job_store(ray, cfg)
    if actor is not None:
        return actor
    actor_cls = ray.remote(_ResearchJobStore)
    return actor_cls.options(
        name=_job_store_actor_name(cfg),
        namespace=getattr(cfg, "ray_namespace", None) or None,
        lifetime="detached",
    ).remote()


def _ensure_ray(cfg: Any) -> Any:
    ray = importlib.import_module("ray")
    if ray.is_initialized():
        log.info(
            "job_runner | Using existing Ray runtime mode=%s address=%s namespace=%s dashboard=%s",
            getattr(cfg, "ray_mode", "local"),
            getattr(cfg, "ray_address", "auto") or "auto",
            getattr(cfg, "ray_namespace", None) or "-",
            _ray_dashboard_url(ray) or "unavailable",
        )
        return ray

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if getattr(cfg, "ray_namespace", None):
        init_kwargs["namespace"] = cfg.ray_namespace
    if getattr(cfg, "ray_mode", "local") == "remote":
        init_kwargs["address"] = getattr(cfg, "ray_address", "auto") or "auto"

    ray.init(**init_kwargs)
    log.info(
        "job_runner | Ray initialized mode=%s address=%s namespace=%s dashboard=%s",
        getattr(cfg, "ray_mode", "local"),
        init_kwargs.get("address", "local"),
        init_kwargs.get("namespace", "-"),
        _ray_dashboard_url(ray) or "unavailable",
    )
    return ray


def _ray_dashboard_url(ray: Any) -> str:
    candidates: list[Any] = []
    try:
        runtime_context = getattr(ray, "get_runtime_context", None)
        if callable(runtime_context):
            candidates.append(runtime_context())
    except Exception:
        pass
    try:
        worker = getattr(getattr(ray, "_private", None), "worker", None)
        candidates.append(getattr(worker, "global_worker", None))
    except Exception:
        pass

    for candidate in candidates:
        for attr in ("dashboard_url", "webui_url"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        node = getattr(candidate, "node", None)
        for attr in ("webui_url", "dashboard_url"):
            value = getattr(node, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        address_info = getattr(node, "address_info", None)
        if isinstance(address_info, dict):
            for key in ("webui_url", "dashboard_url"):
                value = address_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _ray_runtime_env(spec: dict[str, Any], cfg: Any) -> dict[str, Any]:
    runtime_env: dict[str, Any] = {}
    env = spec.get("env") or {}
    if env:
        env_vars = {str(k): str(v) for k, v in env.items() if v is not None}
        if getattr(cfg, "ray_mode", "local") != "local":
            env_vars.pop("PYTHONPATH", None)
        if env_vars:
            runtime_env["env_vars"] = env_vars
    if getattr(cfg, "ray_mode", "local") != "local":
        runtime_env["working_dir"] = str(Path(__file__).resolve().parents[2])
    return runtime_env


def _status(
    job_id: str,
    status: str,
    spec: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "job_id": job_id,
        "job_dir": spec["job_dir"],
        "runner": spec.get("runner", "local_process"),
        "task_path": spec.get("task_path"),
        "status": status,
        "stage": spec.get("stage"),
        "proposal_name": spec.get("proposal_name"),
        "artifact_id": spec.get("artifact_id"),
        "experiment_id": spec.get("experiment_id"),
        "submitted_at": spec.get("submitted_at") or now,
        "updated_at": now,
        "result_path": str(Path(spec["job_dir"]) / "result.json"),
        "stdout_path": str(Path(spec["job_dir"]) / "stdout.log"),
        "stderr_path": str(Path(spec["job_dir"]) / "stderr.log"),
        "error": error,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("job_dir")
    args = parser.parse_args()

    if args.cmd == "run":
        run_job(args.job_dir)


if __name__ == "__main__":
    main()
