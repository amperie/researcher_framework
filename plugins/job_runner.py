"""Durable local job runner for long-running plugin tasks.

The runner stores each job in a directory:

    job.json      immutable task spec
    payload.json  callable payload
    status.json   submitted/running/succeeded/failed metadata
    result.json   task return value on success
    stdout.log    worker stdout
    stderr.log    worker stderr and exception traces

Only the local-process runner is implemented now. The public shape is small so
future runners such as Ray can implement the same submit/check methods.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from plugins.task_runner import load_callable


TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}


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
        stdout = (job_dir / "stdout.log").open("a", encoding="utf-8")
        stderr = (job_dir / "stderr.log").open("a", encoding="utf-8")
        subprocess.Popen(
            cmd,
            cwd=job_spec.get("cwd") or None,
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


def get_runner(name: str | None = None) -> JobRunner:
    runner = name or "local_process"
    if runner == "local_process":
        return LocalProcessRunner()
    raise ValueError(f"Unknown job runner {runner!r}")


def run_job(job_dir: str) -> None:
    """Worker entry point. Runs one job and writes durable status/result files."""
    root = Path(job_dir).resolve()
    spec = _read_json(root / "job.json")
    payload = _read_json(root / "payload.json")
    job_id = spec["job_id"]

    _write_json(root / "status.json", _status(job_id, "running", spec))
    try:
        task = load_callable(spec["task_path"])
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


def _worker_command(spec: dict[str, Any], job_dir: Path) -> list[str]:
    python = spec.get("python") or sys.executable
    parts = str(python).split()
    return parts + ["-u", "-m", "plugins.job_runner", "run", str(job_dir)]


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
