"""Tests for the durable local job runner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from core.plugins.job_runner import LocalProcessRunner, run_job


def echo_task(payload):
    return {"echo": payload["value"]}


def test_run_job_executes_task_and_writes_result(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps({
            "job_id": "job1",
            "job_dir": str(job_dir),
            "plugin_name": "neuralsignal",
            "task_path": "tests.test_job_runner.echo_task",
            "runner": "local_process",
            "stage": "unit",
        }),
        encoding="utf-8",
    )
    (job_dir / "payload.json").write_text(json.dumps({"value": 42}), encoding="utf-8")

    with patch("core.plugins.job_runner.setup_plugin_file_logging") as setup_plugin_logging:
        run_job(str(job_dir))

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert status["status"] == "succeeded"
    assert result == {"echo": 42}
    setup_plugin_logging.assert_called_once()


def test_local_process_submit_writes_job_files_and_launches_module(tmp_path):
    runner = LocalProcessRunner()
    spec = {
        "job_id": "job1",
        "job_dir": str(tmp_path / "job1"),
        "plugin_name": "neuralsignal",
        "task_path": "tests.test_job_runner.echo_task",
        "payload": {"value": 1},
        "python": "python",
        "cwd": str(tmp_path),
        "env": {
            "PYTHONPATH": "x",
            "RESEARCH_PLUGIN_LOG": "neuralsignal",
            "RESEARCH_PLUGIN_LOGGERS": "core.plugins.neuralsignal,core.plugins.task_runner,core.plugins.job_runner",
        },
        "stage": "unit",
        "proposal_name": "p1",
    }

    with patch("core.plugins.job_runner.subprocess.Popen") as popen:
        job = runner.submit(spec)

    job_dir = Path(spec["job_dir"])
    assert (job_dir / "job.json").exists()
    assert (job_dir / "payload.json").exists()
    assert job["status"] == "submitted"
    cmd = popen.call_args.args[0]
    assert cmd[-4:-1] == ["-m", "plugins.job_runner", "run"]
    assert cmd[-1] == str(job_dir.resolve())
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert popen.call_args.kwargs["env"]["PYTHONPATH"] == "x"
    assert popen.call_args.kwargs["env"]["RESEARCH_PLUGIN_LOG"] == "neuralsignal"

