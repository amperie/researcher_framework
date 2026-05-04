"""Tests for the durable local job runner."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.plugins.job_runner import LocalProcessRunner, RayRunner, get_runner, run_job


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
    assert cmd[-4:-1] == ["-m", "core.plugins.job_runner", "run"]
    assert cmd[-1] == str(job_dir.resolve())
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert popen.call_args.kwargs["env"]["PYTHONPATH"] == "x"
    assert popen.call_args.kwargs["env"]["RESEARCH_PLUGIN_LOG"] == "neuralsignal"


def test_get_runner_returns_ray_runner():
    with patch("core.plugins.job_runner.get_config", return_value=SimpleNamespace(ray_mode="local", ray_address="auto", ray_namespace="ns")):
        runner = get_runner("ray")

    assert isinstance(runner, RayRunner)


def test_ray_runner_submit_starts_local_ray_and_passes_runtime_env(tmp_path):
    runner = RayRunner.__new__(RayRunner)
    runner._cfg = SimpleNamespace(ray_mode="local", ray_address="auto", ray_namespace="research")
    spec = {
        "job_id": "job-ray-local",
        "job_dir": str(tmp_path / "job-ray-local"),
        "plugin_name": "neuralsignal",
        "task_path": "tests.test_job_runner.echo_task",
        "payload": {"value": 7},
        "cwd": str(tmp_path),
        "env": {
            "PYTHONPATH": "ns_path;repo_path",
            "RESEARCH_PLUGIN_LOG": "neuralsignal",
        },
        "stage": "unit",
        "proposal_name": "p1",
    }
    ray_task = MagicMock()
    ray_options = MagicMock(return_value=ray_task)
    ray_remote = MagicMock()
    ray_remote.options = ray_options
    ray_module = MagicMock()
    ray_module.is_initialized.return_value = False
    ray_module.remote.return_value = ray_remote

    with patch("core.plugins.job_runner.importlib.import_module", return_value=ray_module):
        job = runner.submit(spec)

    ray_module.init.assert_called_once_with(ignore_reinit_error=True, namespace="research")
    runtime_env = ray_options.call_args.kwargs["runtime_env"]
    assert runtime_env["env_vars"]["PYTHONPATH"] == "ns_path;repo_path"
    assert "working_dir" not in runtime_env
    submitted_spec = ray_task.remote.call_args.args[0]
    assert submitted_spec["job_dir"] == str((tmp_path / "job-ray-local").resolve())
    assert submitted_spec["task_path"] == "tests.test_job_runner.echo_task"
    assert job["status"] == "submitted"


def test_ray_runner_submit_connects_to_remote_cluster_when_configured(tmp_path):
    runner = RayRunner.__new__(RayRunner)
    runner._cfg = SimpleNamespace(ray_mode="remote", ray_address="ray://cluster.example:10001", ray_namespace=None)
    spec = {
        "job_id": "job-ray-remote",
        "job_dir": str(tmp_path / "job-ray-remote"),
        "plugin_name": "neuralsignal",
        "task_path": "tests.test_job_runner.echo_task",
        "payload": {"value": 9},
        "cwd": str(tmp_path),
        "env": {},
        "stage": "unit",
        "proposal_name": "p2",
    }
    ray_task = MagicMock()
    ray_remote = MagicMock()
    ray_remote.options.return_value = ray_task
    ray_module = MagicMock()
    ray_module.is_initialized.return_value = False
    ray_module.remote.return_value = ray_remote

    with patch("core.plugins.job_runner.importlib.import_module", return_value=ray_module):
        runner.submit(spec)

    ray_module.init.assert_called_once_with(ignore_reinit_error=True, address="ray://cluster.example:10001")


def test_ray_runner_remote_mode_drops_pythonpath_from_runtime_env(tmp_path):
    runner = RayRunner.__new__(RayRunner)
    runner._cfg = SimpleNamespace(ray_mode="remote", ray_address="ray://cluster.example:10001", ray_namespace=None)
    spec = {
        "job_id": "job-ray-remote-env",
        "job_dir": str(tmp_path / "job-ray-remote-env"),
        "plugin_name": "neuralsignal",
        "task_path": "tests.test_job_runner.echo_task",
        "payload": {"value": 9},
        "cwd": str(tmp_path),
        "env": {
            "PYTHONPATH": "local_only_path",
            "RESEARCH_PLUGIN_LOG": "neuralsignal",
        },
        "stage": "unit",
        "proposal_name": "p3",
    }
    ray_task = MagicMock()
    ray_options = MagicMock(return_value=ray_task)
    ray_remote = MagicMock()
    ray_remote.options = ray_options
    ray_module = MagicMock()
    ray_module.is_initialized.return_value = False
    ray_module.remote.return_value = ray_remote

    with patch("core.plugins.job_runner.importlib.import_module", return_value=ray_module):
        runner.submit(spec)

    runtime_env = ray_options.call_args.kwargs["runtime_env"]
    assert "PYTHONPATH" not in runtime_env["env_vars"]
    assert runtime_env["env_vars"]["RESEARCH_PLUGIN_LOG"] == "neuralsignal"

