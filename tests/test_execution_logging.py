from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.plugins.execution import run_task
from core.plugins.job_runner import _ray_execute_task


def test_run_task_sync_passes_logger_to_external_bridge(monkeypatch):
    captured = {}

    def fake_call_external_task(*args, **kwargs):
        captured["log"] = kwargs.get("log")
        return {"ok": True}

    monkeypatch.setattr("core.plugins.execution.call_external_task", fake_call_external_task)

    result = run_task(
        {
            "task_path": "pkg.fn",
            "payload": {},
            "python": "python",
            "timeout": 10,
            "plugin_name": "trading",
            "logger_prefixes": ["core.plugins.trading"],
        },
        profile={"name": "trading", "execution": {"runner": "sync"}},
        purpose="execute_experiment",
    )

    assert result == {"ok": True}
    assert captured["log"] is not None


def test_ray_execute_task_passes_logger_to_external_bridge(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_call_external_task(*args, **kwargs):
        captured["log"] = kwargs.get("log")
        return {"ok": True}

    class FakeActor:
        class _UpsertRemote:
            def remote(self, *args, **kwargs):
                return True

        upsert = _UpsertRemote()

    fake_ray = SimpleNamespace(
        get_actor=lambda *args, **kwargs: FakeActor(),
        get=lambda value: value,
    )

    monkeypatch.setattr("core.plugins.job_runner.importlib.import_module", lambda name: fake_ray if name == "ray" else None)
    monkeypatch.setattr("core.plugins.job_runner.call_external_task", fake_call_external_task)

    spec = {
        "job_id": "job-1",
        "job_dir": str(tmp_path),
        "task_path": "pkg.fn",
        "python": "python",
        "timeout": 10,
        "plugin_name": "trading",
        "logger_prefixes": ["core.plugins.trading"],
        "ray_namespace": "ns",
        "env": {},
    }

    result = _ray_execute_task(spec, {}, "actor")

    assert result["status"] == "succeeded"
    assert captured["log"] is not None
