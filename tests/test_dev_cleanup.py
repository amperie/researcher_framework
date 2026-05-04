from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.maintenance.dev_cleanup import cleanup_dev_workspace, run_periodic_dev_cleanup


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dev_root=str(tmp_path / "dev"),
        experiments_dir=str(tmp_path / "dev" / "experiments"),
        maintenance={
            "dev_cleanup": {
                "enabled": True,
                "interval_hours": 12,
                "state_keep_latest": 3,
                "state_max_age_days": 14,
                "papers_max_age_days": 30,
                "generated_tests_max_age_days": 14,
                "terminal_jobs_max_age_days": 7,
            }
        },
    )


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _set_mtime(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    import os
    os.utime(path, (ts, ts))


def test_cleanup_keeps_recent_state_but_deletes_stale_papers_and_tests_and_terminal_jobs(tmp_path):
    cfg = _cfg(tmp_path)
    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)

    state_dir = Path(cfg.dev_root) / "state"
    recent1 = state_dir / "after_recent_1.json"
    recent2 = state_dir / "after_recent_2.json"
    old_kept = state_dir / "after_old_kept.json"
    old_deleted = state_dir / "after_old_deleted.json"
    for path in (recent1, recent2, old_kept, old_deleted):
        _write(path, "{}")
    _set_mtime(recent1, now - timedelta(days=1))
    _set_mtime(recent2, now - timedelta(days=2))
    _set_mtime(old_kept, now - timedelta(days=20))
    _set_mtime(old_deleted, now - timedelta(days=21))

    old_paper = Path(cfg.dev_root) / "papers" / "old.digest"
    _write(old_paper, "digest")
    _set_mtime(old_paper, now - timedelta(days=40))

    old_test = Path(cfg.experiments_dir) / "neuralsignal" / "tests" / "test_old.py"
    _write(old_test, "def test_old(): pass")
    _set_mtime(old_test, now - timedelta(days=20))

    job_dir = Path(cfg.experiments_dir) / "neuralsignal" / "jobs" / "job-1"
    _write(
        job_dir / "status.json",
        json.dumps({"status": "succeeded", "updated_at": (now - timedelta(days=10)).isoformat()}),
    )
    _write(job_dir / "stdout.log", "done")

    running_job_dir = Path(cfg.experiments_dir) / "neuralsignal" / "jobs" / "job-2"
    _write(
        running_job_dir / "status.json",
        json.dumps({"status": "running", "updated_at": (now - timedelta(days=10)).isoformat()}),
    )

    with patch("core.maintenance.dev_cleanup.get_config", return_value=cfg):
        summary = cleanup_dev_workspace(now=now)

    assert recent1.exists()
    assert recent2.exists()
    assert old_kept.exists()
    assert not old_deleted.exists()
    assert not old_paper.exists()
    assert not old_test.exists()
    assert not job_dir.exists()
    assert running_job_dir.exists()
    assert summary.deleted_files
    assert summary.deleted_dirs


def test_periodic_cleanup_skips_when_interval_not_elapsed(tmp_path):
    cfg = _cfg(tmp_path)
    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    state_file = Path(cfg.dev_root) / ".maintenance" / "dev_cleanup_state.json"
    _write(state_file, json.dumps({"last_run_at": (now - timedelta(hours=2)).isoformat()}))

    with patch("core.maintenance.dev_cleanup.get_config", return_value=cfg):
        summary = run_periodic_dev_cleanup(now=now)

    assert summary.skipped is True
    assert summary.reason == "interval_not_elapsed"


def test_periodic_cleanup_updates_state_file_after_run(tmp_path):
    cfg = _cfg(tmp_path)
    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)

    with patch("core.maintenance.dev_cleanup.get_config", return_value=cfg):
        summary = run_periodic_dev_cleanup(now=now)

    state_file = Path(cfg.dev_root) / ".maintenance" / "dev_cleanup_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert summary.skipped is False
    assert data["last_run_at"] == now.isoformat()
