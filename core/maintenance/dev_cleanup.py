"""Periodic cleanup for disposable files under the configured dev root."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from configs.config import get_config
from core.utils.logger import get_logger

log = get_logger(__name__)

_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}


@dataclass
class CleanupSummary:
    deleted_files: list[str] = field(default_factory=list)
    deleted_dirs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""


def run_periodic_dev_cleanup(*, now: datetime | None = None) -> CleanupSummary:
    """Run cleanup if the configured interval has elapsed."""
    cfg = get_config()
    cleanup_cfg = _cleanup_cfg(cfg)
    summary = CleanupSummary()

    if not cleanup_cfg.get("enabled", True):
        summary.skipped = True
        summary.reason = "disabled"
        return summary

    current = _as_utc(now)
    state_path = Path(cfg.dev_root) / ".maintenance" / "dev_cleanup_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _read_state_file(state_path)

    last_run = _parse_dt(state.get("last_run_at"))
    interval_hours = int(cleanup_cfg.get("interval_hours", 12) or 12)
    if last_run and current - last_run < timedelta(hours=interval_hours):
        summary.skipped = True
        summary.reason = "interval_not_elapsed"
        return summary

    summary = cleanup_dev_workspace(now=current)
    state_path.write_text(
        json.dumps({"last_run_at": current.isoformat()}, indent=2),
        encoding="utf-8",
    )
    return summary


def cleanup_dev_workspace(*, now: datetime | None = None) -> CleanupSummary:
    """Delete stale disposable files under the configured dev root."""
    cfg = get_config()
    cleanup_cfg = _cleanup_cfg(cfg)
    current = _as_utc(now)
    summary = CleanupSummary()

    dev_root = Path(cfg.dev_root).resolve()
    state_dir = Path(cfg.dev_root) / "state"
    papers_dir = Path(cfg.dev_root) / "papers"
    experiments_dir = Path(cfg.experiments_dir)

    _cleanup_state_snapshots(
        state_dir=state_dir,
        keep_latest=int(cleanup_cfg.get("state_keep_latest", 8) or 8),
        max_age_days=int(cleanup_cfg.get("state_max_age_days", 14) or 14),
        dev_root=dev_root,
        now=current,
        summary=summary,
    )
    _cleanup_old_files(
        base_dir=papers_dir,
        pattern="*.digest",
        max_age_days=int(cleanup_cfg.get("papers_max_age_days", 30) or 30),
        dev_root=dev_root,
        now=current,
        summary=summary,
    )
    _cleanup_generated_tests(
        experiments_dir=experiments_dir,
        max_age_days=int(cleanup_cfg.get("generated_tests_max_age_days", 14) or 14),
        dev_root=dev_root,
        now=current,
        summary=summary,
    )
    _cleanup_terminal_job_dirs(
        experiments_dir=experiments_dir,
        max_age_days=int(cleanup_cfg.get("terminal_jobs_max_age_days", 7) or 7),
        dev_root=dev_root,
        now=current,
        summary=summary,
    )
    return summary


def _cleanup_cfg(cfg: Any) -> dict[str, Any]:
    maintenance = getattr(cfg, "maintenance", None) or {}
    if not isinstance(maintenance, dict):
        return {}
    cleanup = maintenance.get("dev_cleanup") or {}
    return cleanup if isinstance(cleanup, dict) else {}


def _cleanup_state_snapshots(
    *,
    state_dir: Path,
    keep_latest: int,
    max_age_days: int,
    dev_root: Path,
    now: datetime,
    summary: CleanupSummary,
) -> None:
    if not state_dir.exists():
        return
    files = sorted(
        (path for path in state_dir.rglob("after_*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    to_keep = {path.resolve() for path in files[: max(0, keep_latest)]}
    cutoff = now - timedelta(days=max_age_days)
    for path in files[max(0, keep_latest):]:
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError as exc:
            summary.errors.append(f"state:{path} stat failed: {exc}")
            continue
        if path.resolve() in to_keep or modified >= cutoff:
            continue
        _delete_file(path, dev_root=dev_root, summary=summary)


def _cleanup_old_files(
    *,
    base_dir: Path,
    pattern: str,
    max_age_days: int,
    dev_root: Path,
    now: datetime,
    summary: CleanupSummary,
) -> None:
    if not base_dir.exists():
        return
    cutoff = now - timedelta(days=max_age_days)
    for path in base_dir.glob(pattern):
        if not path.is_file():
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError as exc:
            summary.errors.append(f"file:{path} stat failed: {exc}")
            continue
        if modified < cutoff:
            _delete_file(path, dev_root=dev_root, summary=summary)


def _cleanup_generated_tests(
    *,
    experiments_dir: Path,
    max_age_days: int,
    dev_root: Path,
    now: datetime,
    summary: CleanupSummary,
) -> None:
    if not experiments_dir.exists():
        return
    cutoff = now - timedelta(days=max_age_days)
    for tests_dir in experiments_dir.glob("*/tests"):
        if not tests_dir.is_dir():
            continue
        for path in tests_dir.glob("test_*.py"):
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError as exc:
                summary.errors.append(f"tests:{path} stat failed: {exc}")
                continue
            if modified < cutoff:
                _delete_file(path, dev_root=dev_root, summary=summary)
        _delete_empty_dir(tests_dir, dev_root=dev_root, summary=summary)


def _cleanup_terminal_job_dirs(
    *,
    experiments_dir: Path,
    max_age_days: int,
    dev_root: Path,
    now: datetime,
    summary: CleanupSummary,
) -> None:
    if not experiments_dir.exists():
        return
    cutoff = now - timedelta(days=max_age_days)
    for jobs_dir in experiments_dir.glob("*/jobs"):
        if not jobs_dir.is_dir():
            continue
        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            status = _read_job_status(job_dir / "status.json")
            if not status:
                continue
            if status.get("status") not in _TERMINAL_JOB_STATUSES:
                continue
            updated_at = _parse_dt(status.get("updated_at"))
            if updated_at is None or updated_at >= cutoff:
                continue
            _delete_dir(job_dir, dev_root=dev_root, summary=summary)
        _delete_empty_dir(jobs_dir, dev_root=dev_root, summary=summary)


def _read_state_file(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("dev_cleanup | failed to read state file %s: %s", path, exc)
    return {}


def _read_job_status(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception as exc:
        log.warning("dev_cleanup | failed to read job status %s: %s", path, exc)
    return None


def _parse_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _delete_file(path: Path, *, dev_root: Path, summary: CleanupSummary) -> None:
    if not _is_within(path, dev_root):
        summary.errors.append(f"refused to delete outside dev_root: {path}")
        return
    try:
        path.unlink(missing_ok=True)
        summary.deleted_files.append(str(path))
    except Exception as exc:
        summary.errors.append(f"delete file failed {path}: {exc}")


def _delete_dir(path: Path, *, dev_root: Path, summary: CleanupSummary) -> None:
    if not _is_within(path, dev_root):
        summary.errors.append(f"refused to delete outside dev_root: {path}")
        return
    try:
        shutil.rmtree(path)
        summary.deleted_dirs.append(str(path))
    except Exception as exc:
        summary.errors.append(f"delete dir failed {path}: {exc}")


def _delete_empty_dir(path: Path, *, dev_root: Path, summary: CleanupSummary) -> None:
    if not _is_within(path, dev_root):
        summary.errors.append(f"refused to delete outside dev_root: {path}")
        return
    try:
        next(path.iterdir())
    except StopIteration:
        try:
            path.rmdir()
            summary.deleted_dirs.append(str(path))
        except Exception as exc:
            summary.errors.append(f"delete empty dir failed {path}: {exc}")
    except FileNotFoundError:
        return
    except Exception as exc:
        summary.errors.append(f"scan dir failed {path}: {exc}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False
