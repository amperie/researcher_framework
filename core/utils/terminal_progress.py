"""Single-line terminal progress for long pipeline runs."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from typing import Any

PROGRESS_PREFIX = "__RF_PROGRESS__ "

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "enabled": True,
    "stage": "",
    "stage_index": 0,
    "stage_total": 0,
    "stage_done": 0,
    "hpo_done": 0,
    "hpo_running": 0,
    "hpo_total": 0,
    "message": "",
}


def configure_pipeline(steps: list[str]) -> None:
    with _LOCK:
        _STATE.update({
            "stage_total": len(steps),
            "stage_index": 0,
            "stage_done": 0,
            "stage": steps[0] if steps else "",
            "message": "",
        })
    render()


def start_stage(stage: str) -> None:
    with _LOCK:
        total = int(_STATE.get("stage_total") or 0)
        _STATE["stage"] = stage
        if total:
            _STATE["stage_index"] = min(total, int(_STATE.get("stage_done") or 0) + 1)
        _STATE["message"] = "running"
    render()


def finish_stage(stage: str) -> None:
    with _LOCK:
        _STATE["stage"] = stage
        _STATE["stage_done"] = max(int(_STATE.get("stage_done") or 0), int(_STATE.get("stage_index") or 0))
        _STATE["message"] = "done"
    render()


def configure_hpo(total: int) -> None:
    with _LOCK:
        _STATE.update({"hpo_done": 0, "hpo_running": 0, "hpo_total": max(0, int(total or 0))})
    render()


def update_hpo(*, done: int, running: int, total: int | None = None, message: str = "") -> None:
    with _LOCK:
        if total is not None:
            _STATE["hpo_total"] = max(0, int(total or 0))
        _STATE["hpo_done"] = max(0, int(done or 0))
        _STATE["hpo_running"] = max(0, int(running or 0))
        if message:
            _STATE["message"] = message
    render()


def emit_hpo_update(*, done: int, running: int, total: int, message: str = "") -> None:
    if str(os.environ.get("RESEARCH_PROGRESS_BRIDGE") or "").strip() != "1":
        update_hpo(done=done, running=running, total=total, message=message)
        return
    payload = {
        "type": "hpo",
        "done": int(done or 0),
        "running": int(running or 0),
        "total": int(total or 0),
        "message": str(message or ""),
    }
    print(PROGRESS_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def handle_progress_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text.startswith(PROGRESS_PREFIX):
        return False
    try:
        payload = json.loads(text[len(PROGRESS_PREFIX):])
    except json.JSONDecodeError:
        return True
    if payload.get("type") == "hpo":
        update_hpo(
            done=int(payload.get("done") or 0),
            running=int(payload.get("running") or 0),
            total=int(payload.get("total") or 0),
            message=str(payload.get("message") or ""),
        )
    return True


def clear() -> None:
    if not _enabled():
        return
    width = _width()
    print("\r" + (" " * width) + "\r", end="", file=sys.stderr, flush=True)


def render() -> None:
    if not _enabled():
        return
    with _LOCK:
        text = _format_state(dict(_STATE))
    width = _width()
    if len(text) > width:
        text = text[: max(0, width - 1)]
    print("\r" + text.ljust(width), end="", file=sys.stderr, flush=True)


def _format_state(state: dict[str, Any]) -> str:
    stage_total = int(state.get("stage_total") or 0)
    stage_done = int(state.get("stage_done") or 0)
    stage_index = int(state.get("stage_index") or 0)
    stage = str(state.get("stage") or "")
    hpo_total = int(state.get("hpo_total") or 0)
    hpo_done = int(state.get("hpo_done") or 0)
    hpo_running = int(state.get("hpo_running") or 0)
    work_done = stage_done + hpo_done
    work_total = stage_total + hpo_total
    pct = 0.0 if work_total <= 0 else min(100.0, 100.0 * work_done / work_total)
    hpo = f"trials {hpo_done}/{hpo_total} done, {hpo_running} running" if hpo_total else "trials n/a"
    stage_part = f"stage {stage_index}/{stage_total} {stage}" if stage_total else f"stage {stage}"
    msg = str(state.get("message") or "").strip()
    suffix = f" | {msg}" if msg else ""
    return f"[{pct:5.1f}%] {stage_part} | {hpo}{suffix}"


def _enabled() -> bool:
    return str(os.environ.get("RESEARCH_PROGRESS", "1")).strip().lower() not in {"0", "false", "no"}


def _width() -> int:
    return max(80, shutil.get_terminal_size((120, 20)).columns)
