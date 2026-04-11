"""Experiment runner node — execute the generated script in an isolated subprocess."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from configs.config import get_config
from graph.state import ResearchState
from utils.logger import get_logger

log = get_logger(__name__)


def experiment_runner_node(state: ResearchState) -> dict:
    """Run the generated experiment script and capture its results.

    Reads:
        state['generated_code']      — Python script to run
        state['experiment_id']       — used as the script filename

    Writes:
        state['experiment_id']       — set here if not already present
        state['execution_stdout']    — full captured stdout
        state['execution_stderr']    — full captured stderr
        state['execution_success']   — True if exit code 0 and last line is valid JSON
        state['raw_results']         — parsed JSON dict from the last stdout line

    The script is executed with:
      - cfg.neuralsignal_python as the interpreter (e.g. "uv run python")
      - cfg.neuralsignal_src_path prepended to PYTHONPATH
      - cfg.experiment_timeout_seconds as the wall-clock limit

    Result contract: the script must print a JSON dict as its LAST line of stdout.
    """
    cfg = get_config()
    code = state.get("generated_code") or ""
    experiment_id = state.get("experiment_id") or str(uuid4())

    # ------------------------------------------------------------------
    # Locate / write the script
    # ------------------------------------------------------------------
    experiments_dir = Path(cfg.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    script_path = experiments_dir / f"{experiment_id}.py"

    if code and not script_path.exists():
        script_path.write_text(code, encoding="utf-8")
        log.debug("experiment_runner_node | Script written — %s", script_path)

    if not script_path.exists():
        msg = f"Script not found and no generated_code in state: {script_path}"
        log.error("experiment_runner_node | %s", msg)
        return {
            "experiment_id": experiment_id,
            "execution_stdout": "",
            "execution_stderr": msg,
            "execution_success": False,
            "raw_results": {},
            "errors": [msg],
        }

    # ------------------------------------------------------------------
    # Build environment — inject neuralsignal src onto PYTHONPATH
    # ------------------------------------------------------------------
    env = os.environ.copy()
    ns_path = str(Path(cfg.neuralsignal_src_path).resolve())
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ns_path}{os.pathsep}{existing_pp}" if existing_pp else ns_path

    # ------------------------------------------------------------------
    # Run the script
    # ------------------------------------------------------------------
    command = cfg.neuralsignal_python.split() + [str(script_path)]
    log.info(
        "experiment_runner_node | Launching — command=%s, timeout=%ds",
        command,
        cfg.experiment_timeout_seconds,
    )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=cfg.experiment_timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        msg = f"Experiment timed out after {cfg.experiment_timeout_seconds}s"
        log.error("experiment_runner_node | %s", msg)
        return {
            "experiment_id": experiment_id,
            "execution_stdout": "",
            "execution_stderr": msg,
            "execution_success": False,
            "raw_results": {},
            "errors": [msg],
        }

    log.info(
        "experiment_runner_node | Subprocess finished — exit_code=%d, "
        "stdout_lines=%d, stderr_lines=%d",
        result.returncode,
        result.stdout.count("\n"),
        result.stderr.count("\n"),
    )

    if result.returncode != 0:
        log.warning(
            "experiment_runner_node | Non-zero exit — stderr tail:\n%s",
            result.stderr[-800:],
        )

    # ------------------------------------------------------------------
    # Parse the last non-empty stdout line as JSON
    # ------------------------------------------------------------------
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    last_line = stdout_lines[-1] if stdout_lines else ""

    try:
        raw_results = json.loads(last_line)
        success = result.returncode == 0
        log.info(
            "experiment_runner_node | Results parsed — keys=%s, status=%s",
            list(raw_results.keys()),
            raw_results.get("status"),
        )
    except (json.JSONDecodeError, ValueError):
        raw_results = {}
        success = False
        log.warning(
            "experiment_runner_node | Could not parse JSON from last stdout line: %r",
            last_line[:200],
        )

    return {
        "experiment_id": experiment_id,
        "execution_stdout": result.stdout,
        "execution_stderr": result.stderr,
        "execution_success": success,
        "raw_results": raw_results,
    }
