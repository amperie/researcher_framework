"""Experiment runner node — execute the generated script in an isolated subprocess."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from configs.config import get_config
from graph.state import ResearchState
from utils.dataset_manager import DatasetManager
from utils.logger import get_logger

log = get_logger(__name__)


def _run_single(
    script: dict,
    cfg,
    env: dict,
) -> dict:
    """Run one generated script and return a result dict.

    Args:
        script: Entry from state['generated_scripts'].
        cfg:    Active Config instance.
        env:    Environment dict with PYTHONPATH already set.

    Returns:
        Dict with keys: experiment_id, proposal_name, stdout, stderr,
        success, raw_results.
    """
    experiment_id = script["experiment_id"]
    proposal_name = script.get("proposal_name", "")
    script_path = Path(script["script_path"])

    # Write script to disk if the code_generation_node hasn't already done so.
    if not script_path.exists():
        code = script.get("code", "")
        if code:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(code, encoding="utf-8")
            log.debug("experiment_runner_node | Script written — %s", script_path)
        else:
            msg = f"Script not found and no code in state: {script_path}"
            log.error("experiment_runner_node | %s", msg)
            return {
                "experiment_id": experiment_id,
                "proposal_name": proposal_name,
                "stdout": "",
                "stderr": msg,
                "success": False,
                "raw_results": {},
            }

    command = cfg.neuralsignal_python.split() + [str(script_path)]
    log.info(
        "experiment_runner_node | Launching — proposal=%r, experiment_id=%s, timeout=%ds",
        proposal_name, experiment_id, cfg.experiment_timeout_seconds,
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
        log.error("experiment_runner_node | %s (proposal=%r)", msg, proposal_name)
        return {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "stdout": "",
            "stderr": msg,
            "success": False,
            "raw_results": {},
        }

    log.info(
        "experiment_runner_node | Finished — proposal=%r, exit_code=%d, "
        "stdout_lines=%d, stderr_lines=%d",
        proposal_name,
        result.returncode,
        result.stdout.count("\n"),
        result.stderr.count("\n"),
    )
    if result.returncode != 0:
        log.warning(
            "experiment_runner_node | Non-zero exit (proposal=%r) — stderr tail:\n%s",
            proposal_name, result.stderr[-800:],
        )

    # Parse last non-empty stdout line as JSON
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    last_line = stdout_lines[-1] if stdout_lines else ""
    try:
        raw_results = json.loads(last_line)
        success = result.returncode == 0
        log.info(
            "experiment_runner_node | Results parsed (proposal=%r) — keys=%s",
            proposal_name, list(raw_results.keys()),
        )
    except (json.JSONDecodeError, ValueError):
        raw_results = {}
        success = False
        log.warning(
            "experiment_runner_node | JSON parse failed (proposal=%r) — last line: %r",
            proposal_name, last_line[:200],
        )

    return {
        "experiment_id": experiment_id,
        "proposal_name": proposal_name,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": success,
        "raw_results": raw_results,
    }


def experiment_runner_node(state: ResearchState) -> dict:
    """Run every generated experiment script and collect results.

    Reads:
        state['generated_scripts']   — list of {experiment_id, code,
                                         experiment_config, proposal_name,
                                         script_path}
        Falls back to legacy state['generated_code'] / state['experiment_id']
        if generated_scripts is absent (backward compat).

    Writes:
        state['experiment_results']  — one result dict per script
        state['execution_success']   — True if ≥1 script succeeded
        state['experiment_id']       — ID of first successful script (compat)
        state['execution_stdout']    — stdout of first successful script (compat)
        state['execution_stderr']    — stderr of first successful script (compat)
        state['raw_results']         — raw_results of first successful script (compat)

    Each script is executed with cfg.neuralsignal_src_path on PYTHONPATH.
    """
    cfg = get_config()

    # ------------------------------------------------------------------
    # Resolve script list — prefer new generated_scripts, fall back to legacy
    # ------------------------------------------------------------------
    scripts = state.get("generated_scripts") or []
    if not scripts:
        legacy_code = state.get("generated_code") or ""
        legacy_id = state.get("experiment_id") or str(uuid4())
        if legacy_code:
            log.debug("experiment_runner_node | No generated_scripts — using legacy generated_code")
            experiments_dir = Path(cfg.experiments_dir)
            experiments_dir.mkdir(parents=True, exist_ok=True)
            script_path = experiments_dir / f"{legacy_id}.py"
            if not script_path.exists():
                script_path.write_text(legacy_code, encoding="utf-8")
            scripts = [{
                "experiment_id": legacy_id,
                "code": legacy_code,
                "experiment_config": state.get("experiment_config", {}),
                "proposal_name": "",
                "script_path": str(script_path),
            }]
        else:
            msg = "No generated_scripts or generated_code found in state"
            log.error("experiment_runner_node | %s", msg)
            return {"execution_success": False, "errors": [msg]}

    # ------------------------------------------------------------------
    # Build shared environment
    # ------------------------------------------------------------------
    env = os.environ.copy()
    ns_path = str(Path(cfg.neuralsignal_src_path).resolve())
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ns_path}{os.pathsep}{existing_pp}" if existing_pp else ns_path

    # ------------------------------------------------------------------
    # Run all scripts
    # ------------------------------------------------------------------
    log.info("experiment_runner_node | Running %d script(s)", len(scripts))
    experiment_results: list[dict] = []
    dm = DatasetManager(cfg.mongo_url, cfg.datasets_db_name)
    dataset_ids: list[str] = []
    cache_hits = 0

    for idx, script in enumerate(scripts):
        log.info("experiment_runner_node | Script %d/%d — proposal=%r",
                 idx + 1, len(scripts), script.get("proposal_name"))

        experiment_config = script.get("experiment_config") or {}
        feature_set_name = experiment_config.get("feature_set_name", "")

        # --- cache check ---
        cached = dm.find_cached(feature_set_name, experiment_config) if feature_set_name else None
        if cached:
            log.info("experiment_runner_node | Cache hit — dataset_id=%s", cached["dataset_id"])
            result = dm.build_cached_result(cached, script)
            dataset_ids.append(cached["dataset_id"])
            cache_hits += 1
            experiment_results.append(result)
            continue

        result = _run_single(script, cfg, env)

        # --- save on success ---
        if result["success"] and feature_set_name:
            raw = result.get("raw_results") or {}
            col_names = raw.get("column_names")
            col_vals = raw.get("column_values")
            if col_names and col_vals is not None:
                try:
                    entry = dm.save_dataset(
                        result["experiment_id"], feature_set_name,
                        experiment_config, col_names, col_vals,
                    )
                    result["dataset_id"] = entry["dataset_id"]
                    result["from_cache"] = False
                    dataset_ids.append(entry["dataset_id"])
                    log.info("experiment_runner_node | Dataset saved — id=%s", entry["dataset_id"])
                except Exception as exc:
                    log.warning("experiment_runner_node | Dataset save failed: %s", exc)
            else:
                log.debug("experiment_runner_node | No column data in results — skipping save")

        experiment_results.append(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_success = sum(1 for r in experiment_results if r["success"])
    log.info("experiment_runner_node | Complete — %d/%d scripts succeeded",
             n_success, len(experiment_results))

    # ------------------------------------------------------------------
    # Populate legacy fields from first successful result (or first overall)
    # ------------------------------------------------------------------
    best = next(
        (r for r in experiment_results if r["success"]),
        experiment_results[0],
    )

    return {
        "experiment_results": experiment_results,
        # Legacy compat fields
        "experiment_id":    best["experiment_id"],
        "execution_stdout": best["stdout"],
        "execution_stderr": best["stderr"],
        "execution_success": n_success > 0,
        "raw_results":      best["raw_results"],
        # Dataset registry fields
        "dataset_ids":        dataset_ids,
        "dataset_cache_hits": cache_hits,
    }
