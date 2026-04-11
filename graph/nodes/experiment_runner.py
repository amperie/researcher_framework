"""Experiment runner node — execute the generated script in an isolated subprocess."""
from __future__ import annotations

from graph.state import ResearchState
from logger import get_logger

log = get_logger(__name__)


def experiment_runner_node(state: ResearchState) -> dict:
    """Run the generated experiment script and capture its results.

    Reads:
        state['generated_code']

    Writes:
        state['experiment_id']       — UUID for this run
        state['execution_stdout']    — full captured stdout
        state['execution_stderr']    — full captured stderr
        state['execution_success']   — True if exit code 0 and last line is valid JSON
        state['raw_results']         — parsed JSON dict from last stdout line

    Execution:
        1. Generate experiment_id = str(uuid4()).
        2. Write generated_code to {cfg.experiments_dir}/{experiment_id}.py.
        3. Run via subprocess with cfg.neuralsignal_src_path on PYTHONPATH.
        4. Parse last non-empty line of stdout as JSON → raw_results.
        5. Set execution_success = (returncode == 0) and raw_results parsed OK.

    TODO: implement steps 1–5.
    TODO: on TimeoutExpired, set execution_success=False and record in errors.
    TODO: on JSON parse failure, set execution_success=False and record in errors.
    """
    code_len = len(state.get("generated_code") or "")
    log.info("experiment_runner_node | Preparing experiment subprocess — code_len=%d", code_len)

    # TODO: experiment_id = str(uuid4())
    # log.info("experiment_runner_node | Assigned experiment_id=%r", experiment_id)

    # TODO: write script to disk
    # log.debug("experiment_runner_node | Script written to %r", script_path)

    # TODO: launch subprocess
    # log.info("experiment_runner_node | Launching subprocess — command=%s, timeout=%ds",
    #          command, cfg.experiment_timeout_seconds)

    # On completion:
    # log.info("experiment_runner_node | Subprocess finished — exit_code=%d, "
    #          "stdout_lines=%d, stderr_lines=%d",
    #          result.returncode,
    #          result.stdout.count("\n"),
    #          result.stderr.count("\n"))
    # if result.returncode != 0:
    #     log.warning("experiment_runner_node | Non-zero exit — stderr tail: %s",
    #                 result.stderr[-500:])

    # On timeout:
    # log.error("experiment_runner_node | Subprocess timed out after %ds", timeout)

    # On JSON parse:
    # log.info("experiment_runner_node | Parsed results — keys=%s", list(raw_results.keys()))
    # or:
    # log.warning("experiment_runner_node | Could not parse JSON from stdout last line: %r",
    #             last_line[:200])

    raise NotImplementedError("experiment_runner_node is not yet implemented")
