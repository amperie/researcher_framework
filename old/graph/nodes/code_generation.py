"""Code generation node — call Claude Code CLI to produce a runnable experiment script."""
from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from uuid import uuid4

from configs.config import get_config
from configs.prompts import CODE_GENERATION
from graph.state import ResearchState
from utils.logger import get_logger

log = get_logger(__name__)

# Matches ```python ... ``` or ``` ... ``` fences (greedy from first to last fence)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Return the content of the first code fence found, or the text as-is."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _parse_claude_output(raw: str) -> str:
    """Extract the code string from Claude Code CLI output.

    The CLI with --output-format json returns a JSON object whose 'result'
    field holds the model response.  Fall back to the raw text if parsing
    fails (e.g. the CLI was invoked without --output-format json).
    """
    try:
        data = json.loads(raw)
        # Claude Code JSON envelope: {"type": "result", "result": "...", ...}
        if isinstance(data, dict) and "result" in data:
            return data["result"]
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def _run_streaming(
    command: list[str],
    prompt: str,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run *command* with *prompt* on stdin, streaming stdout/stderr to the logger.

    stdout lines are logged at DEBUG level as they arrive so progress is visible
    in the log file without waiting for the subprocess to finish.  Both streams
    are collected and returned in a CompletedProcess-compatible object.

    Raises:
        FileNotFoundError: if the command binary is not found.
        subprocess.TimeoutExpired: if *timeout* seconds elapse before the
            process exits.
    """
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    # Write the full prompt and close stdin so the CLI knows input is done.
    try:
        process.stdin.write(prompt)
        process.stdin.close()
    except BrokenPipeError:
        pass  # process already exited

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_done = threading.Event()

    def _read_stdout() -> None:
        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                log.debug("code_generation_node | [claude] %s", line)
                stdout_lines.append(raw_line)
        finally:
            stdout_done.set()

    def _read_stderr() -> None:
        for raw_line in process.stderr:
            line = raw_line.rstrip("\n")
            if line:
                log.debug("code_generation_node | [claude stderr] %s", line)
            stderr_lines.append(raw_line)

    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err = threading.Thread(target=_read_stderr, daemon=True)
    t_out.start()
    t_err.start()

    # Wait for stdout to drain, honouring the timeout.
    finished = stdout_done.wait(timeout=timeout)
    if not finished:
        process.kill()
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        raise subprocess.TimeoutExpired(command, timeout)

    t_err.join(timeout=5)
    process.wait(timeout=5)

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def code_generation_node(state: ResearchState) -> dict:
    """Generate a runnable experiment script for every feature proposal.

    Reads:
        state['feature_proposals']    — one script generated per proposal
        state['research_direction']

    Writes:
        state['generated_scripts']    — list of {experiment_id, code,
                                          experiment_config, proposal_name,
                                          script_path}, one per proposal
        state['generated_code']       — generated_scripts[0]['code']  (compat)
        state['experiment_id']        — generated_scripts[0]['experiment_id'] (compat)
        state['experiment_config']    — generated_scripts[0]['experiment_config'] (compat)

    Each script is written to {cfg.experiments_dir}/{experiment_id}.py.
    Per-proposal failures are logged and skipped; if every proposal fails
    the node returns an error.
    """
    proposals = state.get("feature_proposals") or []
    if not proposals:
        log.error("code_generation_node | No feature proposals in state")
        return {"errors": ["No feature proposals to generate code for"]}

    cfg = get_config()
    direction = state.get("research_direction", "")
    command = cfg.claude_command.split() + ["--print", "--output-format", "json"]
    experiments_dir = Path(cfg.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    log.info("code_generation_node | Generating scripts for %d proposals", len(proposals))

    generated_scripts: list[dict] = []
    errors: list[str] = []

    for idx, proposal in enumerate(proposals):
        proposal_name = proposal.get("name", f"proposal_{idx}")
        experiment_id = str(uuid4())

        log.info(
            "code_generation_node | Script %d/%d — proposal=%r, experiment_id=%s",
            idx + 1, len(proposals), proposal_name, experiment_id,
        )

        # ------------------------------------------------------------------
        # Build prompt
        # ------------------------------------------------------------------
        proposal_json = json.dumps(proposal, indent=2)
        prompt = (
            f"{CODE_GENERATION}\n\n"
            f"{'=' * 63}\n"
            f"TASK\n"
            f"{'=' * 63}\n\n"
            f"Research direction: {direction}\n\n"
            f"Experiment ID (embed in results dict): {experiment_id}\n\n"
            f"FeatureSet proposal:\n{proposal_json}"
        )

        log.debug("code_generation_node | Prompt length=%d chars", len(prompt))

        # ------------------------------------------------------------------
        # Call the Claude Code CLI
        # ------------------------------------------------------------------
        try:
            result = _run_streaming(command, prompt, cfg.code_generation_timeout_seconds)
        except FileNotFoundError:
            msg = (
                f"Claude Code CLI not found: {cfg.claude_command!r}. "
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )
            log.error("code_generation_node | %s", msg)
            # CLI missing is fatal — no point continuing
            return {"errors": [msg]}
        except subprocess.TimeoutExpired:
            msg = (f"Claude Code CLI timed out after {cfg.code_generation_timeout_seconds}s "
                   f"on proposal {proposal_name!r}")
            log.warning("code_generation_node | %s — skipping", msg)
            errors.append(msg)
            continue

        log.info(
            "code_generation_node | CLI finished — exit_code=%d, "
            "stdout=%d chars, stderr=%d chars",
            result.returncode, len(result.stdout), len(result.stderr),
        )
        if result.returncode != 0:
            log.warning("code_generation_node | Non-zero exit — stderr: %s",
                        result.stderr[:500])

        # ------------------------------------------------------------------
        # Extract and clean generated code
        # ------------------------------------------------------------------
        raw_response = _parse_claude_output(result.stdout)
        code = _strip_fences(raw_response)

        if not code:
            msg = f"Claude Code returned empty response for proposal {proposal_name!r}"
            log.warning("code_generation_node | %s — skipping", msg)
            errors.append(msg)
            continue

        log.info("code_generation_node | Script extracted — %d lines, %d chars",
                 len(code.splitlines()), len(code))
        log.debug("code_generation_node | First 300 chars:\n%s", code[:300])

        # ------------------------------------------------------------------
        # Write script to disk
        # ------------------------------------------------------------------
        script_path = experiments_dir / f"{experiment_id}.py"
        script_path.write_text(code, encoding="utf-8")
        log.info("code_generation_node | Script saved — %s", script_path)

        experiment_config = {
            "experiment_id": experiment_id,
            "proposal_name": proposal_name,
            "class_name": proposal.get("class_name"),
            "feature_set_name": proposal.get("feature_set_name"),
            "target_behavior": proposal.get("target_behavior"),
            "script_path": str(script_path),
        }

        generated_scripts.append({
            "experiment_id": experiment_id,
            "code": code,
            "experiment_config": experiment_config,
            "proposal_name": proposal_name,
            "script_path": str(script_path),
        })

    # ------------------------------------------------------------------
    # All proposals failed
    # ------------------------------------------------------------------
    if not generated_scripts:
        log.error("code_generation_node | All %d proposals failed code generation",
                  len(proposals))
        return {"errors": errors or ["All proposals failed code generation"]}

    log.info("code_generation_node | Done — %d/%d scripts generated successfully",
             len(generated_scripts), len(proposals))

    first = generated_scripts[0]
    return {
        "generated_scripts": generated_scripts,
        # Legacy single-value fields for downstream compat
        "generated_code":    first["code"],
        "experiment_id":     first["experiment_id"],
        "experiment_config": first["experiment_config"],
        **({"errors": errors} if errors else {}),
    }
