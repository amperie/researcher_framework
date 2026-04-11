"""Code generation node — call Claude Code CLI to produce a runnable experiment script."""
from __future__ import annotations

import json
import re
import subprocess
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


def code_generation_node(state: ResearchState) -> dict:
    """Generate a runnable FeatureSetBase experiment script via the Claude Code CLI.

    Reads:
        state['feature_proposals']    — uses proposals[0]
        state['research_direction']

    Writes:
        state['generated_code']       — complete Python script (string)
        state['experiment_id']        — UUID for this run
        state['experiment_config']    — structured params dict

    The script is also written to {cfg.experiments_dir}/{experiment_id}.py so
    that experiment_runner_node (and humans) can inspect or re-run it directly.
    """
    proposals = state.get("feature_proposals") or []
    if not proposals:
        log.error("code_generation_node | No feature proposals in state")
        return {"errors": ["No feature proposals to generate code for"]}

    top = proposals[0]
    direction = state.get("research_direction", "")
    experiment_id = str(uuid4())

    log.info(
        "code_generation_node | Generating script — proposal=%r, experiment_id=%s",
        top.get("name"),
        experiment_id,
    )

    # ------------------------------------------------------------------
    # Build prompt: system instructions + proposal JSON
    # ------------------------------------------------------------------
    proposal_json = json.dumps(top, indent=2)
    prompt = (
        f"{CODE_GENERATION}\n\n"
        f"{'=' * 63}\n"
        f"TASK\n"
        f"{'=' * 63}\n\n"
        f"Research direction: {direction}\n\n"
        f"Experiment ID (embed in results dict): {experiment_id}\n\n"
        f"FeatureSet proposal:\n{proposal_json}"
    )

    # ------------------------------------------------------------------
    # Call the Claude Code CLI
    # ------------------------------------------------------------------
    cfg = get_config()
    command = cfg.claude_command.split() + ["--print", "--output-format", "json"]

    log.info("code_generation_node | Calling Claude Code CLI — command=%s, timeout=%ds",
             command, cfg.code_generation_timeout_seconds)
    log.debug("code_generation_node | Prompt length=%d chars", len(prompt))

    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=cfg.code_generation_timeout_seconds,
        )
    except FileNotFoundError:
        msg = (
            f"Claude Code CLI not found: {cfg.claude_command!r}. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )
        log.error("code_generation_node | %s", msg)
        return {"errors": [msg]}
    except subprocess.TimeoutExpired:
        msg = f"Claude Code CLI timed out after {cfg.code_generation_timeout_seconds}s"
        log.error("code_generation_node | %s", msg)
        return {"errors": [msg]}

    log.info(
        "code_generation_node | CLI finished — exit_code=%d, stdout=%d chars, stderr=%d chars",
        result.returncode,
        len(result.stdout),
        len(result.stderr),
    )

    if result.returncode != 0:
        log.warning("code_generation_node | Non-zero exit — stderr: %s", result.stderr[:500])

    # ------------------------------------------------------------------
    # Extract and clean the generated code
    # ------------------------------------------------------------------
    raw_response = _parse_claude_output(result.stdout)
    code = _strip_fences(raw_response)

    if not code:
        msg = "Claude Code returned an empty response"
        log.error("code_generation_node | %s — raw stdout: %r", msg, result.stdout[:300])
        return {"errors": [msg]}

    log.info("code_generation_node | Script extracted — %d lines, %d chars",
             len(code.splitlines()), len(code))
    log.debug("code_generation_node | First 300 chars:\n%s", code[:300])

    # ------------------------------------------------------------------
    # Write script to dev/experiments/<experiment_id>.py
    # ------------------------------------------------------------------
    experiments_dir = Path(cfg.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    script_path = experiments_dir / f"{experiment_id}.py"
    script_path.write_text(code, encoding="utf-8")
    log.info("code_generation_node | Script saved — %s", script_path)

    # ------------------------------------------------------------------
    # Build experiment_config from the top proposal
    # ------------------------------------------------------------------
    experiment_config = {
        "experiment_id": experiment_id,
        "proposal_name": top.get("name"),
        "class_name": top.get("class_name"),
        "feature_set_name": top.get("feature_set_name"),
        "target_behavior": top.get("target_behavior"),
        "script_path": str(script_path),
    }

    return {
        "generated_code": code,
        "experiment_id": experiment_id,
        "experiment_config": experiment_config,
    }
