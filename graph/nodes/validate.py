"""Validate step — generate unit tests and auto-run them with fix-retry loop.

For each implementation:
  1. LLM generates pytest unit tests
  2. Tests are run in a subprocess
  3. On failure (up to max_fix_retries), LLM receives code + test output and fixes the impl
  4. Failures after max retries are recorded but do NOT abort the pipeline

Reads:
    state['implementations']

Writes:
    state['validation_results']
    state['implementations']   — script_path updated if code was fixed
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from configs.config import get_config
from graph.state import ResearchState
from llm.factory import get_llm
from utils.logger import get_logger
from utils.profile_loader import get_prompt

log = get_logger(__name__)


def validate_node(state: ResearchState, profile: dict) -> dict:
    implementations = state.get("implementations") or []
    if not implementations:
        log.warning("validate_node | No implementations to validate")
        return {"validation_results": []}

    validate_cfg = profile.get("validate") or {}
    auto_run: bool = validate_cfg.get("auto_run", True)
    max_retries: int = validate_cfg.get("max_fix_retries", 3)
    test_runner: str = validate_cfg.get("test_runner", "uv run pytest")
    test_output_dir = Path(validate_cfg.get("test_output_dir", "dev/experiments/tests"))
    test_output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = get_prompt(profile, "validate")
    fix_prompt = get_prompt(profile, "validate", "fix_system")
    llm = get_llm("validate", profile)
    cfg = get_config()

    # Inject scan constraints so generated tests use correct field structure
    datasets = profile.get("datasets") or []
    scan_context = ""
    for ds in datasets:
        asf = ds.get("available_scan_fields") or {}
        scan_context += (
            f"Dataset '{ds['name']}' guaranteed fields: {asf.get('guaranteed', [])}\n"
        )

    updated_impls = list(implementations)
    validation_results: list[dict] = []
    errors = list(state.get("errors") or [])

    for idx, impl in enumerate(implementations):
        script_path = impl.get("script_path", "")
        class_name = impl.get("class_name", "unknown")

        if not script_path or not Path(script_path).exists():
            log.warning("validate_node | Skipping %r — no valid script_path", class_name)
            validation_results.append({
                "script_path": script_path,
                "class_name": class_name,
                "passed": False,
                "test_file": "",
                "test_output": "No script to validate",
                "attempts": 0,
            })
            continue

        # Generate test file
        code = Path(script_path).read_text(encoding="utf-8")
        test_file = test_output_dir / f"test_{class_name}.py"

        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Scan field context:\n{scan_context}\n\nCode to test:\n```python\n{code}\n```"
                ),
            ])
            test_code = _strip_fences(resp.content)
            test_file.write_text(test_code, encoding="utf-8")
            log.info("validate_node | Test file written — %s", test_file)
        except Exception as exc:
            log.error("validate_node | Test generation failed for %r: %s", class_name, exc)
            validation_results.append({
                "script_path": script_path,
                "class_name": class_name,
                "passed": False,
                "test_file": str(test_file),
                "test_output": f"Test generation failed: {exc}",
                "attempts": 0,
            })
            errors.append(f"validate: test generation failed for {class_name}: {exc}")
            continue

        if not auto_run:
            log.info("validate_node | auto_run=False — tests written, not executed")
            validation_results.append({
                "script_path": script_path,
                "class_name": class_name,
                "passed": None,  # unknown — not run
                "test_file": str(test_file),
                "test_output": "auto_run=False",
                "attempts": 0,
            })
            continue

        # Run tests with fix-retry loop
        passed = False
        test_output = ""
        attempts = 0
        current_code = code

        while attempts <= max_retries:
            test_output = _run_tests(test_runner, str(test_file), cfg.validate_timeout_seconds)
            passed = "passed" in test_output.lower() and "failed" not in test_output.lower() and "error" not in test_output.lower()

            log.info(
                "validate_node | %r attempt %d/%d — passed=%s",
                class_name, attempts + 1, max_retries + 1, passed,
            )

            if passed or attempts == max_retries:
                break

            # Ask LLM to fix the implementation
            log.info("validate_node | Requesting fix for %r (attempt %d)", class_name, attempts + 1)
            try:
                fix_resp = llm.invoke([
                    SystemMessage(content=fix_prompt),
                    HumanMessage(
                        content=(
                            f"Implementation:\n```python\n{current_code}\n```\n\n"
                            f"Test file:\n```python\n{test_code}\n```\n\n"
                            f"Failure output:\n{test_output[-3000:]}"
                        )
                    ),
                ])
                fixed_code = _strip_fences(fix_resp.content)
                Path(script_path).write_text(fixed_code, encoding="utf-8")
                current_code = fixed_code
                log.info("validate_node | Fixed code written → %s", script_path)
            except Exception as exc:
                log.error("validate_node | Fix generation failed: %s", exc)
                break

            attempts += 1

        validation_results.append({
            "script_path": script_path,
            "class_name": class_name,
            "passed": passed,
            "test_file": str(test_file),
            "test_output": test_output[-2000:],
            "attempts": attempts,
        })

        if not passed:
            errors.append(
                f"validate: {class_name} failed after {attempts} fix attempt(s)"
            )
            log.warning("validate_node | %r did not pass after %d attempts", class_name, attempts)

        # Update the implementation entry with the (possibly fixed) script
        updated_impls[idx] = {**impl, "script_path": script_path, "validated": passed}

    return {
        "implementations": updated_impls,
        "validation_results": validation_results,
        "errors": errors,
    }


def _run_tests(test_runner: str, test_file: str, timeout: int) -> str:
    """Run tests and return combined stdout+stderr output."""
    cmd = test_runner.split() + [test_file, "-v", "--tb=short"]
    log.debug("validate_node | Running: %s", cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        log.debug("validate_node | Test exit code: %d", result.returncode)
        return output
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: tests exceeded {timeout}s"
    except Exception as exc:
        return f"ERROR running tests: {exc}"


def _strip_fences(text: str) -> str:
    import re
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()
