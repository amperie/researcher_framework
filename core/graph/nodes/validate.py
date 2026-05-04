"""Validate generated implementations.

Validation is contract-first. Profiles can configure deterministic contract
tests with ``validate.contract_test``. LLM-generated tests are optional and only
used when ``validate.llm_generate_tests`` is true.

If tests fail, the existing LLM fix loop can still repair the generated
implementation up to ``validate.max_fix_retries``.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from configs.config import get_config, resolve_dev_path
from core.graph.nodes.artifact_refs import (
    register_implementation_artifact,
    register_validation_result_artifact,
    register_validation_test_artifact,
)
from core.graph.nodes.memory import persist_memory_records_for_state
from core.graph.nodes.code_safety import extract_python_source, validate_python_source
from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)


def validate_node(state: ResearchState, profile: dict) -> dict:
    implementations = state.get("implementations") or []
    if not implementations:
        log.warning("validate_node | No implementations to validate")
        return {"validation_results": []}

    validate_cfg = profile.get("validate") or {}
    auto_run: bool = validate_cfg.get("auto_run", True)
    contract_test: str | None = validate_cfg.get("contract_test")
    llm_generate_tests: bool = validate_cfg.get("llm_generate_tests", not bool(contract_test))
    max_retries: int = validate_cfg.get("max_fix_retries", 3)
    test_runner: str = validate_cfg.get("test_runner", "uv run pytest")
    test_output_dir = resolve_dev_path(
        validate_cfg.get("test_output_dir", "dev/experiments/tests")
    )
    test_output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    scan_context = _scan_context(profile)

    updated_impls = list(implementations)
    validation_results: list[dict] = []
    errors = list(state.get("errors") or [])

    for idx, impl in enumerate(implementations):
        script_path = impl.get("script_path", "")
        class_name = impl.get("class_name", "unknown")
        proposal_name = impl.get("proposal_name") or class_name

        if not script_path or not Path(script_path).exists():
            log.warning("validate_node | Skipping %r - no valid script_path", class_name)
            validation_results.append({
                "script_path": script_path,
                "class_name": class_name,
                "proposal_name": proposal_name,
                "passed": False,
                "test_file": "",
                "test_output": "No script to validate",
                "attempts": 0,
                "test_source": "none",
            })
            continue

        code = Path(script_path).read_text(encoding="utf-8")
        test_file = test_output_dir / f"test_{class_name}.py"

        try:
            test_code, test_source = _build_test_code(
                profile=profile,
                contract_test=contract_test,
                llm_generate_tests=llm_generate_tests,
                script_path=script_path,
                class_name=class_name,
                expected_feature_set_name=impl.get("proposal_name") or class_name,
                scan_context=scan_context,
                code=code,
            )
            test_file.write_text(test_code, encoding="utf-8")
            log.info("validate_node | %s test file written - %s", test_source, test_file)
        except Exception as exc:
            log.error("validate_node | Test creation failed for %r: %s", class_name, exc)
            validation_results.append({
                "script_path": script_path,
                "class_name": class_name,
                "proposal_name": proposal_name,
                "passed": False,
                "test_file": str(test_file),
                "test_output": f"Test generation failed: {exc}",
                "attempts": 0,
                "test_source": "failed",
            })
            errors.append(f"validate: test generation failed for {class_name}: {exc}")
            continue

        if not auto_run:
            log.info("validate_node | auto_run=False - tests written, not executed")
            validation_results.append({
                "script_path": script_path,
                "class_name": class_name,
                "proposal_name": proposal_name,
                "passed": None,
                "test_file": str(test_file),
                "test_output": "auto_run=False",
                "attempts": 0,
                "test_source": test_source,
            })
            continue

        test_file_record = register_validation_test_artifact(
            profile,
            proposal_name=proposal_name,
            class_name=class_name,
            test_file=str(test_file),
            test_source=test_source,
            errors=errors,
        )
        passed = False
        test_output = ""
        attempts = 0
        current_code = code

        while attempts <= max_retries:
            preflight_error = _preflight_validation_error(current_code)
            if preflight_error:
                test_output = f"PREFLIGHT VALIDATION ERROR: {preflight_error}"
            else:
                test_output = _run_tests(test_runner, str(test_file), cfg.validate_timeout_seconds)
            passed = _pytest_output_passed(test_output)
            failure_summary = _summarize_test_failure(test_output)

            log.info(
                "validate_node | %r attempt %d/%d - passed=%s%s",
                class_name,
                attempts + 1,
                max_retries + 1,
                passed,
                f" - {failure_summary}" if failure_summary and not passed else "",
            )

            if passed or attempts == max_retries:
                break

            log.info(
                "validate_node | Requesting LLM fix for %r (attempt %d) - failure=%s",
                class_name,
                attempts + 1,
                failure_summary or "(no pytest failure summary parsed)",
            )
            try:
                fix_prompt = get_prompt(profile, "validate", "fix_system")
                llm = get_llm("validate", profile)
                fix_resp = llm.invoke([
                    SystemMessage(content=fix_prompt),
                    HumanMessage(
                        content=(
                            "Your response must be raw ASCII Python source only.\n"
                            "Do not include explanation, markdown fences, bullets, Unicode dashes, or any text before/after the code.\n\n"
                            f"Implementation:\n```python\n{current_code}\n```\n\n"
                            f"Test file:\n```python\n{test_code}\n```\n\n"
                            "Important API constraint:\n"
                            "- FeatureSetBase.make_column_name takes exactly one string payload argument.\n"
                            "- Build the full column name first, then call self.make_column_name(full_name).\n\n"
                            f"Failure output:\n{test_output[-3000:]}"
                        )
                    ),
                ])
                fixed_code = extract_python_source(fix_resp.content)
                validate_python_source(fixed_code, expected_class_name=class_name)
                Path(script_path).write_text(fixed_code, encoding="utf-8")
                current_code = fixed_code
                log.info("validate_node | Fixed code written - %s", script_path)
            except Exception as exc:
                log.error("validate_node | Fix generation failed or was rejected: %s", exc)
                test_output = f"{test_output}\n\nFix response rejected: {exc}"
                break

            attempts += 1

        validation_result = {
            "script_path": script_path,
            "class_name": class_name,
            "proposal_name": proposal_name,
            "passed": passed,
            "test_file": str(test_file),
            "test_output": test_output[-2000:],
            "attempts": attempts,
            "test_source": test_source,
        }
        if test_file_record:
            validation_result["test_file_artifact_id"] = test_file_record["artifact_id"]
            validation_result["test_file_artifact_uri"] = test_file_record["uri"]

        updated_impls[idx] = {
            **impl,
            "script_path": script_path,
            "proposal_name": proposal_name,
            "validated": passed,
        }
        register_implementation_artifact(profile, updated_impls[idx], errors)
        if updated_impls[idx].get("stored_artifact_id"):
            validation_result["implementation_artifact_id"] = updated_impls[idx]["stored_artifact_id"]
            validation_result["implementation_artifact_uri"] = updated_impls[idx].get("stored_artifact_uri", "")

        result_record = register_validation_result_artifact(profile, validation_result, errors)
        if result_record:
            validation_result["stored_artifact_id"] = result_record["artifact_id"]
            validation_result["stored_artifact_uri"] = result_record["uri"]
        validation_results.append(validation_result)

        if not passed:
            errors.append(f"validate: {class_name} failed after {attempts} fix attempt(s)")
            log.warning(
                "validate_node | %r did not pass after %d attempts - %s",
                class_name,
                attempts,
                _summarize_test_failure(test_output) or "see validation_results.test_output",
            )

    delta = {
        "implementations": updated_impls,
        "validation_results": validation_results,
        "errors": errors,
    }
    try:
        persist_memory_records_for_state(profile, {**state, **delta})
    except Exception as exc:
        log.warning("validate_node | Memory persistence failed: %s", exc)
        delta["errors"] = errors + [f"validate: memory persistence failed: {exc}"]
    return delta


def _build_test_code(
    profile: dict,
    contract_test: str | None,
    llm_generate_tests: bool,
    script_path: str,
    class_name: str,
    expected_feature_set_name: str,
    scan_context: str,
    code: str,
) -> tuple[str, str]:
    if contract_test:
        return (
            _build_contract_test(
                contract_test=contract_test,
                script_path=script_path,
                class_name=class_name,
                expected_feature_set_name=expected_feature_set_name,
            ),
            f"contract:{contract_test}",
        )

    if llm_generate_tests:
        system_prompt = get_prompt(profile, "validate")
        llm = get_llm("validate", profile)
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Scan field context:\n{scan_context}\n\nCode to test:\n```python\n{code}\n```"),
        ])
        test_source = extract_python_source(resp.content)
        validate_python_source(test_source)
        return test_source, "llm_generated"

    raise ValueError("No validation contract configured and llm_generate_tests=False")


def _scan_context(profile: dict) -> str:
    datasets = profile.get("datasets") or []
    parts = []
    for ds in datasets:
        asf = ds.get("available_scan_fields") or {}
        parts.append(f"Dataset '{ds['name']}' guaranteed fields: {asf.get('guaranteed', [])}")
    return "\n".join(parts)


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


def _pytest_output_passed(output: str) -> bool:
    lowered = output.lower()
    return " passed" in lowered and "failed" not in lowered and "error" not in lowered


def _summarize_test_failure(output: str, max_chars: int = 700) -> str:
    """Extract a concise pytest failure/error summary for logs."""
    if not output:
        return ""

    lines = [line.rstrip() for line in output.splitlines()]
    selected: list[str] = []

    capture = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        if (
            stripped.startswith(("FAILED ", "ERROR "))
            or lower.startswith(("e       ", "e   ", "assert "))
            or "AssertionError" in stripped
            or "Traceback " in stripped
            or "short test summary info" in lower
        ):
            selected.append(stripped)
            capture = True
            continue

        if capture and (
            "::" in stripped
            or stripped.startswith(("E ", "E\t", "> "))
            or lower.startswith(("failed ", "error "))
        ):
            selected.append(stripped)

        if len(" | ".join(selected)) >= max_chars:
            break

    if not selected:
        tail = [line.strip() for line in lines[-12:] if line.strip()]
        selected = tail

    summary = " | ".join(selected)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def _strip_fences(text: str) -> str:
    return extract_python_source(text)


def _build_contract_test(
    contract_test: str,
    script_path: str,
    class_name: str,
    expected_feature_set_name: str,
) -> str:
    if contract_test == "neuralsignal_feature_set":
        return _build_neuralsignal_feature_set_contract_test(
            script_path=script_path,
            class_name=class_name,
            expected_feature_set_name=expected_feature_set_name,
        )
    raise ValueError(f"Unknown validation contract_test: {contract_test!r}")


def _build_neuralsignal_feature_set_contract_test(
    script_path: str,
    class_name: str,
    expected_feature_set_name: str,
) -> str:
    """Return deterministic pytest code for the FeatureSetBase API contract."""
    script_path_json = json.dumps(str(Path(script_path).resolve()))
    class_name_json = json.dumps(class_name)
    expected_name_json = json.dumps(expected_feature_set_name)
    return f'''\
import importlib.util
import math
import sys
import types

import pandas as pd
import pytest
import torch

SCRIPT_PATH = {script_path_json}
CLASS_NAME = {class_name_json}
EXPECTED_FEATURE_SET_NAME = {expected_name_json}


def test_implementation_does_not_install_runtime_stubs():
    source = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "class FeatureSetBase" not in source, "Implementation must import FeatureSetBase, not define a local stub"
    assert "sys.modules" not in source, "Implementation must not create fake neuralsignal modules"


class FeatureSetBase:
    def __init__(self, config):
        self.config = config

    def make_column_name(self, name):
        prefix = self.config.get("name", "")
        return f"{{prefix}}_{{name}}" if prefix else str(name)


def is_layer_string_match_in_list(layer_name, patterns):
    return any(str(pattern) in str(layer_name) for pattern in patterns)


def _install_neuralsignal_stubs():
    modules = {{
        "neuralsignal": types.ModuleType("neuralsignal"),
        "neuralsignal.core": types.ModuleType("neuralsignal.core"),
        "neuralsignal.core.modules": types.ModuleType("neuralsignal.core.modules"),
        "neuralsignal.core.modules.feature_sets": types.ModuleType("neuralsignal.core.modules.feature_sets"),
        "neuralsignal.core.modules.feature_sets.feature_set_base": types.ModuleType(
            "neuralsignal.core.modules.feature_sets.feature_set_base"
        ),
        "neuralsignal.core.modules.feature_sets.feature_utils": types.ModuleType(
            "neuralsignal.core.modules.feature_sets.feature_utils"
        ),
    }}
    modules["neuralsignal.core.modules.feature_sets.feature_set_base"].FeatureSetBase = FeatureSetBase
    modules["neuralsignal.core.modules.feature_sets.feature_utils"].is_layer_string_match_in_list = (
        is_layer_string_match_in_list
    )
    sys.modules.update(modules)


def _load_class():
    _install_neuralsignal_stubs()
    spec = importlib.util.spec_from_file_location("generated_feature_set", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    cls = getattr(module, CLASS_NAME, None)
    if cls is None:
        candidates = [
            obj for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, FeatureSetBase) and obj is not FeatureSetBase
        ]
        assert candidates, "No FeatureSetBase subclass found"
        cls = candidates[0]
    return cls


def _config(output_format="name_and_value_columns"):
    return {{
        "name": EXPECTED_FEATURE_SET_NAME,
        "output_format": output_format,
        "ffn_layer_patterns": ["mlp", "fc"],
        "attn_layer_patterns": [".q", ".k", ".v", "attn"],
    }}


def _scan():
    return {{
        "outputs": {{
            "layer_0": torch.randn(8, 16),
            "layer_1": torch.randn(8, 16),
            "layer_2": torch.randn(8, 16),
            "layer_3": torch.randn(8, 16),
        }},
        "inputs": {{
            "layer_0": torch.randn(8, 16),
            "layer_1": torch.randn(8, 16),
            "layer_2": torch.randn(8, 16),
            "layer_3": torch.randn(8, 16),
        }},
        "layer_id_to_name": {{
            "layer_0": "model.layers.0.mlp.fc",
            "layer_1": "model.layers.0.attn.q_proj",
            "layer_2": "model.layers.0.attn.k_proj",
            "layer_3": "model.layers.0.attn.o_proj",
        }},
        "layer_order": ["layer_0", "layer_1", "layer_2", "layer_3"],
        "layer_passes": {{"layer_0": 1, "layer_1": 1, "layer_2": 1, "layer_3": 1}},
        "zone_size": 512,
        "ground_truth": 1,
    }}


def test_feature_set_contract_name_and_value_columns():
    cls = _load_class()
    instance = cls(_config("name_and_value_columns"))

    assert hasattr(instance, "get_feature_set_name")
    assert instance.get_feature_set_name() == EXPECTED_FEATURE_SET_NAME
    assert hasattr(instance, "process_feature_set")

    result = instance.process_feature_set(_scan())
    assert isinstance(result, tuple)
    assert len(result) == 2
    cols, vals = result
    assert isinstance(cols, list)
    assert isinstance(vals, list)
    assert len(cols) == len(vals)
    assert len(cols) > 0
    assert all(isinstance(col, str) and col for col in cols)
    assert all(isinstance(val, (float, int)) and not isinstance(val, bool) for val in vals)
    assert all(math.isfinite(float(val)) for val in vals)
    assert any(float(val) != 0.0 for val in vals)


def test_feature_set_contract_pandas_output():
    cls = _load_class()
    instance = cls(_config("pandas"))

    result = instance.process_feature_set(_scan())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert len(result.columns) > 0
    assert all(isinstance(col, str) and col for col in result.columns)
    assert result.notna().all(axis=None)


def test_feature_set_contract_tensor_dict_default():
    cls = _load_class()
    instance = cls(_config("tensor_dict"))

    result = instance.process_feature_set(_scan())
    assert result is None or isinstance(result, dict)


def test_feature_set_contract_invalid_output_format():
    cls = _load_class()
    instance = cls(_config("invalid"))

    with pytest.raises(ValueError):
        instance.process_feature_set(_scan())
'''


def _preflight_validation_error(code: str) -> str:
    """Return a targeted validation error for known contract violations."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "make_column_name"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            continue
        if len(node.args) != 1 or node.keywords:
            return (
                "FeatureSetBase.make_column_name accepts exactly one positional string argument; "
                "combine name parts first and call self.make_column_name(full_name)."
            )
    return ""
