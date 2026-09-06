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
from core.plugins.execution import run_task
from core.plugins.loader import adapter_has, load_adapter
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
    adapter = load_adapter(profile) if profile.get("experiment_adapter") else None

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
                runtime_spec = _validation_runtime_spec(adapter, profile)
                test_output = _run_tests(
                    test_runner,
                    str(test_file.resolve()),
                    cfg.validate_timeout_seconds,
                    profile=profile,
                    runtime_spec=runtime_spec,
                    script_source=current_code,
                    test_source=test_code,
                    script_name=Path(script_path).name,
                    test_name=test_file.name,
                )
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
                profile=profile,
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


def _run_tests(
    test_runner: str,
    test_file: str,
    timeout: int,
    *,
    profile: dict | None = None,
    runtime_spec: dict | None = None,
    script_source: str | None = None,
    test_source: str | None = None,
    script_name: str | None = None,
    test_name: str | None = None,
) -> str:
    """Run tests and return combined stdout+stderr output."""
    cmd = test_runner.split() + [test_file, "-v", "--tb=short"]
    log.debug("validate_node | Running: %s", cmd)
    if runtime_spec:
        try:
            result = run_task(
                {
                    "task_path": "core.plugins.framework_tasks.run_tests",
                    "payload": {
                        "cmd": cmd,
                        "cmd_prefix": test_runner.split(),
                        "cmd_suffix": ["-v", "--tb=short"],
                        "test_path": test_name or Path(test_file).name,
                        "staged_files": {
                            (script_name or "generated_impl.py"): script_source or "",
                            (test_name or Path(test_file).name): test_source or "",
                        },
                        "env": {
                            "GENERATED_SCRIPT_PATH": script_name or "generated_impl.py",
                            "VALIDATION_PLATFORM_ROOT": "",
                        },
                        "timeout": timeout,
                        "cwd": runtime_spec.get("cwd"),
                    },
                    "python": str(runtime_spec["python"]),
                    "timeout": timeout + 5,
                    "plugin_name": str(runtime_spec.get("plugin_name") or "framework"),
                    "logger_prefixes": list(runtime_spec.get("logger_prefixes") or []),
                    "cwd": runtime_spec.get("cwd"),
                    "pythonpath_entries": list(runtime_spec.get("pythonpath_entries") or []),
                    "env": dict(runtime_spec.get("env") or {}),
                    "job_id": f"validate_{Path(test_file).stem}",
                    "job_dir": str(resolve_dev_path(f"dev/experiments/validation/jobs/{Path(test_file).stem}")),
                },
                profile or {},
                "validate",
                default_runner="sync",
            )
            return str(result.get("output") or "")
        except Exception as exc:
            return f"ERROR running tests: {exc}"
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


def _validation_runtime_spec(adapter, profile: dict) -> dict | None:
    if adapter is None or not adapter_has(adapter, "external_runtime_spec"):
        return None
    try:
        spec = adapter.external_runtime_spec(profile, "validate")
    except Exception as exc:
        log.warning("validate_node | Failed to resolve external validation runtime: %s", exc)
        return None
    return spec if isinstance(spec, dict) and spec.get("python") else None


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
    profile: dict,
    contract_test: str,
    script_path: str,
    class_name: str,
    expected_feature_set_name: str,
) -> str:
    if contract_test == "trading_researcher_feature_set":
        return _build_trading_researcher_feature_set_contract_test(
            script_path=script_path,
            class_name=class_name,
            expected_feature_set_name=expected_feature_set_name,
        )
    if contract_test == "trading_algorithm":
        return _build_trading_algorithm_contract_test(
            profile=profile,
            script_path=script_path,
            class_name=class_name,
            expected_algorithm_name=expected_feature_set_name,
        )
    raise ValueError(f"Unknown validation contract_test: {contract_test!r}")


def _build_trading_researcher_feature_set_contract_test(
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
import os
import sys
import types

import pandas as pd
import pytest
import torch

SCRIPT_PATH = os.environ.get("GENERATED_SCRIPT_PATH", {script_path_json})
CLASS_NAME = {class_name_json}
EXPECTED_FEATURE_SET_NAME = {expected_name_json}


def test_implementation_does_not_install_runtime_stubs():
    source = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "class FeatureSetBase" not in source, "Implementation must import FeatureSetBase, not define a local stub"
    assert "sys.modules" not in source, "Implementation must not create fake trading_researcher modules"


class FeatureSetBase:
    def __init__(self, config):
        self.config = config

    def make_column_name(self, name):
        prefix = self.config.get("name", "")
        return f"{{prefix}}_{{name}}" if prefix else str(name)


def is_layer_string_match_in_list(layer_name, patterns):
    return any(str(pattern) in str(layer_name) for pattern in patterns)


def _install_trading_researcher_stubs():
    modules = {{
        "trading_researcher": types.ModuleType("trading_researcher"),
        "trading_researcher.core": types.ModuleType("trading_researcher.core"),
        "trading_researcher.core.modules": types.ModuleType("trading_researcher.core.modules"),
        "trading_researcher.core.modules.feature_sets": types.ModuleType("trading_researcher.core.modules.feature_sets"),
        "trading_researcher.core.modules.feature_sets.feature_set_base": types.ModuleType(
            "trading_researcher.core.modules.feature_sets.feature_set_base"
        ),
        "trading_researcher.core.modules.feature_sets.feature_utils": types.ModuleType(
            "trading_researcher.core.modules.feature_sets.feature_utils"
        ),
    }}
    modules["trading_researcher.core.modules.feature_sets.feature_set_base"].FeatureSetBase = FeatureSetBase
    modules["trading_researcher.core.modules.feature_sets.feature_utils"].is_layer_string_match_in_list = (
        is_layer_string_match_in_list
    )
    sys.modules.update(modules)


def _load_class():
    _install_trading_researcher_stubs()
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


def _build_trading_algorithm_contract_test(
    profile: dict,
    script_path: str,
    class_name: str,
    expected_algorithm_name: str,
) -> str:
    script_path_json = json.dumps(str(Path(script_path).resolve()))
    class_name_json = json.dumps(class_name)
    expected_name_json = json.dumps(expected_algorithm_name)
    return f'''\
import ast
import importlib.util
import os
from pathlib import Path
import sys

PLATFORM_ROOT = os.environ.get("VALIDATION_PLATFORM_ROOT", "")
SCRIPT_PATH = os.environ.get("GENERATED_SCRIPT_PATH", {script_path_json})
CLASS_NAME = {class_name_json}
EXPECTED_ALGORITHM_NAME = {expected_name_json}

if PLATFORM_ROOT and PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData, SignalType


def test_generated_algorithm_imports_real_platform():
    source = Path(SCRIPT_PATH).read_text(encoding="utf-8")
    assert "class Algorithm" not in source, "Implementation must import Algorithm from trading.core.algorithm"
    assert "sys.path" not in source, "Implementation must not mutate sys.path"
    assert "sys.modules" not in source, "Implementation must not inject fake trading modules"
    assert "SignalType.EXIT" not in source, "Algorithms must emit directional signals only; exits belong to the portfolio"
    assert "open_positions" not in source, "Algorithms must not track portfolio position state"
    assert "hold_time_minutes" not in source, "Algorithms must not implement holding-period logic"


def _parse_source_tree():
    return ast.parse(Path(SCRIPT_PATH).read_text(encoding="utf-8"))


def _algorithm_class_def():
    tree = _parse_source_tree()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            return node
    raise AssertionError(f"Class {{CLASS_NAME}} definition not found in source")


def _method_node(class_node, method_name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def _calls_super_reconfigure(method_node):
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "reconfigure":
            continue
        owner = func.value
        if not isinstance(owner, ast.Call):
            continue
        if isinstance(owner.func, ast.Name) and owner.func.id == "super":
            return True
    return False


def _cfg_leaf_synced_attrs(init_node):
    attrs = []
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "self":
                continue
            attr_name = target.attr
            if attr_name.startswith("_"):
                continue
            leaf_names = set()
            for child in ast.walk(node.value):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "get":
                    if child.args and isinstance(child.args[0], ast.Constant) and isinstance(child.args[0].value, str):
                        leaf_names.add(child.args[0].value)
                if isinstance(child, ast.Subscript):
                    key = child.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        leaf_names.add(key.value)
            if attr_name in leaf_names:
                attrs.append(attr_name)
    return sorted(set(attrs))


def _load_class():
    spec = importlib.util.spec_from_file_location("generated_trading_algorithm_contract", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, CLASS_NAME, None)
    assert cls is not None, f"Class {{CLASS_NAME}} not found"
    return cls


def _price(ts_index, close_price):
    minute = 30 + ts_index
    return PriceData(
        symbol="SPY",
        timestamp=f"2024-01-01 09:{{minute:02d}}:00",
        open=float(close_price) - 0.5,
        high=float(close_price) + 0.5,
        low=float(close_price) - 1.0,
        close=float(close_price),
        volume=1000.0 + ts_index,
    )


def _instantiate(cls):
    cfg = {{
        "symbol": "SPY",
        "history_length": 20,
        "name": EXPECTED_ALGORITHM_NAME,
    }}
    try:
        return cls(cfg=cfg, history_length=20)
    except TypeError:
        return cls(cfg)


def _signal_signature(signal):
    return {{
        "type": signal.type.name,
        "symbol": signal.symbol,
        "strength": signal.strength,
    }}


def test_generated_algorithm_is_algorithm_subclass():
    cls = _load_class()
    assert issubclass(cls, Algorithm)


def test_generated_algorithm_crucible_metadata_contract():
    cls = _load_class()
    metadata = getattr(cls, "crucible_metadata", None)
    assert isinstance(metadata, dict), "Generated trading algorithms must define class-level crucible_metadata"
    assert metadata.get("schema_version") == 1
    assert metadata.get("role") == "algorithm"
    assert isinstance(metadata.get("tunables"), dict)
    assert isinstance(metadata.get("fixed_parameters"), list)
    assert isinstance(metadata.get("required_symbols"), list)
    assert isinstance(metadata.get("required_timeframes"), list)
    assert isinstance(metadata.get("required_fields"), list)
    assert isinstance(metadata.get("signal_contract"), dict)
    assert isinstance(metadata.get("statefulness"), dict)
    assert isinstance(metadata.get("dependencies"), list)


def test_generated_algorithm_reconfigure_contract():
    class_node = _algorithm_class_def()
    init_node = _method_node(class_node, "__init__")
    assert init_node is not None, (
        "Generated trading algorithms must implement __init__(self, cfg=None, history_length=0) "
        "and cache tunables on same-named instance attributes for reconfigure()."
    )

    source = Path(SCRIPT_PATH).read_text(encoding="utf-8")
    assert "super().__init__(" in source, "Algorithm __init__ must delegate to Algorithm.__init__"

    attr_names = _cfg_leaf_synced_attrs(init_node)
    assert attr_names, (
        "Algorithm __init__ must cache at least one tunable cfg leaf onto a same-named public instance "
        "attribute so Algorithm.reconfigure() can auto-sync it."
    )

    reconfigure_node = _method_node(class_node, "reconfigure")
    if reconfigure_node is not None:
        assert _calls_super_reconfigure(reconfigure_node), (
            "Custom reconfigure() must call super().reconfigure(new_params) before any subclass-specific rebuild logic."
        )

    cls = _load_class()
    algo = _instantiate(cls)
    for attr_name in attr_names:
        before = getattr(algo, attr_name)
        if isinstance(before, bool):
            updated = not before
        elif isinstance(before, int):
            updated = before + 1
        elif isinstance(before, float):
            updated = before + 1.0
        elif isinstance(before, str):
            updated = before + "_updated"
        else:
            continue
        algo.reconfigure({{attr_name: updated}})
        assert getattr(algo, attr_name) == updated, (
            f"Algorithm.reconfigure() did not update self.{{attr_name}} from cfg leaf '{{attr_name}}'"
        )


def test_generated_algorithm_produces_valid_signals_without_future_leakage():
    cls = _load_class()
    algo_a = _instantiate(cls)
    algo_b = _instantiate(cls)

    prices = [100 + idx * 0.4 for idx in range(30)]
    outputs_a = []
    outputs_b = []

    for idx, price in enumerate(prices):
        tick_a = [_price(idx, price)]
        alt_price = price if idx < len(prices) - 1 else price + 50.0
        tick_b = [_price(idx, alt_price)]
        outputs_a.append(algo_a.on_data(tick_a))
        outputs_b.append(algo_b.on_data(tick_b))

    for result in outputs_a[-5:]:
        assert isinstance(result, list)
        for signal in result:
            assert isinstance(signal, MarketSignal)
            assert signal.symbol == "SPY"
            assert signal.type in {{SignalType.BUY, SignalType.SELL}}
            assert isinstance(signal.strength, int)
            assert 0 <= signal.strength <= 100

    signatures_a = [[_signal_signature(item) for item in result] for result in outputs_a[:-1]]
    signatures_b = [[_signal_signature(item) for item in result] for result in outputs_b[:-1]]
    assert signatures_a == signatures_b, "Signals before the final bar must not depend on unseen future bars"
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

