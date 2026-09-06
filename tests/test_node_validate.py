"""Tests for graph/nodes/validate.py."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.graph.nodes.validate import (
    _build_trading_researcher_feature_set_contract_test,
    _preflight_validation_error,
    _run_tests,
    _strip_fences,
    validate_node,
)


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_strips_python_fence(self):
        result = _strip_fences("```python\ncode here\n```")
        assert result == "code here"

    def test_strips_plain_fence(self):
        result = _strip_fences("```\ncode here\n```")
        assert result == "code here"

    def test_no_fence_unchanged(self):
        result = _strip_fences("plain code")
        assert result == "plain code"

    def test_strips_surrounding_whitespace(self):
        result = _strip_fences("  ```\ncode\n```  ")
        assert result == "code"


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------

class TestRunTests:
    def test_returns_stdout_plus_stderr(self):
        mock_result = MagicMock()
        mock_result.stdout = "test output"
        mock_result.stderr = " extra"

        with patch("subprocess.run", return_value=mock_result):
            output = _run_tests("uv run pytest", "test_file.py", 60)

        assert output == "test output extra"

    def test_timeout_returns_timeout_string(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            output = _run_tests("uv run pytest", "test_file.py", 10)

        assert "TIMEOUT" in output

    def test_exception_returns_error_string(self):
        with patch("subprocess.run", side_effect=OSError("binary not found")):
            output = _run_tests("uv run pytest", "test_file.py", 60)

        assert "ERROR" in output

    def test_external_runtime_uses_shared_task_runner(self):
        runtime_spec = {
            "python": "python",
            "cwd": "E:/tmp/runtime",
            "pythonpath_entries": ["E:/tmp/runtime", "E:/repo"],
            "plugin_name": "trading",
            "logger_prefixes": ["core.plugins.trading"],
        }
        with patch("core.graph.nodes.validate.run_task", return_value={"output": "1 passed"}) as call_task:
            output = _run_tests("uv run pytest", "test_file.py", 60, runtime_spec=runtime_spec)

        assert output == "1 passed"
        assert call_task.call_args.args[0]["task_path"] == "core.plugins.framework_tasks.run_tests"
        assert call_task.call_args.args[0]["python"] == "python"
        assert "staged_files" in call_task.call_args.args[0]["payload"]

    def test_external_runtime_receives_absolute_test_path(self, tmp_path):
        runtime_spec = {
            "python": "python",
            "cwd": "E:/tmp/runtime",
            "pythonpath_entries": ["E:/tmp/runtime", "E:/repo"],
            "plugin_name": "trading",
            "logger_prefixes": ["core.plugins.trading"],
        }
        test_file = tmp_path / "tests" / "test_algo.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_ok(): pass\n", encoding="utf-8")

        with patch("core.graph.nodes.validate.run_task", return_value={"output": "1 passed"}) as call_task:
            _run_tests(
                "uv run pytest",
                str(test_file.resolve()),
                60,
                runtime_spec=runtime_spec,
                script_source="class Algo: pass\n",
                test_source="def test_ok(): pass\n",
                script_name="Algo.py",
                test_name="test_algo.py",
            )

        payload = call_task.call_args.args[0]["payload"]
        assert payload["test_path"] == "test_algo.py"
        assert payload["staged_files"]["Algo.py"] == "class Algo: pass\n"
        assert payload["staged_files"]["test_algo.py"] == "def test_ok(): pass\n"


# ---------------------------------------------------------------------------
# _preflight_validation_error
# ---------------------------------------------------------------------------

class TestPreflightValidationError:
    def test_detects_make_column_name_with_multiple_args(self):
        code = """
class MyClass:
    def f(self):
        return self.make_column_name("a", "b")
"""
        error = _preflight_validation_error(code)
        assert "make_column_name accepts exactly one positional string argument" in error

    def test_allows_single_argument_make_column_name(self):
        code = """
class MyClass:
    def f(self):
        return self.make_column_name("a_b")
"""
        assert _preflight_validation_error(code) == ""


class TestTradingResearcherContractTest:
    def test_contract_scan_includes_qk_and_output_layers(self):
        source = _build_trading_researcher_feature_set_contract_test(
            script_path="dummy.py",
            class_name="DummyClass",
            expected_feature_set_name="dummy_feature",
        )
        assert 'model.layers.0.attn.q_proj' in source
        assert 'model.layers.0.attn.k_proj' in source
        assert 'model.layers.0.attn.o_proj' in source


# ---------------------------------------------------------------------------
# validate_node — no implementations
# ---------------------------------------------------------------------------

class TestValidateNodeEmpty:
    def test_empty_implementations_returns_empty(self):
        result = validate_node({}, {"name": "test", "validate": {}, "datasets": [], "prompts": {}})
        assert result == {"validation_results": []}


# ---------------------------------------------------------------------------
# validate_node — missing script
# ---------------------------------------------------------------------------

class TestValidateNodeMissingScript:
    def _profile(self):
        return {
            "name": "test",
            "validate": {"auto_run": True, "max_fix_retries": 1,
                         "test_runner": "uv run pytest",
                         "test_output_dir": "dev/experiments/tests"},
            "datasets": [],
            "prompts": {
                "validate": {"system": "Write tests.", "fix_system": "Fix code."}
            },
        }

    def test_missing_script_path_marked_failed(self, tmp_path):
        impls = [{"class_name": "TestClass", "script_path": ""}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        with patch("core.graph.nodes.validate.get_llm"):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("pathlib.Path.mkdir"):
                    result = validate_node(
                        {"implementations": impls},
                        self._profile(),
                    )

        assert result["validation_results"][0]["passed"] is False
        assert result["validation_results"][0]["class_name"] == "TestClass"

    def test_nonexistent_script_path_marked_failed(self, tmp_path):
        impls = [{"class_name": "TestClass", "script_path": "/nonexistent/path.py"}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        with patch("core.graph.nodes.validate.get_llm"):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("pathlib.Path.mkdir"):
                    result = validate_node(
                        {"implementations": impls},
                        self._profile(),
                    )

        assert result["validation_results"][0]["passed"] is False


# ---------------------------------------------------------------------------
# validate_node — happy path (tests pass on first run)
# ---------------------------------------------------------------------------

class TestValidateNodePass:
    def _profile(self, auto_run=True):
        return {
            "name": "test",
            "validate": {
                "auto_run": auto_run,
                "max_fix_retries": 2,
                "test_runner": "uv run pytest",
                "test_output_dir": "dev/experiments/tests",
            },
            "datasets": [
                {"name": "ds", "available_scan_fields": {"guaranteed": ["f1"]}}
            ],
            "prompts": {
                "validate": {"system": "Write tests.", "fix_system": "Fix code."}
            },
        }

    def test_auto_run_false_skips_execution(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text("class MyClass: pass", encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="def test_x(): pass")

        test_output_dir = tmp_path / "tests"
        profile = self._profile(auto_run=False)
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                result = validate_node(
                    {"implementations": impls},
                    profile,
                )

        vr = result["validation_results"][0]
        assert vr["passed"] is None
        assert vr["test_output"] == "auto_run=False"

    def test_passes_on_first_run(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text("class MyClass: pass", encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="def test_x(): pass")

        test_output_dir = tmp_path / "tests"

        profile = self._profile()
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("core.graph.nodes.validate._run_tests",
                           return_value="1 passed"):
                    result = validate_node(
                        {"implementations": impls},
                        profile,
                    )

        vr = result["validation_results"][0]
        assert vr["passed"] is True
        assert vr["attempts"] == 0

    def test_test_generation_failure_recorded(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text("class MyClass: pass", encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM timeout")

        test_output_dir = tmp_path / "tests"
        profile = self._profile()
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                result = validate_node(
                    {"implementations": impls},
                    profile,
                )

        vr = result["validation_results"][0]
        assert vr["passed"] is False
        assert any("test generation failed" in e for e in result["errors"])

    def test_persists_memory_after_validation(self):
        impls = [{"class_name": "MyClass", "script_path": "fake.py", "proposal_name": "idea_a"}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)
        profile = self._profile(auto_run=False)
        profile["validate"]["test_output_dir"] = "dev/experiments/tests"

        with patch("core.graph.nodes.validate.get_config", return_value=cfg):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.read_text", return_value="class MyClass:\n    pass\n"):
                        with patch(
                            "core.graph.nodes.validate._build_test_code",
                            return_value=("def test_x(): pass", "contract:test"),
                        ):
                            with patch("pathlib.Path.write_text"):
                                with patch("core.graph.nodes.validate.persist_memory_records_for_state") as persist_memory:
                                    result = validate_node({"implementations": impls}, profile)

        assert result["validation_results"][0]["passed"] is None
        persist_memory.assert_called_once()

    def test_validation_registers_artifact_references(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text("class MyClass: pass", encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script), "proposal_name": "idea_a"}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)
        profile = self._profile()
        profile["validate"]["test_output_dir"] = str(tmp_path / "tests")

        artifact_store = MagicMock()
        artifact_store.store_file.side_effect = [
            {"artifact_id": "validation-test-1", "uri": "s3://bucket/test.py"},
            {"artifact_id": "implementation-1", "uri": "s3://bucket/impl.py"},
        ]
        artifact_store.store_json.return_value = {
            "artifact_id": "validation-result-1",
            "uri": "s3://bucket/validation.json",
        }

        with patch("core.graph.nodes.validate.get_config", return_value=cfg):
            with patch(
                "core.graph.nodes.validate._build_test_code",
                return_value=("def test_x(): pass", "contract:test"),
            ):
                with patch("core.graph.nodes.validate._run_tests", return_value="1 passed"):
                    with patch("core.graph.nodes.artifact_refs.get_artifact_store", return_value=artifact_store):
                        result = validate_node({"implementations": impls}, profile)

        vr = result["validation_results"][0]
        updated_impl = result["implementations"][0]
        assert vr["test_file_artifact_id"] == "validation-test-1"
        assert vr["implementation_artifact_id"] == "implementation-1"
        assert vr["stored_artifact_id"] == "validation-result-1"
        assert updated_impl["stored_artifact_id"] == "implementation-1"

    def test_memory_failure_is_non_fatal(self):
        impls = [{"class_name": "MyClass", "script_path": "fake.py"}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)
        profile = self._profile(auto_run=False)
        profile["validate"]["test_output_dir"] = "dev/experiments/tests"

        with patch("core.graph.nodes.validate.get_config", return_value=cfg):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.read_text", return_value="class MyClass:\n    pass\n"):
                        with patch(
                            "core.graph.nodes.validate._build_test_code",
                            return_value=("def test_x(): pass", "contract:test"),
                        ):
                            with patch("pathlib.Path.write_text"):
                                with patch(
                                    "core.graph.nodes.validate.persist_memory_records_for_state",
                                    side_effect=Exception("memory down"),
                                ):
                                    result = validate_node({"implementations": impls}, profile)

        assert result["validation_results"][0]["passed"] is None
        assert any("memory persistence failed" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# validate_node — fix-retry loop
# ---------------------------------------------------------------------------

class TestValidateNodeFixRetry:
    def _profile(self, max_retries=2):
        return {
            "name": "test",
            "validate": {
                "auto_run": True,
                "max_fix_retries": max_retries,
                "test_runner": "uv run pytest",
                "test_output_dir": "dev/experiments/tests",
            },
            "datasets": [],
            "prompts": {
                "validate": {"system": "Write tests.", "fix_system": "Fix code."}
            },
        }

    def test_retries_on_failure_and_eventually_passes(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text("class MyClass: pass", encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        # First invoke: generate tests; subsequent invoke: fix code.
        mock_llm.invoke.side_effect = [
            MagicMock(content="def test_x(): pass"),
            MagicMock(content="class MyClass:\n    pass\n"),
        ]

        test_output_dir = tmp_path / "tests"
        profile = self._profile(max_retries=2)
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        # First run fails, second run passes
        test_outputs = iter(["1 failed", "1 passed"])

        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("core.graph.nodes.validate._run_tests",
                           side_effect=lambda *a, **kw: next(test_outputs)):
                    result = validate_node(
                        {"implementations": impls},
                        profile,
                    )

        vr = result["validation_results"][0]
        assert vr["passed"] is True
        assert vr["attempts"] == 1

    def test_max_retries_reached_marks_failed(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text("class MyClass: pass", encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="def test_x(): pass"),
            MagicMock(content="class MyClass:\n    pass\n"),
            MagicMock(content="class MyClass:\n    pass\n"),
        ]

        test_output_dir = tmp_path / "tests"
        profile = self._profile(max_retries=2)
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        # Always fails
        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("core.graph.nodes.validate._run_tests",
                           return_value="1 failed"):
                    result = validate_node(
                        {"implementations": impls},
                        profile,
                    )

        vr = result["validation_results"][0]
        assert vr["passed"] is False
        assert any("failed after" in e for e in result["errors"])

    def test_rejects_prose_fix_without_overwriting_script(self, tmp_path):
        script = tmp_path / "my_class.py"
        original_code = "class MyClass:\n    pass\n"
        script.write_text(original_code, encoding="utf-8")
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="def test_x(): pass"),
            MagicMock(content="Looking at the failure, you should change the class."),
        ]

        test_output_dir = tmp_path / "tests"
        profile = self._profile(max_retries=1)
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("core.graph.nodes.validate._run_tests", return_value="1 failed"):
                    result = validate_node(
                        {"implementations": impls},
                        profile,
                    )

        assert script.read_text(encoding="utf-8") == original_code
        assert result["validation_results"][0]["passed"] is False
        assert "Fix response rejected" in result["validation_results"][0]["test_output"]

    def test_preflight_make_column_name_error_triggers_fix_loop(self, tmp_path):
        script = tmp_path / "my_class.py"
        script.write_text(
            "class MyClass:\n"
            "    def make_column_name(self, name):\n"
            "        return name\n"
            "    def process_feature_set(self, scan=None):\n"
            "        return self.make_column_name('a', 'b')\n",
            encoding="utf-8",
        )
        impls = [{"class_name": "MyClass", "script_path": str(script)}]
        cfg = SimpleNamespace(validate_timeout_seconds=30)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="def test_x(): pass"),
            MagicMock(
                content=(
                    "class MyClass:\n"
                    "    def make_column_name(self, name):\n"
                    "        return name\n"
                    "    def process_feature_set(self, scan=None):\n"
                    "        return self.make_column_name('a_b')\n"
                )
            ),
        ]

        test_output_dir = tmp_path / "tests"
        profile = self._profile(max_retries=1)
        profile["validate"]["test_output_dir"] = str(test_output_dir)

        with patch("core.graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("core.graph.nodes.validate.get_config", return_value=cfg):
                with patch("core.graph.nodes.validate._run_tests", return_value="1 passed"):
                    result = validate_node(
                        {"implementations": impls},
                        profile,
                    )

        vr = result["validation_results"][0]
        assert vr["passed"] is True
        assert vr["attempts"] == 1

