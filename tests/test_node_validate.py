"""Tests for graph/nodes/validate.py."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from graph.nodes.validate import _run_tests, _strip_fences, validate_node


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

        with patch("graph.nodes.validate.get_llm"):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
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

        with patch("graph.nodes.validate.get_llm"):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
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

        with patch("graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
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

        with patch("graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
                with patch("graph.nodes.validate._run_tests",
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

        with patch("graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
                result = validate_node(
                    {"implementations": impls},
                    profile,
                )

        vr = result["validation_results"][0]
        assert vr["passed"] is False
        assert any("test generation failed" in e for e in result["errors"])


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

        with patch("graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
                with patch("graph.nodes.validate._run_tests",
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
        with patch("graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
                with patch("graph.nodes.validate._run_tests",
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

        with patch("graph.nodes.validate.get_llm", return_value=mock_llm):
            with patch("graph.nodes.validate.get_config", return_value=cfg):
                with patch("graph.nodes.validate._run_tests", return_value="1 failed"):
                    result = validate_node(
                        {"implementations": impls},
                        profile,
                    )

        assert script.read_text(encoding="utf-8") == original_code
        assert result["validation_results"][0]["passed"] is False
        assert "Fix response rejected" in result["validation_results"][0]["test_output"]
