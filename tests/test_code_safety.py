"""Tests for LLM-generated Python source guards."""
from __future__ import annotations

import pytest

from graph.nodes.code_safety import (
    CodeSafetyError,
    extract_python_source,
    validate_python_source,
)


def test_extract_python_source_uses_only_fenced_block():
    response = "Here is the fix:\n```python\nclass MyClass:\n    pass\n```\nDone."

    assert extract_python_source(response) == "class MyClass:\n    pass"


def test_validate_python_source_rejects_prose():
    with pytest.raises(CodeSafetyError, match="valid Python"):
        validate_python_source("Looking at the error, the class should be fixed.")


def test_validate_python_source_rejects_markdown_fences():
    with pytest.raises(CodeSafetyError, match="markdown"):
        validate_python_source("```python\nclass MyClass:\n    pass\n```")


def test_validate_python_source_requires_expected_class():
    with pytest.raises(CodeSafetyError, match="Expected class"):
        validate_python_source("class OtherClass:\n    pass\n", expected_class_name="MyClass")


def test_validate_python_source_accepts_expected_class():
    validate_python_source("class MyClass:\n    pass\n", expected_class_name="MyClass")
