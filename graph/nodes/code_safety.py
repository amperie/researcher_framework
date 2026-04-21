"""Utilities for accepting only real Python source from LLM code responses."""
from __future__ import annotations

import ast
import re


class CodeSafetyError(ValueError):
    """Raised when an LLM response is not acceptable Python source."""


_FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)\n```",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_python_source(text: str) -> str:
    """Extract Python source from an LLM response.

    If the response contains a fenced code block, only the first fenced block is
    used. This prevents explanatory text before or after the block from being
    written into Python files.
    """
    stripped = text.strip()
    match = _FENCED_CODE_RE.search(stripped)
    if match:
        return match.group("code").strip()

    stripped = re.sub(r"^```(?:python|py)?\s*\n", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


def validate_python_source(source: str, expected_class_name: str | None = None) -> None:
    """Reject prose, markdown, syntax errors, and wrong implementation classes."""
    if not source.strip():
        raise CodeSafetyError("LLM returned empty Python source")

    if "```" in source:
        raise CodeSafetyError("LLM response still contains markdown code fences")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CodeSafetyError(f"LLM response is not valid Python: {exc}") from exc

    if expected_class_name is None:
        return

    class_names = [
        node.name for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    if expected_class_name not in class_names:
        raise CodeSafetyError(
            f"Expected class {expected_class_name!r}; found classes {class_names!r}"
        )
