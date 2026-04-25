"""Shared utility functions used across the pipeline."""
from __future__ import annotations

import json
import sys
from typing import Any


def extract_json_array(text: str) -> list:
    """Extract the first JSON array found in *text*, tolerating markdown fences.

    Uses bracket counting to find the outermost `[...]` so nested arrays
    are not mistaken for the end of the outer array.

    Raises:
        ValueError: If no JSON array is found or it cannot be parsed.
    """
    start = text.find("[")
    if start == -1:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Found JSON array bounds but failed to parse: {exc}"
                    ) from exc

    raise ValueError(f"Unmatched '[' in LLM response: {text!r}")


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object found in *text*, tolerating markdown fences."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Found JSON object bounds but failed to parse: {exc}"
                    ) from exc

    raise ValueError(f"Unmatched '{{' in LLM response: {text!r}")


def load_yaml_section(section: str, config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load a named top-level section from the project YAML config file.

    Returns an empty dict if the file is missing, the key is absent, or parsing fails.
    """
    try:
        import yaml
        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get(section, {})
    except FileNotFoundError:
        print(
            f"[utils] config file not found at {config_path!r}; "
            f"using defaults for section {section!r}.",
            file=sys.stderr,
        )
        return {}
    except Exception as exc:
        print(f"[utils] Failed to parse {config_path!r}: {exc}; using defaults.", file=sys.stderr)
        return {}


def fmt_value(v) -> str:
    """Compact human-readable summary of *v* for display purposes."""
    if isinstance(v, str):
        return f"{len(v)} chars"
    if isinstance(v, list):
        return f"{len(v)} items"
    if isinstance(v, dict):
        return f"{len(v)} keys"
    return repr(v)
