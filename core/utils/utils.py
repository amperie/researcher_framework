"""Shared utility functions used across the pipeline."""
from __future__ import annotations

import json
import sys
from typing import Any


def extract_json_array(text: str) -> list:
    """Extract the first parseable JSON array found in *text*."""
    result = _extract_json_block(text, opener="[", closer="]", expected_type=list)
    if not isinstance(result, list):
        raise ValueError(f"No JSON array found in LLM response: {text!r}")
    return result


def extract_json_object(text: str) -> dict:
    """Extract the first parseable JSON object found in *text*."""
    result = _extract_json_block(text, opener="{", closer="}", expected_type=dict)
    if not isinstance(result, dict):
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return result


def _extract_json_block(text: str, *, opener: str, closer: str, expected_type: type) -> Any:
    starts = [i for i, ch in enumerate(text) if ch == opener]
    if not starts:
        name = "array" if expected_type is list else "object"
        raise ValueError(f"No JSON {name} found in LLM response: {text!r}")

    parse_errors: list[str] = []
    saw_unmatched = False

    for start in starts:
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
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start: i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError as exc:
                        parse_errors.append(str(exc))
                        break
                    if isinstance(parsed, expected_type):
                        return parsed
                    parse_errors.append(f"decoded JSON was {type(parsed).__name__}, expected {expected_type.__name__}")
                    break
        else:
            saw_unmatched = True

    if parse_errors:
        name = "array" if expected_type is list else "object"
        raise ValueError(f"Found JSON {name} bounds but failed to parse: {parse_errors[0]}")
    if saw_unmatched:
        raise ValueError(f"Unmatched {opener!r} in LLM response: {text!r}")

    name = "array" if expected_type is list else "object"
    raise ValueError(f"No JSON {name} found in LLM response: {text!r}")


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
