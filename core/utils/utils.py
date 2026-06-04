"""Shared utility functions used across the pipeline."""
from __future__ import annotations

import json
import sys
from typing import Any


def extract_json_array(text: str) -> list:
    """Extract the first parseable JSON array found in *text*.

    If the array is truncated (unmatched '['), salvages complete JSON objects
    from within the partial array rather than raising.
    """
    try:
        result = _extract_json_block(text, opener="[", closer="]", expected_type=list)
        if not isinstance(result, list):
            raise ValueError(f"No JSON array found in LLM response: {text!r}")
        return result
    except ValueError as exc:
        if "Unmatched '['" not in str(exc):
            raise
        # Response was truncated — recover any complete top-level objects inside the array.
        recovered = _recover_objects_from_truncated_array(text)
        if recovered:
            import logging
            logging.getLogger(__name__).warning(
                "extract_json_array: truncated response; recovered %d complete item(s)", len(recovered)
            )
            return recovered
        raise


def _recover_objects_from_truncated_array(text: str) -> list:
    """Return a list of complete JSON objects parsed from a truncated array string."""
    # Find the opening '[' of the outermost array.
    array_start = text.find("[")
    if array_start == -1:
        return []
    items: list = []
    pos = array_start + 1
    n = len(text)
    while pos < n:
        # Skip whitespace and commas between items.
        while pos < n and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= n or text[pos] != "{":
            break
        # Scan for the matching closing '}'.
        depth = 0
        in_string = False
        escape_next = False
        end = pos
        for j in range(pos, n):
            ch = text[j]
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
                    end = j
                    break
        else:
            break  # Object was not closed — truncated; stop here.
        candidate = text[pos: end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            break
        pos = end + 1
    return items


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
