"""Shared utility functions used across the pipeline and dev tooling."""
from __future__ import annotations

import json
import re
import sys
from typing import Any


def extract_json_array(text: str) -> list:
    """Extract the first JSON array found in *text*, tolerating markdown fences.

    Uses bracket counting to find the outermost `[...]` so that nested arrays
    inside the payload are not mistaken for the end of the outer array.

    Args:
        text: Raw LLM response that may contain a JSON array, optionally
              wrapped in triple-backtick fences.

    Returns:
        Parsed list.

    Raises:
        ValueError: If no JSON array is found or the array cannot be parsed.
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


def load_yaml_section(section: str, config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load a named top-level section from the project YAML config file.

    Args:
        section:     Top-level key to extract (e.g. ``'research_node'``).
        config_path: Path to the YAML file, relative to the working directory.
                     Defaults to ``configs/config.yaml``.

    Returns:
        The section dict, or an empty dict if the file is missing, the key is
        absent, or parsing fails — so callers always get a usable value.
    """
    try:
        import yaml  # pyyaml — ships as a transitive dep of mlflow

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
    except Exception as exc:  # noqa: BLE001
        print(f"[utils] Failed to parse {config_path!r}: {exc}; using defaults.", file=sys.stderr)
        return {}


def fmt_value(v) -> str:
    """Return a compact human-readable summary of *v* for display purposes.

    Strings  → '<n> chars'
    Lists    → '<n> items'
    Dicts    → '<n> keys'
    Anything else → repr(v)
    """
    if isinstance(v, str):
        return f"{len(v)} chars"
    if isinstance(v, list):
        return f"{len(v)} items"
    if isinstance(v, dict):
        return f"{len(v)} keys"
    return repr(v)
