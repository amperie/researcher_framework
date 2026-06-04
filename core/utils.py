"""Small shared utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def extract_json_array(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    first_array_error: json.JSONDecodeError | None = None

    for idx, char in enumerate(text or ""):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError as exc:
            first_array_error = first_array_error or exc
            salvaged = _salvage_object_array(text[idx:], decoder)
            if salvaged:
                return salvaged
            continue
        if isinstance(value, list):
            return value

    if first_array_error:
        raise ValueError(f"Unmatched '[' in LLM response: {text!r}") from first_array_error
    raise ValueError(f"No JSON array found in LLM response: {text!r}")


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    first_object_error: json.JSONDecodeError | None = None

    for idx, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError as exc:
            first_object_error = first_object_error or exc
            continue
        if isinstance(value, dict):
            return value

    if first_object_error:
        raise ValueError(f"Unmatched '{{' in LLM response: {text!r}") from first_object_error
    raise ValueError(f"No JSON object found in LLM response: {text!r}")


def load_yaml_section(path: str | Path, section: str, default: Any = None) -> Any:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data.get(section, default)


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _salvage_object_array(text: str, decoder: json.JSONDecoder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    idx = 1
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text) or text[idx] == "]":
            break
        if text[idx] != "{":
            idx += 1
            continue
        try:
            value, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            break
        if isinstance(value, dict) and _is_array_item_boundary(text[idx + end :]):
            items.append(value)
            idx += end
            continue
        idx += 1
    return items


def _is_array_item_boundary(tail: str) -> bool:
    stripped = tail.lstrip()
    return not stripped or stripped[0] in ",]"
