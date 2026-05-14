from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


RESEARCH_TOOL_CATALOG_DIR = Path("configs/research_tools")


class ResearchToolCatalogError(ValueError):
    pass


def load_research_tool_catalog() -> dict[str, dict[str, Any]]:
    if not RESEARCH_TOOL_CATALOG_DIR.exists():
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(RESEARCH_TOOL_CATALOG_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ResearchToolCatalogError(f"Research tool catalog must be a mapping: {path}")
        tools = raw.get("tools") or {}
        if not isinstance(tools, dict):
            raise ResearchToolCatalogError(f"Research tool catalog 'tools' must be a mapping: {path}")
        for ref_name, value in tools.items():
            if not isinstance(value, dict):
                raise ResearchToolCatalogError(f"Research tool catalog entry must be a mapping: {path} ref={ref_name!r}")
            catalog[str(ref_name)] = deepcopy(value)
    return catalog


def resolve_research_tool_refs(
    tools: list[Any],
    *,
    path_key: str,
) -> list[dict[str, Any]]:
    catalog = load_research_tool_catalog()
    resolved: list[dict[str, Any]] = []
    for item in tools or []:
        resolved.append(_resolve_one_tool(item, catalog=catalog, path_key=path_key))
    return resolved


def _resolve_one_tool(
    item: Any,
    *,
    catalog: dict[str, dict[str, Any]],
    path_key: str,
) -> dict[str, Any]:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise ResearchToolCatalogError("Research tool path cannot be empty")
        return {
            path_key: text,
            "name": text.rsplit(".", 1)[-1],
            "enabled": True,
        }
    if not isinstance(item, dict):
        raise ResearchToolCatalogError(f"Invalid research tool entry: {item!r}")

    ref_name = str(item.get("ref") or "").strip()
    if ref_name:
        base = deepcopy(catalog.get(ref_name) or {})
        if not base:
            raise ResearchToolCatalogError(f"Unknown research tool ref: {ref_name}")
        merged = _deep_merge(base, {k: v for k, v in item.items() if k != "ref"})
    else:
        merged = deepcopy(item)

    path = str(merged.get(path_key) or "").strip()
    if not path:
        raise ResearchToolCatalogError(f"Research tool missing {path_key}: {item!r}")
    merged[path_key] = path
    merged["name"] = str(merged.get("name") or path.rsplit(".", 1)[-1]).strip()
    merged["enabled"] = bool(merged.get("enabled", True))
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
