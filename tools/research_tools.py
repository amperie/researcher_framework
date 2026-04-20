"""Profile-driven research collection tools.

Each function accepts the same signature so profile YAML can choose which tools
the research node runs:

    collect_x(direction: str, profile: dict, tool_cfg: dict, state: dict) -> list[dict]

Returned records are normalized research artifacts. The research node scores all
artifacts before deciding what to include in summaries and downstream context.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from plugins.loader import adapter_has, load_adapter
from tools.arxiv_tool import search_arxiv
from tools.chroma_tool import ChromaStore
from utils.logger import get_logger

log = get_logger(__name__)


def collect_arxiv(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect arXiv paper metadata as research artifacts."""
    research_cfg = profile.get("research") or {}
    domain_context = research_cfg.get("domain_context", "")
    max_results = int(tool_cfg.get("max_results", 20))
    query = tool_cfg.get("query") or f"{direction} {domain_context}"[:300]

    papers = search_arxiv(query, max_results)
    artifacts: list[dict[str, Any]] = []
    for paper in papers:
        artifacts.append({
            "artifact_id": f"arxiv:{paper.get('arxiv_id', paper.get('url', 'unknown'))}",
            "source": tool_cfg.get("name", "arxiv"),
            "source_type": "paper",
            "title": paper.get("title", ""),
            "summary": paper.get("abstract", ""),
            "url": paper.get("url", ""),
            "published": paper.get("published", ""),
            "metadata": {
                "arxiv_id": paper.get("arxiv_id", ""),
                "categories": tool_cfg.get("categories", []),
            },
            "raw": paper,
        })
    return artifacts


def collect_prior_experiments(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Retrieve semantically similar prior experiment records from ChromaDB."""
    storage_cfg = profile.get("storage") or {}
    collection = tool_cfg.get("collection") or storage_cfg.get("chroma_collection")
    n_results = int(tool_cfg.get("n_results", 5))

    store = ChromaStore(collection_name=collection)
    records = store.query_similar(direction, n_results=n_results)
    artifacts = []
    for record in records:
        artifacts.append({
            "artifact_id": f"prior_experiment:{record.get('id', 'unknown')}",
            "source": tool_cfg.get("name", "prior_experiments"),
            "source_type": "prior_experiment",
            "title": record.get("metadata", {}).get("proposal_name") or record.get("id", "prior experiment"),
            "summary": record.get("document", ""),
            "metadata": {
                **(record.get("metadata") or {}),
                "distance": record.get("distance"),
            },
            "raw": record,
        })
    return artifacts


def collect_adapter_context(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ask the active domain adapter for environment and domain context."""
    adapter = load_adapter(profile)
    records: list[dict[str, Any]] = []

    if adapter_has(adapter, "validate_environment"):
        env = adapter.validate_environment(profile)
        records.append({
            "artifact_id": f"{profile.get('name', 'profile')}:adapter_environment",
            "source": tool_cfg.get("name", "adapter_context"),
            "source_type": "environment",
            "title": "Adapter environment",
            "summary": _compact_mapping(env),
            "metadata": env,
            "raw": env,
        })

    if adapter_has(adapter, "build_context"):
        context = adapter.build_context(profile, state)
        records.append({
            "artifact_id": f"{profile.get('name', 'profile')}:adapter_context",
            "source": tool_cfg.get("name", "adapter_context"),
            "source_type": "platform_context",
            "title": "Domain adapter context",
            "summary": _compact_mapping(context),
            "metadata": {"keys": list(context.keys())},
            "raw": context,
        })

    return records


def collect_profile_context(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expose selected profile sections as scoreable research artifacts."""
    include = tool_cfg.get("include") or [
        "base_classes",
        "datasets",
        "data_sources",
        "risk_constraints",
        "evaluation",
    ]
    artifacts = []
    for key in include:
        value = profile.get(key)
        if value in (None, [], {}):
            continue
        artifacts.append({
            "artifact_id": f"profile:{key}",
            "source": tool_cfg.get("name", "profile_context"),
            "source_type": "profile_context",
            "title": f"Profile section: {key}",
            "summary": _compact_mapping(value),
            "metadata": {"section": key},
            "raw": value,
        })
    return artifacts


def collect_strategy_library(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Inspect a local platform source tree for strategy-related files."""
    platform = profile.get("platform") or {}
    root = Path(tool_cfg.get("path") or platform.get("source_path") or "").expanduser()
    if not root.exists():
        return [{
            "artifact_id": "strategy_library:missing",
            "source": tool_cfg.get("name", "strategy_library"),
            "source_type": "platform_inventory",
            "title": "Strategy library path not found",
            "summary": f"Configured strategy library path does not exist: {root}",
            "metadata": {"path": str(root), "exists": False},
            "raw": {},
        }]

    patterns = tool_cfg.get("patterns") or ["*strategy*.py", "*signal*.py", "*backtest*.py"]
    max_files = int(tool_cfg.get("max_files", 50))
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    files = sorted(set(files))[:max_files]

    rel_files = [str(p.relative_to(root)) for p in files if p.is_file()]
    return [{
        "artifact_id": "strategy_library:file_inventory",
        "source": tool_cfg.get("name", "strategy_library"),
        "source_type": "platform_inventory",
        "title": "Strategy and backtest file inventory",
        "summary": "\n".join(rel_files) if rel_files else "No matching strategy/backtest files found.",
        "metadata": {"path": str(root), "file_count": len(rel_files), "patterns": patterns},
        "raw": {"files": rel_files},
    }]


def load_research_tool(path: str):
    """Load a collector function from a dotted path."""
    module_name, _, func_name = path.rpartition(".")
    if not module_name or not func_name:
        raise ValueError(f"Invalid research tool path: {path!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"Research tool is not callable: {path}")
    return fn


def _compact_mapping(value: Any, max_chars: int = 3000) -> str:
    text = repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text

