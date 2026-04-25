"""Research step - collect, score, digest, and summarize research artifacts.

The active profile controls which research tools run. Each tool returns
artifacts, and this node uses the profile scoring prompt to grade every input
before selecting the highest-value context for downstream steps.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.tools.arxiv_tool import load_cached_digest, save_digest, download_paper_text
from core.tools.research_tools import collect_arxiv, load_research_tool
from core.utils import extract_json_object
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)


def research_node(state: ResearchState, profile: dict) -> dict:
    direction = state.get("research_direction", "")
    if not direction:
        return {"errors": (state.get("errors") or []) + ["research: no research_direction in state"]}

    research_cfg = profile.get("research") or {}
    tool_cfgs = _research_tool_configs(research_cfg)
    if not tool_cfgs:
        log.warning("research_node | No research tools configured")
        return {"research_artifacts": [], "research_papers": [], "research_summary": "", "paper_digests": []}

    artifacts, collection_errors = _collect_artifacts(direction, profile, state, tool_cfgs)
    if not artifacts:
        log.warning("research_node | No artifacts found for direction=%r", direction)
        return {
            "research_artifacts": [],
            "research_papers": [],
            "research_summary": "",
            "paper_digests": [],
            "errors": (state.get("errors") or []) + collection_errors,
        }

    log.info("research_node | Collected %d artifacts from %d tools", len(artifacts), len(tool_cfgs))

    llm = get_llm("research", profile)
    scored = _score_artifacts(direction, profile, artifacts, tool_cfgs, llm)
    selected = [
        a for a in scored
        if int(a.get("relevance_score", 0)) >= int(a.get("score_threshold", 6))
    ]
    selected.sort(key=lambda a: int(a.get("relevance_score", 0)), reverse=True)
    log.info("research_node | %d/%d artifacts passed configured thresholds", len(selected), len(scored))

    summary = _summarize_artifacts(direction, profile, selected, llm)
    paper_digests = _digest_top_papers(profile, selected, tool_cfgs, llm)

    research_papers = [
        _paper_compat(a)
        for a in selected
        if a.get("source_type") == "paper"
    ]

    return {
        "research_artifacts": selected,
        "research_papers": research_papers,
        "research_summary": summary,
        "paper_digests": paper_digests,
        "errors": (state.get("errors") or []) + collection_errors,
    }


def _research_tool_configs(research_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured tools, accepting legacy `sources` as arXiv tools."""
    tools = list(research_cfg.get("tools") or [])
    if tools:
        return tools

    legacy_sources = research_cfg.get("sources") or []
    converted = []
    for source in legacy_sources:
        if source.get("type", "arxiv") == "arxiv":
            converted.append({
                **source,
                "name": source.get("name", "arxiv"),
                "tool": "tools.research_tools.collect_arxiv",
            })
    return converted


def _collect_artifacts(
    direction: str,
    profile: dict,
    state: ResearchState,
    tool_cfgs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []

    for tool_cfg in tool_cfgs:
        tool_path = tool_cfg.get("tool")
        name = tool_cfg.get("name") or tool_path or "unknown"
        try:
            collector = load_research_tool(tool_path) if tool_path else collect_arxiv
            records = collector(direction, profile, tool_cfg, state)
            for record in records or []:
                normalized = _normalize_artifact(record, tool_cfg)
                artifacts.append(normalized)
            log.info("research_node | Tool %r returned %d artifacts", name, len(records or []))
        except Exception as exc:
            log.warning("research_node | Tool %r failed: %s", name, exc, exc_info=True)
            errors.append(f"research tool {name} failed: {exc}")

    deduped: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        deduped[artifact["artifact_id"]] = artifact
    return list(deduped.values()), errors


def _normalize_artifact(record: dict[str, Any], tool_cfg: dict[str, Any]) -> dict[str, Any]:
    artifact_id = record.get("artifact_id") or f"{record.get('source', tool_cfg.get('name', 'tool'))}:{record.get('title', id(record))}"
    return {
        "artifact_id": str(artifact_id),
        "source": record.get("source") or tool_cfg.get("name", "unknown"),
        "source_type": record.get("source_type") or tool_cfg.get("type", "unknown"),
        "title": record.get("title", ""),
        "summary": record.get("summary", ""),
        "url": record.get("url", ""),
        "published": record.get("published", ""),
        "metadata": record.get("metadata") or {},
        "raw": record.get("raw", record),
        "score_threshold": int(tool_cfg.get("relevance_score_threshold", tool_cfg.get("score_threshold", 6))),
    }


def _score_artifacts(
    direction: str,
    profile: dict,
    artifacts: list[dict[str, Any]],
    tool_cfgs: list[dict[str, Any]],
    llm,
) -> list[dict[str, Any]]:
    score_prompt = _get_research_prompt(profile, "artifact_score_system", "score_system")
    max_to_score = (profile.get("research") or {}).get("max_artifacts_to_score")
    artifacts_to_score = artifacts[:int(max_to_score)] if max_to_score else artifacts

    scored: list[dict[str, Any]] = []
    for artifact in artifacts_to_score:
        try:
            resp = llm.invoke([
                SystemMessage(content=score_prompt),
                HumanMessage(content=_score_request(direction, artifact)),
            ])
            result = extract_json_object(resp.content)
            artifact["relevance_score"] = int(result.get("score", 0))
            artifact["relevance_reason"] = result.get("reason", "")
            if "usefulness" in result:
                artifact["usefulness"] = result.get("usefulness")
            if "risks" in result:
                artifact["risks"] = result.get("risks")
        except Exception as exc:
            log.warning("research_node | Scoring failed for %r: %s", artifact.get("title"), exc)
            artifact["relevance_score"] = 0
            artifact["relevance_reason"] = f"Scoring failed: {exc}"
        scored.append(artifact)

    return scored


def _score_request(direction: str, artifact: dict[str, Any]) -> str:
    payload = {
        "research_direction": direction,
        "artifact": {
            "source": artifact.get("source"),
            "source_type": artifact.get("source_type"),
            "title": artifact.get("title"),
            "summary": artifact.get("summary"),
            "metadata": artifact.get("metadata"),
            "url": artifact.get("url"),
            "published": artifact.get("published"),
        },
    }
    return json.dumps(payload, indent=2, default=str)[:10000]

def _summarize_artifacts(direction: str, profile: dict, artifacts: list[dict[str, Any]], llm) -> str:
    if not artifacts:
        return ""

    summary_prompt = _get_research_prompt(profile, "artifact_summary_system", "summary_system")
    max_summary_items = int((profile.get("research") or {}).get("max_summary_artifacts", 12))
    artifact_text = "\n\n".join(
        (
            f"[{a.get('source_type')} | {a.get('source')} | score {a.get('relevance_score')}]\n"
            f"Title: {a.get('title')}\n"
            f"Reason: {a.get('relevance_reason', '')}\n"
            f"Summary: {a.get('summary', '')}"
        )
        for a in artifacts[:max_summary_items]
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=summary_prompt),
            HumanMessage(content=f"Research direction: {direction}\n\nTop scored artifacts:\n{artifact_text}"),
        ])
        log.info("research_node | Summary generated (%d chars)", len(resp.content))
        return resp.content
    except Exception as exc:
        log.error("research_node | Summary generation failed: %s", exc, exc_info=True)
        return ""


def _digest_top_papers(profile: dict, artifacts: list[dict[str, Any]], tool_cfgs: list[dict[str, Any]], llm) -> list[dict]:
    paper_artifacts = [a for a in artifacts if a.get("source_type") == "paper"]
    max_digest = max((int(t.get("max_papers_to_digest", 0)) for t in tool_cfgs), default=0)
    if not paper_artifacts or max_digest <= 0:
        return []

    digest_prompt = get_prompt(profile, "research", "digest_system")
    digests: list[dict] = []

    for artifact in paper_artifacts[:max_digest]:
        arxiv_id = (artifact.get("metadata") or {}).get("arxiv_id", "")
        cached = load_cached_digest(arxiv_id) if arxiv_id else None
        if cached:
            digests.append(cached)
            continue

        text = download_paper_text(arxiv_id) if arxiv_id else None
        if not text:
            log.warning("research_node | No full text for %s - skipping digest", arxiv_id)
            continue

        try:
            resp = llm.invoke([
                SystemMessage(content=digest_prompt),
                HumanMessage(content=f"Paper: {artifact.get('title')}\n\n{text[:8000]}"),
            ])
            record = {
                "arxiv_id": arxiv_id,
                "title": artifact.get("title", ""),
                "published": artifact.get("published", ""),
                "abstract": artifact.get("summary", ""),
                "digest": resp.content,
            }
            save_digest(arxiv_id, record)
            digests.append(record)
        except Exception as exc:
            log.warning("research_node | Digest failed for %s: %s", arxiv_id, exc)

    return digests


def _get_research_prompt(profile: dict, preferred_key: str, fallback_key: str) -> str:
    try:
        return get_prompt(profile, "research", preferred_key)
    except KeyError:
        return get_prompt(profile, "research", fallback_key)


def _paper_compat(artifact: dict[str, Any]) -> dict:
    raw = artifact.get("raw") or {}
    metadata = artifact.get("metadata") or {}
    return {
        "title": artifact.get("title", ""),
        "abstract": artifact.get("summary", ""),
        "url": artifact.get("url", ""),
        "arxiv_id": metadata.get("arxiv_id", raw.get("arxiv_id", "")),
        "published": artifact.get("published", ""),
        "relevance_score": artifact.get("relevance_score", 0),
        "relevance_reason": artifact.get("relevance_reason", ""),
    }
