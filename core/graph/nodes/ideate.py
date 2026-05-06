"""Ideate step - brainstorm experiment ideas grounded in the research.

Reads:
    state['research_summary']
    state['paper_digests']    - preferred; falls back to research_papers abstracts

Writes:
    state['ideas']  - list of raw brainstorm items
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from core.graph.nodes.memory import persist_memory_records_for_state
from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils import extract_json_array
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)
_RAW_RESPONSE_LOG_LIMIT = 12000


def ideate_node(state: ResearchState, profile: dict) -> dict:
    direction = state.get("research_direction", "")
    summary = state.get("research_summary", "")
    digests = state.get("paper_digests") or []
    papers = state.get("research_papers") or []
    artifacts = state.get("research_artifacts") or []

    system_prompt = get_prompt(profile, "ideate")
    llm = get_llm("ideate", profile)
    ideate_cfg = (profile.get("llm") or {}).get("prompt_context") or {}
    max_digest_items = int(ideate_cfg.get("ideate_max_digests", 2) or 2)
    max_artifact_items = int(ideate_cfg.get("ideate_max_artifacts", 5) or 5)

    context_parts = [f"Research direction: {direction}", "", f"Research summary:\n{summary}"]
    if digests:
        context_parts.append("\nPaper digests:")
        for digest in digests[:max_digest_items]:
            context_parts.append(f"\n[{digest['title']}]\n{_truncate_text(digest['digest'], 1800)}")
    elif papers:
        context_parts.append("\nRelevant papers (abstracts):")
        for paper in papers[:5]:
            context_parts.append(
                f"\n[{paper['title']} - score {paper.get('relevance_score', '?')}]\n"
                f"{_truncate_text(paper['abstract'], 900)}"
            )

    if artifacts:
        context_parts.append("\nTop scored research artifacts:")
        for artifact in artifacts[:max_artifact_items]:
            context_parts.append(
                f"\n[{artifact.get('source_type')} | {artifact.get('source')} | score {artifact.get('relevance_score', '?')}]\n"
                f"{artifact.get('title', '')}\n{_truncate_text(artifact.get('summary', ''), 700)}\n"
                f"Usefulness: {_truncate_text(artifact.get('usefulness', ''), 220)}\n"
                f"Risks: {_truncate_text(artifact.get('risks', ''), 220)}"
            )

    context = "\n".join(context_parts)

    log.info("ideate_node | Generating ideas - direction=%r, digests=%d", direction, len(digests))
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ])
        ideas = extract_json_array(resp.content)
        log.info("ideate_node | Generated %d ideas", len(ideas))
        for i, idea in enumerate(ideas):
            log.debug("  Idea %d: %s", i + 1, idea.get("name"))
    except Exception as exc:
        log.error("ideate_node | Failed: %s", exc, exc_info=True)
        if "resp" in locals():
            log.error(
                "ideate_node | Raw LLM response follows:\n%s",
                _truncate_text(str(getattr(resp, "content", "") or ""), _RAW_RESPONSE_LOG_LIMIT),
            )
        return {"ideas": [], "errors": (state.get("errors") or []) + [f"ideate failed: {exc}"]}

    delta = {"ideas": ideas}
    try:
        persist_memory_records_for_state(profile, {**state, **delta})
    except Exception as exc:
        log.warning("ideate_node | Memory persistence failed: %s", exc)
        delta["errors"] = (state.get("errors") or []) + [f"ideate: memory persistence failed: {exc}"]
    return delta


def _truncate_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
