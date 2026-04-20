"""Ideate step — brainstorm experiment ideas grounded in the research.

Reads:
    state['research_summary']
    state['paper_digests']    — preferred; falls back to research_papers abstracts

Writes:
    state['ideas']  — list of raw brainstorm items
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ResearchState
from llm.factory import get_llm
from utils import extract_json_array
from utils.logger import get_logger
from utils.profile_loader import get_prompt

log = get_logger(__name__)


def ideate_node(state: ResearchState, profile: dict) -> dict:
    direction = state.get("research_direction", "")
    summary = state.get("research_summary", "")
    digests = state.get("paper_digests") or []
    papers = state.get("research_papers") or []
    artifacts = state.get("research_artifacts") or []

    system_prompt = get_prompt(profile, "ideate")
    llm = get_llm("ideate", profile)

    context_parts = [f"Research direction: {direction}", "", f"Research summary:\n{summary}"]
    if digests:
        context_parts.append("\nPaper digests:")
        for d in digests:
            context_parts.append(f"\n[{d['title']}]\n{d['digest']}")
    elif papers:
        context_parts.append("\nRelevant papers (abstracts):")
        for p in papers[:5]:
            context_parts.append(f"\n[{p['title']} — score {p.get('relevance_score', '?')}]\n{p['abstract']}")

    if artifacts:
        context_parts.append("\nTop scored research artifacts:")
        for a in artifacts[:10]:
            context_parts.append(
                f"\n[{a.get('source_type')} | {a.get('source')} | score {a.get('relevance_score', '?')}]\n"
                f"{a.get('title', '')}\n{a.get('summary', '')}\n"
                f"Usefulness: {a.get('usefulness', '')}\nRisks: {a.get('risks', '')}"
            )

    context = "\n".join(context_parts)

    log.info("ideate_node | Generating ideas — direction=%r, digests=%d", direction, len(digests))
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
        return {"ideas": [], "errors": (state.get("errors") or []) + [f"ideate failed: {exc}"]}

    return {"ideas": ideas}
