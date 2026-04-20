"""Research step — search for relevant papers and build a research context.

Reads:
    state['research_direction']

Writes:
    state['research_papers']   — papers scored for relevance
    state['research_summary']  — LLM-synthesised summary
    state['paper_digests']     — structured full-text extractions (cached)
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ResearchState
from llm.factory import get_llm
from tools.arxiv_tool import search_arxiv, load_cached_digest, save_digest, download_paper_text
from utils import extract_json_object
from utils.logger import get_logger
from utils.profile_loader import get_prompt

log = get_logger(__name__)


def research_node(state: ResearchState, profile: dict) -> dict:
    direction = state.get("research_direction", "")
    if not direction:
        return {"errors": (state.get("errors") or []) + ["research: no research_direction in state"]}

    research_cfg = profile.get("research") or {}
    sources = research_cfg.get("sources") or []
    domain_context = research_cfg.get("domain_context", "")

    # --- Gather papers from all configured sources ---
    all_papers: list[dict] = []
    for source in sources:
        src_type = source.get("type", "arxiv")
        if src_type == "arxiv":
            max_results = source.get("max_results", 20)
            query = f"{direction} {domain_context}"[:300]
            papers = search_arxiv(query, max_results)
            all_papers.extend(papers)
        else:
            log.warning("research_node | Unknown source type %r — skipping", src_type)

    if not all_papers:
        log.warning("research_node | No papers found for direction=%r", direction)
        return {"research_papers": [], "research_summary": "", "paper_digests": []}

    log.info("research_node | Fetched %d papers total", len(all_papers))

    # --- Score papers for relevance ---
    score_prompt = get_prompt(profile, "research", "score_system")
    llm = get_llm("research", profile)
    threshold = max(s.get("relevance_score_threshold", 6) for s in sources if s.get("type") == "arxiv") if sources else 6

    scored: list[dict] = []
    for paper in all_papers:
        try:
            resp = llm.invoke([
                SystemMessage(content=score_prompt),
                HumanMessage(content=f"Research direction: {direction}\n\nPaper title: {paper['title']}\n\nAbstract: {paper['abstract']}"),
            ])
            result = extract_json_object(resp.content)
            paper["relevance_score"] = int(result.get("score", 0))
            paper["relevance_reason"] = result.get("reason", "")
        except Exception as exc:
            log.warning("research_node | Scoring failed for %r: %s", paper.get("title"), exc)
            paper["relevance_score"] = 0

        if paper["relevance_score"] >= threshold:
            scored.append(paper)

    scored.sort(key=lambda p: p["relevance_score"], reverse=True)
    log.info("research_node | %d/%d papers passed threshold %d", len(scored), len(all_papers), threshold)

    # --- Research summary ---
    summary = ""
    if scored:
        summary_prompt = get_prompt(profile, "research", "summary_system")
        papers_text = "\n\n".join(
            f"[{p['title']} — score {p['relevance_score']}]\n{p['abstract']}"
            for p in scored[:10]
        )
        try:
            resp = llm.invoke([
                SystemMessage(content=summary_prompt),
                HumanMessage(content=f"Research direction: {direction}\n\nPapers:\n{papers_text}"),
            ])
            summary = resp.content
            log.info("research_node | Summary generated (%d chars)", len(summary))
        except Exception as exc:
            log.error("research_node | Summary generation failed: %s", exc, exc_info=True)

    # --- Digest top papers ---
    max_digest = max((s.get("max_papers_to_digest", 3) for s in sources if s.get("type") == "arxiv"), default=3)
    digest_prompt = get_prompt(profile, "research", "digest_system")
    digests: list[dict] = []

    for paper in scored[:max_digest]:
        arxiv_id = paper.get("arxiv_id", "")
        cached = load_cached_digest(arxiv_id) if arxiv_id else None
        if cached:
            digests.append(cached)
            log.debug("research_node | Digest cache hit — %s", arxiv_id)
            continue

        text = download_paper_text(arxiv_id) if arxiv_id else None
        if not text:
            log.warning("research_node | No full text for %s — skipping digest", arxiv_id)
            continue

        try:
            resp = llm.invoke([
                SystemMessage(content=digest_prompt),
                HumanMessage(content=f"Paper: {paper['title']}\n\n{text[:8000]}"),
            ])
            record = {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "published": paper.get("published", ""),
                "abstract": paper["abstract"],
                "digest": resp.content,
            }
            save_digest(arxiv_id, record)
            digests.append(record)
            log.info("research_node | Digest generated — %s", arxiv_id)
        except Exception as exc:
            log.warning("research_node | Digest failed for %s: %s", arxiv_id, exc)

    return {
        "research_papers": scored,
        "research_summary": summary,
        "paper_digests": digests,
    }
