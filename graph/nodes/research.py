"""Research node — arxiv search and summarization."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from configs.config import get_config
from graph.state import ResearchState
from llm.factory import get_llm
from utils.logger import get_logger
from configs.prompts import RESEARCH_DIGEST, RESEARCH_SCORE, RESEARCH_SUMMARY
from tools.arxiv_tool import download_paper_text, load_cached_digest, save_digest, search_arxiv
from utils import extract_json_array, load_yaml_section

log = get_logger(__name__)


def research_node(state: ResearchState) -> dict:
    """Search arxiv and summarise relevant papers.

    Reads:
        state['research_direction']

    Writes:
        state['arxiv_papers']      — papers sorted by relevance_score descending
        state['research_summary']  — LLM synthesis of the top papers
    """
    direction = state.get("research_direction", "")
    cfg = get_config()
    llm = get_llm()

    log.info("research_node | Searching arxiv — direction=%r, max=%d",
             direction, cfg.max_arxiv_papers)

    papers = search_arxiv(direction, max_results=cfg.max_arxiv_papers)
    log.info("research_node | Fetched %d papers", len(papers))

    if not papers:
        log.warning("research_node | No papers found — returning empty state")
        return {"arxiv_papers": [], "research_summary": "No relevant papers found."}

    # ------------------------------------------------------------------
    # Score all papers in a single LLM call
    # ------------------------------------------------------------------
    papers_block = "\n\n".join(
        f"[{i + 1}] Title: {p['title']}\nAbstract: {p['abstract']}"
        for i, p in enumerate(papers)
    )
    score_prompt = (
        f"Research direction: {direction}\n\n"
        f"Papers to score ({len(papers)} total):\n\n{papers_block}\n\n"
        f"Return a JSON array of {len(papers)} integers (0–10), one per paper."
    )

    try:
        score_response = llm.invoke([
            SystemMessage(content=RESEARCH_SCORE),
            HumanMessage(content=score_prompt),
        ])
        scores = extract_json_array(score_response.content)
        if len(scores) != len(papers):
            raise ValueError(
                f"Expected {len(papers)} scores, got {len(scores)}"
            )
        log.debug("research_node | Scores: %s", scores)
    except Exception as exc:
        log.warning("research_node | Scoring failed (%s) — defaulting all scores to 5", exc)
        scores = [5] * len(papers)

    for paper, score in zip(papers, scores):
        paper["relevance_score"] = int(score)

    papers.sort(key=lambda p: p["relevance_score"], reverse=True)
    log.info("research_node | Top paper — %r (score=%d)",
             papers[0]["title"], papers[0]["relevance_score"])

    # ------------------------------------------------------------------
    # Filter by relevance threshold
    # ------------------------------------------------------------------
    node_cfg = load_yaml_section("research_node")
    threshold = int(node_cfg.get("relevance_score_threshold", 0))
    if threshold > 0:
        before = len(papers)
        papers = [p for p in papers if p["relevance_score"] >= threshold]
        log.info("research_node | Threshold=%d — kept %d/%d papers",
                 threshold, len(papers), before)
        if not papers:
            log.warning("research_node | All papers filtered out at threshold=%d", threshold)
            return {"arxiv_papers": [], "research_summary": "No papers met the relevance threshold."}

    # ------------------------------------------------------------------
    # Synthesise a research summary from the top papers
    # ------------------------------------------------------------------
    top_n = min(5, len(papers))
    top_block = "\n\n".join(
        f"[{i + 1}] {p['title']} (score {p['relevance_score']})\n{p['abstract']}"
        for i, p in enumerate(papers[:top_n])
    )
    summary_prompt = (
        f"Research direction: {direction}\n\n"
        f"Top {top_n} papers by relevance:\n\n{top_block}"
    )

    try:
        summary_response = llm.invoke([
            SystemMessage(content=RESEARCH_SUMMARY),
            HumanMessage(content=summary_prompt),
        ])
        research_summary = summary_response.content.strip()
        log.info("research_node | Summary generated — %d chars", len(research_summary))
    except Exception as exc:
        log.warning("research_node | Summary generation failed (%s) — using fallback", exc)
        research_summary = "\n".join(
            f"- {p['title']} (score {p['relevance_score']})" for p in papers[:top_n]
        )

    # ------------------------------------------------------------------
    # Download and digest full paper text for the top N papers
    # ------------------------------------------------------------------
    max_to_digest = int(node_cfg.get("max_papers_to_digest", 3))
    digests: list[dict] = []

    for paper in papers[:max_to_digest]:
        arxiv_id = paper["arxiv_id"]

        cached = load_cached_digest(arxiv_id)
        if cached:
            log.info("research_node | Digest loaded from cache — %s", arxiv_id)
            digests.append(cached)
            continue

        full_text = download_paper_text(arxiv_id)
        if not full_text:
            log.warning("research_node | No HTML available for %s — skipping digest", arxiv_id)
            continue

        try:
            digest_response = llm.invoke([
                SystemMessage(content=RESEARCH_DIGEST),
                HumanMessage(content=(
                    f"Research direction: {direction}\n\n"
                    f"Paper: {paper['title']}\n\n"
                    f"{full_text[:40000]}"
                )),
            ])
            record = {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "published": paper["published"],
                "abstract": paper["abstract"],
                "digest": digest_response.content.strip(),
            }
            save_digest(arxiv_id, record)
            digests.append(record)
            log.info("research_node | Digest extracted and cached — %s (%d chars)",
                     arxiv_id, len(record["digest"]))
        except Exception as exc:
            log.warning("research_node | Digest extraction failed for %s: %s", arxiv_id, exc)

    log.info("research_node | Digests ready: %d/%d", len(digests), max_to_digest)

    return {
        "arxiv_papers": papers,
        "research_summary": research_summary,
        "paper_digests": digests,
    }
