"""Arxiv search tool.

Thin wrapper around the `arxiv` Python client focused on LLM internals,
mechanistic interpretability, and activation-based probing research.
"""
from __future__ import annotations

import arxiv

from logger import get_logger

log = get_logger(__name__)


def search_arxiv(query: str, max_results: int) -> list[dict]:
    """Search arxiv and return the top results as structured dicts.

    Args:
        query: Free-text search query (e.g. 'activation patching transformer').
        max_results: Maximum number of papers to return.

    Returns:
        List of dicts with keys:
            - title (str)
            - abstract (str)
            - url (str)          — HTML abstract page URL
            - arxiv_id (str)     — short ID like '2310.01234'
            - published (str)    — ISO date string
    """
    log.info("Searching arxiv — query=%r, max_results=%d", query, max_results)
    results = list(
        arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        ).results()
    )
    log.debug("arxiv returned %d raw results", len(results))
    papers = []
    for r in results:
        log.debug("  [%s] %s", r.get_short_id(), r.title)
        papers.append({
            "title": r.title,
            "abstract": r.summary.replace("\n", " "),
            "url": r.entry_id,
            "arxiv_id": r.get_short_id(),
            "published": r.published.date().isoformat(),
        })
    return papers


def get_paper_details(arxiv_id: str) -> dict:
    """Fetch full metadata for a single paper by its arxiv ID.

    Args:
        arxiv_id: Short arxiv ID (e.g. '2310.01234') or full URL.

    Returns:
        Dict with the same keys as search_arxiv results plus:
            - authors (list[str])
            - pdf_url (str)
            - categories (list[str])
    """
    log.debug("Fetching paper details — arxiv_id=%r", arxiv_id)
    result = next(arxiv.Search(id_list=[arxiv_id]).results())
    log.info("Fetched paper: %r (%s)", result.title, arxiv_id)
    return {
        "title": result.title,
        "abstract": result.summary.replace("\n", " "),
        "url": result.entry_id,
        "arxiv_id": result.get_short_id(),
        "published": result.published.date().isoformat(),
        "authors": [a.name for a in result.authors],
        "pdf_url": result.pdf_url,
        "categories": result.categories,
    }
