"""Arxiv search tool.

Thin wrapper around the `arxiv` Python client focused on LLM internals,
mechanistic interpretability, and activation-based probing research.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import arxiv

from utils.logger import get_logger

log = get_logger(__name__)

_PAPERS_CACHE_DIR = Path("dev/papers")
_HTML_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal HTMLParser subclass that collects visible text."""

    _SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    # Collapse runs of blank lines
    text = parser.get_text()
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------------------------
# Digest cache (dev/papers/<safe_id>.digest)
# ---------------------------------------------------------------------------

def _safe_id(arxiv_id: str) -> str:
    """Convert an arxiv ID to a filesystem-safe filename stem."""
    return re.sub(r"[^\w\-]", "_", arxiv_id)


def _digest_cache_path(arxiv_id: str) -> Path:
    return _PAPERS_CACHE_DIR / f"{_safe_id(arxiv_id)}.digest"


def load_cached_digest(arxiv_id: str) -> dict | None:
    """Return the cached digest record for *arxiv_id*, or None if not cached."""
    path = _digest_cache_path(arxiv_id)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        log.debug("arxiv_tool | Cache hit — %s", arxiv_id)
        return record
    except Exception as exc:
        log.warning("arxiv_tool | Failed to read digest cache for %s: %s", arxiv_id, exc)
        return None


def save_digest(arxiv_id: str, record: dict) -> None:
    """Write *record* to the digest cache for *arxiv_id*."""
    path = _digest_cache_path(arxiv_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log.debug("arxiv_tool | Digest cached — %s -> %s", arxiv_id, path)


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


def download_paper_text(arxiv_id: str) -> str | None:
    """Download and return the plain-text content of a paper's arxiv HTML page.

    arxiv provides an HTML rendering for most papers submitted after ~2022 at
    https://arxiv.org/html/<id>. Older papers or those without HTML versions
    return a 404, in which case this function returns None.

    Args:
        arxiv_id: Short arxiv ID (e.g. '2310.01234v1').

    Returns:
        Plain text extracted from the HTML, or None if unavailable.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    log.debug("arxiv_tool | Fetching HTML — %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuralSignalResearcher/1.0"})
        with urllib.request.urlopen(req, timeout=_HTML_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        text = _html_to_text(html)
        log.info("arxiv_tool | Downloaded paper HTML — %s (%d chars)", arxiv_id, len(text))
        return text
    except urllib.error.HTTPError as exc:
        log.warning("arxiv_tool | No HTML version for %s (HTTP %d)", arxiv_id, exc.code)
        return None
    except Exception as exc:
        log.warning("arxiv_tool | Failed to fetch HTML for %s: %s", arxiv_id, exc)
        return None
