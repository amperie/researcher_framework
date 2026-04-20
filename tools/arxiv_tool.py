"""Arxiv search and paper download tool."""
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
_HTML_TIMEOUT = 15


class _TextExtractor(HTMLParser):
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
    parser = _TextExtractor()
    parser.feed(html)
    return re.sub(r"\n{3,}", "\n\n", parser.get_text())


def _safe_id(arxiv_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", arxiv_id)


def _digest_cache_path(arxiv_id: str) -> Path:
    return _PAPERS_CACHE_DIR / f"{_safe_id(arxiv_id)}.digest"


def load_cached_digest(arxiv_id: str) -> dict | None:
    path = _digest_cache_path(arxiv_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("arxiv_tool | Failed to read digest cache for %s: %s", arxiv_id, exc)
        return None


def save_digest(arxiv_id: str, record: dict) -> None:
    path = _digest_cache_path(arxiv_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log.debug("arxiv_tool | Digest cached — %s", arxiv_id)


def search_arxiv(query: str, max_results: int) -> list[dict]:
    """Search arxiv and return scored paper dicts."""
    log.info("arxiv_tool | Searching — query=%r, max_results=%d", query, max_results)
    results = list(
        arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        ).results()
    )
    papers = []
    for r in results:
        papers.append({
            "title": r.title,
            "abstract": r.summary.replace("\n", " "),
            "url": r.entry_id,
            "arxiv_id": r.get_short_id(),
            "published": r.published.date().isoformat(),
        })
    log.debug("arxiv_tool | Returned %d papers", len(papers))
    return papers


def download_paper_text(arxiv_id: str) -> str | None:
    """Download plain text from the arxiv HTML page for *arxiv_id*."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    log.debug("arxiv_tool | Fetching HTML — %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=_HTML_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        text = _html_to_text(html)
        log.info("arxiv_tool | Downloaded %s (%d chars)", arxiv_id, len(text))
        return text
    except urllib.error.HTTPError as exc:
        log.warning("arxiv_tool | No HTML for %s (HTTP %d)", arxiv_id, exc.code)
        return None
    except Exception as exc:
        log.warning("arxiv_tool | Failed to fetch %s: %s", arxiv_id, exc)
        return None
