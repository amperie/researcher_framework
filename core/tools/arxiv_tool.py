"""Arxiv search and paper download tool."""
from __future__ import annotations

import json
import re
import time
from hashlib import sha256
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import arxiv
import requests

from configs.config import dev_path
from core.utils.logger import get_logger

log = get_logger(__name__)

_PAPERS_CACHE_DIR = dev_path("papers")
_SEARCH_CACHE_DIR = dev_path("paper_searches")
_HTML_TIMEOUT = 15
_DEFAULT_SEARCH_RESULTS = 8
_MAX_SEARCH_RESULTS = 10
_MAX_QUERY_CHARS = 180
_SEARCH_CACHE_VERSION = "3"
_ARXIV_REQUEST_TIMEOUT = 20
_ARXIV_DELAY_SECONDS = 10.0
_ARXIV_NUM_RETRIES = 0
_RATE_LIMIT_COOLDOWN_SECONDS = 60 * 60


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


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        return "arxiv"
    lowered = text.lower()
    for prefix in (
        "research direction:",
        "direction:",
        "goal:",
        "task:",
    ):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            lowered = text.lower()
            break

    stop_phrases = (
        "focus on systematic trading",
        "return a json",
        "emit signals",
        "when we should buy and sell",
        "only the ",
    )
    cut_index = len(text)
    for phrase in stop_phrases:
        idx = lowered.find(phrase)
        if idx >= 0:
            cut_index = min(cut_index, idx)
    text = text[:cut_index].strip(" ,;:.")
    if len(text) > _MAX_QUERY_CHARS:
        trimmed = text[:_MAX_QUERY_CHARS]
        last_space = trimmed.rfind(" ")
        if last_space > 80:
            trimmed = trimmed[:last_space]
        text = trimmed.strip(" ,;:.")
    return text or "arxiv"


def _effective_max_results(max_results: int) -> int:
    requested = int(max_results or _DEFAULT_SEARCH_RESULTS)
    if requested < 1:
        return 1
    return min(requested, _MAX_SEARCH_RESULTS)


def _normalize_categories(categories: list[str] | None = None) -> list[str]:
    normalized: list[str] = []
    for item in categories or []:
        value = " ".join(str(item or "").split())
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _build_query(query: str, categories: list[str] | None = None, *, match_any: bool = False) -> str:
    normalized_query = _normalize_query(query)
    normalized_categories = _normalize_categories(categories)
    if match_any:
        terms = [term for term in normalized_query.split() if term]
        if len(terms) > 1:
            normalized_query = " OR ".join(terms)
    if not normalized_categories:
        return normalized_query
    category_clause = " OR ".join(f"cat:{item}" for item in normalized_categories)
    return f"({normalized_query}) AND ({category_clause})"


def _search_cache_path(
    query: str,
    max_results: int,
    categories: list[str] | None = None,
    *,
    match_any: bool = False,
) -> Path:
    normalized = _normalize_query(query)
    effective_max = _effective_max_results(max_results)
    normalized_categories = ",".join(_normalize_categories(categories))
    key = sha256(
        f"{_SEARCH_CACHE_VERSION}|{normalized}|{effective_max}|{normalized_categories}|{match_any}".encode("utf-8")
    ).hexdigest()[:24]
    return _SEARCH_CACHE_DIR / f"{key}.json"


def load_cached_search(
    query: str,
    max_results: int,
    categories: list[str] | None = None,
    *,
    match_any: bool = False,
) -> list[dict] | None:
    path = _search_cache_path(query, max_results, categories, match_any=match_any)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("arxiv_tool | Failed to read search cache for %s: %s", path.name, exc)
        return None
    papers = payload.get("papers")
    if isinstance(papers, list):
        return papers
    unavailable_at = float(payload.get("unavailable_at") or payload.get("rate_limited_at") or 0)
    if unavailable_at and time.time() - unavailable_at < _RATE_LIMIT_COOLDOWN_SECONDS:
        log.info("arxiv_tool | Rate-limit cooldown cache hit - %s", path.name)
        return []
    return None


def save_search(
    query: str,
    max_results: int,
    papers: list[dict],
    categories: list[str] | None = None,
    *,
    match_any: bool = False,
) -> None:
    path = _search_cache_path(query, max_results, categories, match_any=match_any)
    payload = {
        "query": _normalize_query(query),
        "max_results": _effective_max_results(max_results),
        "categories": _normalize_categories(categories),
        "match_any": bool(match_any),
        "papers": papers,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.debug("arxiv_tool | Search cached - %s", path.name)
    except Exception as exc:
        log.warning("arxiv_tool | Failed to write search cache %s: %s", path.name, exc)


def save_rate_limited_search(
    query: str,
    max_results: int,
    categories: list[str] | None = None,
    *,
    reason: str = "rate_limited",
    match_any: bool = False,
) -> None:
    path = _search_cache_path(query, max_results, categories, match_any=match_any)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "query": _normalize_query(query),
                    "max_results": _effective_max_results(max_results),
                    "categories": _normalize_categories(categories),
                    "match_any": bool(match_any),
                    "unavailable_at": time.time(),
                    "reason": reason,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("arxiv_tool | Failed to write rate-limit cache %s: %s", path.name, exc)


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
    log.debug("arxiv_tool | Digest cached - %s", arxiv_id)


def search_arxiv(
    query: str,
    max_results: int,
    categories: list[str] | None = None,
    *,
    match_any: bool = False,
) -> list[dict]:
    """Search arxiv and return scored paper dicts."""
    normalized_categories = _normalize_categories(categories)
    cached = load_cached_search(query, max_results, normalized_categories, match_any=match_any)
    if cached is not None:
        log.info(
            "arxiv_tool | Search cache hit query=%r categories=%s max_results=%d papers=%d",
            _build_query(query, normalized_categories, match_any=match_any),
            normalized_categories,
            _effective_max_results(max_results),
            len(cached),
        )
        return cached

    normalized_query = _build_query(query, normalized_categories, match_any=match_any)
    effective_max = _effective_max_results(max_results)
    log.info("arxiv_tool | Searching query=%r max_results=%d", normalized_query, effective_max)
    client = arxiv.Client(
        page_size=effective_max,
        delay_seconds=_ARXIV_DELAY_SECONDS,
        num_retries=_ARXIV_NUM_RETRIES,
    )
    session_get = client._session.get
    client._session.get = lambda url, **kwargs: session_get(
        url,
        timeout=_ARXIV_REQUEST_TIMEOUT,
        **kwargs,
    )
    try:
        results = list(
            client.results(
                arxiv.Search(
                    query=normalized_query,
                    max_results=effective_max,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
            )
        )
    except arxiv.HTTPError as exc:
        if exc.status == 429:
            log.warning("arxiv_tool | Rate limited by arXiv; cooling down query=%r", normalized_query)
            save_rate_limited_search(query, max_results, normalized_categories, reason="rate_limited", match_any=match_any)
            return []
        raise
    except (requests.exceptions.RequestException, TimeoutError, OSError) as exc:
        log.warning("arxiv_tool | arXiv unavailable; cooling down query=%r error=%s", normalized_query, exc)
        save_rate_limited_search(query, max_results, normalized_categories, reason="network_unavailable", match_any=match_any)
        return []
    papers = []
    for result in results:
        papers.append({
            "title": result.title,
            "abstract": result.summary.replace("\n", " "),
            "url": result.entry_id,
            "arxiv_id": result.get_short_id(),
            "published": result.published.date().isoformat(),
            "categories": list(getattr(result, "categories", None) or []),
        })
    save_search(query, max_results, papers, normalized_categories, match_any=match_any)
    log.debug("arxiv_tool | Returned %d papers", len(papers))
    return papers


def download_paper_text(arxiv_id: str) -> str | None:
    """Download plain text from the arxiv HTML page for *arxiv_id*."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    log.debug("arxiv_tool | Fetching HTML - %s", url)
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
