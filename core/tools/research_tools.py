"""Profile-driven research collection tools.

Each function accepts the same signature so profile YAML can choose which tools
the research node runs:

    collect_x(direction: str, profile: dict, tool_cfg: dict, state: dict) -> list[dict]

Returned records are normalized research artifacts. The research node scores all
artifacts before deciding what to include in summaries and downstream context.
"""
from __future__ import annotations

from html.parser import HTMLParser
import importlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import urllib.request

from configs.config import dev_path
from core.llm.factory import get_llm
from core.memory import MemoryService, default_memory_record_to_artifact
from core.plugins.loader import adapter_has, load_adapter
from core.tools.arxiv_tool import search_arxiv
from core.utils.logger import get_logger
from core.utils.utils import extract_json_array

log = get_logger(__name__)
_WEB_CACHE_DIR = dev_path("web_research")
_WEB_TIMEOUT = 20
_USER_AGENT = "trading_researcher/1.0"
_ARXIV_QUERY_MAX_TERMS = 8
_ARXIV_QUERY_MAX_CHARS = 96
_ARXIV_STOPWORDS = {
    "about", "across", "after", "against", "also", "analysis", "and", "around",
    "based", "before", "between", "build", "can", "compare", "could", "does",
    "detect", "detection", "direction", "each", "effect", "evaluate", "feature",
    "features", "find", "for", "from", "how", "into", "investigate", "is",
    "method", "model", "models", "more", "not", "only", "over", "probe",
    "probing", "research", "should", "signal", "signals", "than", "that",
    "the", "their", "these", "this", "through", "using", "what", "when",
    "via", "where", "whether", "which", "with", "without",
}


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
    categories = [str(item) for item in (tool_cfg.get("categories") or []) if str(item).strip()]
    queries = _build_arxiv_queries(direction, domain_context, profile, tool_cfg)
    match_any = bool(tool_cfg.get("match_any", False))

    papers = _search_arxiv_queries(queries, max_results, categories, match_any=match_any)
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
                "categories": paper.get("categories") or categories,
                "query": paper.get("query", ""),
            },
            "raw": paper,
        })
    return artifacts


def _build_arxiv_queries(
    direction: str,
    domain_context: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
) -> list[str]:
    explicit = str(tool_cfg.get("query") or "").strip()
    configured = [str(term).strip() for term in tool_cfg.get("query_terms") or [] if str(term).strip()]
    if explicit or configured or not tool_cfg.get("llm_query_rewrite"):
        return [_build_arxiv_query(direction, domain_context, tool_cfg)]

    llm_queries = _llm_arxiv_queries(direction, domain_context, profile, tool_cfg)
    return llm_queries or [_build_arxiv_query(direction, domain_context, tool_cfg)]


def _build_arxiv_query(direction: str, domain_context: str, tool_cfg: dict[str, Any]) -> str:
    explicit = str(tool_cfg.get("query") or "").strip()
    if explicit:
        return explicit
    configured = [str(term).strip() for term in tool_cfg.get("query_terms") or [] if str(term).strip()]
    if configured:
        terms = configured
    else:
        terms = _targeted_query_terms(direction)
        if len(terms) < 4:
            terms.extend(term for term in _targeted_query_terms(domain_context) if term not in terms)
    query = " ".join(terms[:_ARXIV_QUERY_MAX_TERMS])
    if len(query) <= _ARXIV_QUERY_MAX_CHARS:
        return query or "arxiv"
    trimmed = query[:_ARXIV_QUERY_MAX_CHARS]
    return trimmed[:trimmed.rfind(" ")].strip() or trimmed.strip() or "arxiv"


def _llm_arxiv_queries(
    direction: str,
    domain_context: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
) -> list[str]:
    max_queries = max(1, min(int(tool_cfg.get("max_queries", 4)), 5))
    step_name = str(tool_cfg.get("query_llm_step") or "arxiv_query_rewrite")
    try:
        llm = get_llm(step_name, profile)
        resp = llm.invoke([
            ("system", (
                "Rewrite the research direction into compact arXiv search queries. "
                "Return only a JSON array of 2-5 strings. Each query should be 3-8 "
                "important technical terms, with no boolean operators, punctuation-heavy "
                "syntax, explanations, or category filters."
            )),
            ("human", (
                f"Research direction:\n{direction}\n\n"
                f"Domain context:\n{domain_context}\n\n"
                f"Return at most {max_queries} queries."
            )),
        ])
        return _normalize_llm_queries(getattr(resp, "content", str(resp)), max_queries=max_queries)
    except Exception as exc:
        log.warning("research_tools | arXiv LLM query rewrite failed: %s", exc)
        return []


def _normalize_llm_queries(text: str, *, max_queries: int) -> list[str]:
    try:
        raw_items = extract_json_array(text)
    except Exception:
        raw_items = [line.strip("-* \t") for line in str(text or "").splitlines()]

    queries: list[str] = []
    for item in raw_items:
        query = item.get("query") if isinstance(item, dict) else item
        query = _clean_arxiv_query_string(str(query or ""))
        if query and query.lower() != "arxiv" and query not in queries:
            queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _clean_arxiv_query_string(query: str) -> str:
    query = re.sub(r"\b(AND|OR|NOT)\b", " ", str(query or ""), flags=re.IGNORECASE)
    query = re.sub(r"[^\w\s\-]", " ", query)
    query = " ".join(query.split())
    if len(query) <= _ARXIV_QUERY_MAX_CHARS:
        return query
    trimmed = query[:_ARXIV_QUERY_MAX_CHARS]
    return trimmed[:trimmed.rfind(" ")].strip() or trimmed.strip()


def _search_arxiv_queries(
    queries: list[str],
    max_results: int,
    categories: list[str],
    *,
    match_any: bool,
) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        for paper in search_arxiv(query, max_results, categories=categories, match_any=match_any):
            key = str(paper.get("arxiv_id") or paper.get("url") or paper.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
            paper = dict(paper)
            paper["query"] = query
            papers.append(paper)
            if len(papers) >= max_results:
                return papers
    return papers


def _targeted_query_terms(text: str) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").replace("-", " "))
    terms: list[str] = []
    for token in normalized.split():
        term = token.lower()
        if len(term) < 3 and term not in {"ai", "ml"}:
            continue
        if term in _ARXIV_STOPWORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) >= _ARXIV_QUERY_MAX_TERMS:
            break
    return terms


def collect_prior_experiments(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Backward-compatible alias for memory retrieval."""
    return collect_memory(direction, profile, tool_cfg, state)


def collect_memory(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Retrieve semantically similar canonical memory records."""
    n_results = int(tool_cfg.get("n_results", 5))

    service = MemoryService.for_profile(profile)
    hits = service.search(direction, n_results=n_results)
    source_name = tool_cfg.get("name", "memory")
    try:
        adapter = load_adapter(profile)
    except Exception:
        adapter = None
    artifacts: list[dict[str, Any]] = []
    for hit in hits:
        record = hit.get("record") or {}
        distance = hit.get("distance")
        artifact: dict[str, Any] | None = None
        if adapter_has(adapter, "memory_record_to_artifact"):
            artifact = adapter.memory_record_to_artifact(profile, record, state)
        if artifact:
            artifact.setdefault("source", source_name)
            metadata = dict(artifact.get("metadata") or {})
            metadata.setdefault("distance", distance)
            artifact["metadata"] = metadata
        else:
            artifact = default_memory_record_to_artifact(record, source_name=source_name, distance=distance)
        artifacts.append(artifact)
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
        env = adapter.validate_environment(profile, state)
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


def collect_ssrn_finance(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    return _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "ssrn_finance"),
        source_type="paper",
        domains=["ssrn.com"],
        title="SSRN Financial Economics search results",
        extra_query_terms=["financial economics", "trading", "asset pricing"],
    )


def collect_nber_asset_pricing(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "nber_asset_pricing"),
        source_type="paper",
        domains=["nber.org"],
        title="NBER asset pricing search results",
        extra_query_terms=["asset pricing", "market timing", "trading strategy"],
    )
    artifacts.append({
        "artifact_id": "nber:asset_pricing_program",
        "source": tool_cfg.get("name", "nber_asset_pricing"),
        "source_type": "research_program",
        "title": "NBER Asset Pricing program",
        "summary": (
            "NBER Asset Pricing is a strong source for macro-finance, return predictability, "
            "factor, and regime papers that can be converted into systematic trading hypotheses."
        ),
        "url": "https://www.nber.org/programs/ap/ap.html",
        "metadata": {"domain": "nber.org"},
        "raw": {"url": "https://www.nber.org/programs/ap/ap.html"},
    })
    return artifacts


def collect_repec_finance(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "repec_finance"),
        source_type="paper",
        domains=["ideas.repec.org"],
        title="IDEAS/RePEc finance search results",
        extra_query_terms=["economics finance anomaly strategy"],
    )
    artifacts.append({
        "artifact_id": "repec:ideas_root",
        "source": tool_cfg.get("name", "repec_finance"),
        "source_type": "research_index",
        "title": "IDEAS/RePEc economics and finance index",
        "summary": (
            "IDEAS/RePEc indexes a large economics and finance literature corpus and is useful "
            "for discovering older anomalies, factor ideas, and macro-finance signals that "
            "arXiv often misses."
        ),
        "url": "https://ideas.repec.org/",
        "metadata": {"domain": "ideas.repec.org"},
        "raw": {"url": "https://ideas.repec.org/"},
    })
    return artifacts


def collect_quantpedia_strategies(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "quantpedia_strategies"),
        source_type="strategy_catalog",
        domains=["quantpedia.com"],
        title="Quantpedia strategy search results",
        extra_query_terms=["trading strategy quant"],
    )
    artifacts.append({
        "artifact_id": "quantpedia:overview",
        "source": tool_cfg.get("name", "quantpedia_strategies"),
        "source_type": "strategy_catalog",
        "title": "Quantpedia strategy encyclopedia",
        "summary": (
            "Quantpedia converts academic papers into structured strategy descriptions, categories, "
            "and implementation hints. It is high leverage for turning literature into concrete "
            "backtest proposals."
        ),
        "url": "https://quantpedia.com/",
        "metadata": {"domain": "quantpedia.com"},
        "raw": {"url": "https://quantpedia.com/"},
    })
    return artifacts


def collect_quantconnect_strategies(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "quantconnect_strategies"),
        source_type="implementation_reference",
        domains=["quantconnect.com"],
        title="QuantConnect community strategy search results",
        extra_query_terms=["strategy algorithm backtest"],
    )
    artifacts.append({
        "artifact_id": "quantconnect:community_strategies",
        "source": tool_cfg.get("name", "quantconnect_strategies"),
        "source_type": "implementation_reference",
        "title": "QuantConnect community strategies",
        "summary": (
            "QuantConnect community strategies and forum discussions are useful for turning paper "
            "ideas into executable algorithm patterns, parameter conventions, and validation setups."
        ),
        "url": "https://www.quantconnect.com/docs/v2/cloud-platform/community/strategies",
        "metadata": {"domain": "quantconnect.com"},
        "raw": {"url": "https://www.quantconnect.com/docs/v2/cloud-platform/community/strategies"},
    })
    return artifacts


def collect_reddit_trading(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    default_terms = str(tool_cfg.get("default_terms") or "algorithmic trading strategies").strip()
    artifacts = _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "reddit_trading"),
        source_type="community_discussion",
        domains=["reddit.com"],
        title="Reddit trading discussion search results",
        extra_query_terms=[default_terms],
    )
    artifacts.append({
        "artifact_id": "reddit:trading_context",
        "source": tool_cfg.get("name", "reddit_trading"),
        "source_type": "community_discussion",
        "title": "Reddit trading discussion context",
        "summary": (
            "Reddit discussions are useful for finding practitioner heuristics, implementation gotchas, "
            "crowded retail ideas, and references to strategy variants that may not appear in formal papers. "
            f"Default search bias: {default_terms}."
        ),
        "url": "https://www.reddit.com/",
        "metadata": {"domain": "reddit.com", "default_terms": default_terms},
        "raw": {"url": "https://www.reddit.com/", "default_terms": default_terms},
    })
    return artifacts


def collect_aqr_research(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = _collect_domain_search_artifacts(
        direction=direction,
        tool_cfg=tool_cfg,
        source_name=tool_cfg.get("name", "aqr_research"),
        source_type="paper",
        domains=["aqr.com"],
        title="AQR research search results",
        extra_query_terms=["momentum value carry trend factor"],
    )
    artifacts.append({
        "artifact_id": "aqr:research_library",
        "source": tool_cfg.get("name", "aqr_research"),
        "source_type": "paper",
        "title": "AQR research library",
        "summary": (
            "AQR research is a strong source for factor definitions, cross-asset signals, and "
            "implementation-oriented discussions of momentum, value, carry, and trend strategies."
        ),
        "url": "https://www.aqr.com/search?Topics=Momentum",
        "metadata": {"domain": "aqr.com"},
        "raw": {"url": "https://www.aqr.com/search?Topics=Momentum"},
    })
    return artifacts


def collect_fred_series_context(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    series = [
        {"code": "FEDFUNDS", "label": "Federal Funds Effective Rate", "use": "policy regime and liquidity sensitivity"},
        {"code": "DGS10", "label": "10-Year Treasury Constant Maturity Rate", "use": "rate regime and duration pressure"},
        {"code": "DGS2", "label": "2-Year Treasury Constant Maturity Rate", "use": "front-end policy sensitivity"},
        {"code": "T10Y2Y", "label": "10Y-2Y Treasury spread", "use": "curve regime / inversion filter"},
        {"code": "VIXCLS", "label": "CBOE VIX", "use": "volatility regime filter"},
        {"code": "CPIAUCSL", "label": "CPI All Urban Consumers", "use": "inflation regime"},
        {"code": "UNRATE", "label": "Unemployment Rate", "use": "macro slowdown / recession proxy"},
        {"code": "INDPRO", "label": "Industrial Production Index", "use": "growth / activity regime"},
    ]
    return [{
        "artifact_id": "fred:macro_series_context",
        "source": tool_cfg.get("name", "fred_series_context"),
        "source_type": "macro_context",
        "title": "FRED macro series for regime-aware trading research",
        "summary": "\n".join(f"{item['code']}: {item['label']} - {item['use']}" for item in series),
        "url": "https://fred.stlouisfed.org/",
        "metadata": {"series_codes": [item["code"] for item in series], "research_direction": direction},
        "raw": {"series": series},
    }]


def collect_cftc_cot(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    return [{
        "artifact_id": "cftc:cot_context",
        "source": tool_cfg.get("name", "cftc_cot"),
        "source_type": "market_structure",
        "title": "CFTC Commitments of Traders context",
        "summary": (
            "COT reports are useful for futures-positioning, crowding, and sentiment strategies. "
            "They break out weekly open interest by trader category and can support regime filters, "
            "contrarian positioning signals, and cross-asset risk appetite context."
        ),
        "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        "metadata": {
            "release_schedule_url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm",
            "reports_url": "https://www.cftc.gov/MarketReports/index.htm",
            "research_direction": direction,
        },
        "raw": {
            "report_types": ["legacy", "disaggregated", "traders in financial futures", "supplemental"],
        },
    }]


def collect_sec_filings_signals(
    direction: str,
    profile: dict[str, Any],
    tool_cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    filing_types = [
        {"form": "10-K", "use": "annual fundamentals, risk language, accounting ratios"},
        {"form": "10-Q", "use": "quarterly drift, revisions, margin changes"},
        {"form": "8-K", "use": "event-driven signals around material disclosures"},
        {"form": "DEF 14A", "use": "governance and compensation signals"},
        {"form": "13F", "use": "institutional holdings and crowding ideas"},
        {"form": "Form 4", "use": "insider trading signals"},
    ]
    return [{
        "artifact_id": "sec:filings_signal_context",
        "source": tool_cfg.get("name", "sec_filings_signals"),
        "source_type": "filings_context",
        "title": "SEC EDGAR filing types for signal generation",
        "summary": "\n".join(f"{item['form']}: {item['use']}" for item in filing_types),
        "url": "https://www.sec.gov/search-filings",
        "metadata": {"filing_types": [item["form"] for item in filing_types], "research_direction": direction},
        "raw": {"filing_types": filing_types},
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


def _collect_domain_search_artifacts(
    *,
    direction: str,
    tool_cfg: dict[str, Any],
    source_name: str,
    source_type: str,
    domains: list[str],
    title: str,
    extra_query_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    max_results = int(tool_cfg.get("max_results", 6))
    query = _normalize_search_query(direction, extra_query_terms or [])
    records = _search_domain_results(query=query, domains=domains, max_results=max_results)
    artifacts: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        url = str(item.get("url") or "")
        artifacts.append({
            "artifact_id": f"{source_name}:{idx}:{_slug(url or item.get('title') or source_name)}",
            "source": source_name,
            "source_type": source_type,
            "title": str(item.get("title") or title),
            "summary": str(item.get("snippet") or ""),
            "url": url,
            "metadata": {"domain": item.get("domain", ""), "query": query},
            "raw": item,
        })
    return artifacts


def _normalize_search_query(direction: str, extra_terms: list[str]) -> str:
    text = " ".join(str(direction or "").split())
    query_parts = [text] if text else []
    query_parts.extend(str(term).strip() for term in extra_terms if str(term).strip())
    return " ".join(query_parts).strip()


def _search_domain_results(query: str, domains: list[str], max_results: int) -> list[dict[str, Any]]:
    if not query or not domains or max_results <= 0:
        return []
    cached = _load_cached_search(query, domains, max_results)
    if cached is not None:
        return cached
    site_clause = " OR ".join(f"site:{domain}" for domain in domains)
    full_query = f"{query} ({site_clause})"
    url = f"https://duckduckgo.com/html/?q={quote_plus(full_query)}"
    html = _fetch_text(url)
    parser = _DuckDuckGoParser(max_results=max_results)
    parser.feed(html)
    parser.close()
    results = parser.results[:max_results]
    _save_cached_search(query, domains, max_results, results)
    return results


def _cache_path(query: str, domains: list[str], max_results: int) -> Path:
    key = _slug(f"{query}__{'_'.join(sorted(domains))}__{max_results}")[:180]
    return _WEB_CACHE_DIR / f"{key}.json"


def _load_cached_search(query: str, domains: list[str], max_results: int) -> list[dict[str, Any]] | None:
    path = _cache_path(query, domains, max_results)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("research_tools | Failed to read web cache %s: %s", path.name, exc)
        return None
    results = payload.get("results")
    return results if isinstance(results, list) else None


def _save_cached_search(query: str, domains: list[str], max_results: int, results: list[dict[str, Any]]) -> None:
    path = _cache_path(query, domains, max_results)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "domains": list(domains),
        "max_results": int(max_results),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_WEB_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "")).strip("_").lower()
    return text or "item"


class _DuckDuckGoParser(HTMLParser):
    def __init__(self, *, max_results: int):
        super().__init__()
        self.max_results = max_results
        self.results: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_map = dict(attrs)
        class_name = attr_map.get("class") or ""
        if tag == "a" and "result__a" in class_name and len(self.results) < self.max_results:
            self._finalize_current()
            href = _unwrap_duckduckgo_url(attr_map.get("href") or "")
            self._current = {"url": href, "title": "", "snippet": "", "domain": _extract_domain(href)}
            self._capture_title = True
            return
        if tag == "a":
            self._capture_title = False
        if tag in {"a", "div"} and ("result__snippet" in class_name or "result__extras__url" in class_name) and self._current is not None:
            self._capture_snippet = True

    def handle_endtag(self, tag: str):
        if tag == "a" and self._capture_title:
            self._capture_title = False
        if tag in {"a", "div"}:
            self._capture_snippet = False

    def handle_data(self, data: str):
        if self._current is None:
            return
        if self._capture_title:
            self._current["title"] = f"{self._current.get('title', '')} {data}"
        elif self._capture_snippet:
            self._current["snippet"] = f"{self._current.get('snippet', '')} {data}"

    def close(self):
        self._finalize_current()
        super().close()

    def _finalize_current(self):
        if self._current is None:
            return
        self._current["title"] = _clean_whitespace(self._current.get("title", ""))
        self._current["snippet"] = _clean_whitespace(self._current.get("snippet", ""))
        if self._current.get("url") and self._current.get("title"):
            self.results.append(self._current)
        self._current = None
        self._capture_title = False
        self._capture_snippet = False


def _unwrap_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if "duckduckgo.com" not in parsed.netloc:
        return url
    target = parse_qs(parsed.query).get("uddg")
    if target:
        return unquote(target[0])
    return url


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _clean_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())
