from __future__ import annotations

from unittest.mock import patch

from core.tools.research_tools import (
    collect_aqr_research,
    collect_cftc_cot,
    collect_fred_series_context,
    collect_nber_asset_pricing,
    collect_quantconnect_strategies,
    collect_quantpedia_strategies,
    collect_reddit_trading,
    collect_repec_finance,
    collect_sec_filings_signals,
    collect_ssrn_finance,
)


PROFILE = {"name": "trading"}


def _mock_results() -> list[dict]:
    return [{
        "title": "Momentum Research Paper",
        "snippet": "Useful summary for trading strategy ideation.",
        "url": "https://example.com/paper",
        "domain": "example.com",
    }]


def test_domain_collectors_map_search_results_to_artifacts():
    collectors = [
        collect_ssrn_finance,
        collect_nber_asset_pricing,
        collect_repec_finance,
        collect_quantpedia_strategies,
        collect_quantconnect_strategies,
        collect_aqr_research,
    ]
    for collector in collectors:
        with patch("core.tools.research_tools._search_domain_results", return_value=_mock_results()):
            artifacts = collector(
                "momentum trading on SPY",
                PROFILE,
                {"name": collector.__name__, "max_results": 3},
                {},
            )
        assert artifacts
        assert any(item["url"] == "https://example.com/paper" for item in artifacts)
        assert all(item["source"] == collector.__name__ for item in artifacts[:-1] or artifacts)


def test_reddit_trading_uses_default_algorithmic_trading_query_bias():
    captured: dict[str, str] = {}

    def _fake_search(*, query: str, domains: list[str], max_results: int):
        captured["query"] = query
        captured["domains"] = ",".join(domains)
        return _mock_results()

    with patch("core.tools.research_tools._search_domain_results", side_effect=_fake_search):
        artifacts = collect_reddit_trading(
            "SPY momentum ideas",
            PROFILE,
            {"name": "reddit_trading", "max_results": 3},
            {},
        )

    assert "algorithmic trading strategies" in captured["query"]
    assert captured["domains"] == "reddit.com"
    assert artifacts
    assert artifacts[0]["source"] == "reddit_trading"


def test_fred_series_context_returns_curated_series():
    artifacts = collect_fred_series_context(
        "macro regime filters for SPY",
        PROFILE,
        {"name": "fred_series_context"},
        {},
    )
    assert len(artifacts) == 1
    assert "FEDFUNDS" in artifacts[0]["summary"]
    assert "VIXCLS" in artifacts[0]["summary"]


def test_cftc_cot_returns_market_structure_context():
    artifacts = collect_cftc_cot(
        "positioning and crowding signals",
        PROFILE,
        {"name": "cftc_cot"},
        {},
    )
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["release_schedule_url"].startswith("https://www.cftc.gov/")


def test_sec_filings_signals_returns_filing_taxonomy():
    artifacts = collect_sec_filings_signals(
        "event driven equity signals",
        PROFILE,
        {"name": "sec_filings_signals"},
        {},
    )
    assert len(artifacts) == 1
    assert "10-K" in artifacts[0]["summary"]
    assert "Form 4" in artifacts[0]["summary"]
