from __future__ import annotations

import json

from core.graph.nodes.plan_implementation import _proposal_execution_context


def test_plan_context_preserves_fixed_symbol_strategies():
    proposals = [
        {
            "name": "fixed_symbol_strategy",
            "symbol": "SPY",
            "symbols": ["SPY", "SH"],
            "mode": "backtest",
            "data_source": "alpaca",
            "timeframe": "5min",
        }
    ]

    context = json.loads(_proposal_execution_context(proposals))

    assert "guidance" in context
    assert any("fixed-symbol strategies are allowed" in line for line in context["guidance"])
    assert context["proposals"][0]["expected_symbols"] == ["SPY", "SH"]
