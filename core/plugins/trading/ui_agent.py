"""UI-facing trading agent wrapper.

This module is intentionally small and JSON-oriented so product APIs can call the
researcher without depending on graph internals, brainstorm sessions, or local
experiment directory structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, Callable

Progress = Callable[[int, str, dict[str, Any] | None], None]


def create_idea_batch(payload: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    direction = _required_text(payload, "direction")
    max_ideas = max(1, min(8, int(payload.get("maxIdeas") or 4)))
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    symbols = _symbols(payload, constraints)
    _progress(progress, 10, "Researching direction", {"direction": direction})
    templates = _idea_templates(direction)
    _progress(
        progress,
        35,
        "Collected research artifacts",
        {"artifactCount": len(templates), "sources": ["profile_context", "strategy_templates"]},
    )
    _progress(progress, 60, "Scoring candidate ideas", {"ideaCount": min(max_ideas, len(templates))})
    ideas = [_idea(direction, symbols, i, template) for i, template in enumerate(_idea_templates(direction), 1)]
    _progress(progress, 90, "Finalizing idea batch", {"ideaCount": len(ideas[:max_ideas])})
    batch_id = f"idea_batch_{_digest({'direction': direction, 'constraints': constraints})}"
    return {
        "id": batch_id,
        "direction": direction,
        "status": "succeeded",
        "message": "Generated idea batch from trading direction.",
        "ideas": ideas[:max_ideas],
        "createdAt": _now(),
        "updatedAt": _now(),
        "provenance": {
            "profile": "trading",
            "adapter": "core.plugins.trading.ui_agent",
            "mode": "deterministic-wrapper",
        },
    }


def create_component_draft(
    payload: dict[str, Any], progress: Progress | None = None
) -> dict[str, Any]:
    direction = _required_text(payload, "direction")
    idea = payload.get("idea")
    if not isinstance(idea, dict):
        raise ValueError("idea is required")
    generation_mode = str(payload.get("generationMode") or "algorithm-only")
    _progress(
        progress,
        10,
        "Planning component generation",
        {"generationMode": generation_mode, "ideaId": str(idea.get("id") or "")},
    )
    class_seed = str(idea.get("name") or "AgentGeneratedStrategy")
    algorithm_class = _class_name(class_seed, suffix="Algorithm")
    portfolio_class = _class_name(class_seed, suffix="Portfolio")
    symbols = _list(idea.get("proposedSymbols")) or ["SPY"]
    draft_id = f"component_draft_{_digest({'direction': direction, 'idea': idea, 'mode': generation_mode})}"
    _progress(
        progress,
        35,
        "Generating algorithm source",
        {"className": algorithm_class, "symbols": symbols},
    )
    algorithm_source = _algorithm_source(
        class_name=algorithm_class,
        description=str(idea.get("description") or direction),
        symbols=symbols,
    )
    if generation_mode in {"algorithm-and-portfolio", "portfolio-only"}:
        _progress(
            progress,
            65,
            "Generating portfolio source",
            {"className": portfolio_class, "symbols": symbols},
        )
    portfolio_source = (
        _portfolio_source(
            class_name=portfolio_class,
            description=str(idea.get("suggestedPortfolioBehavior") or "Fixed fractional risk portfolio."),
            symbols=symbols,
        )
        if generation_mode in {"algorithm-and-portfolio", "portfolio-only"}
        else None
    )
    _progress(
        progress,
        90,
        "Finalizing component draft",
        {"hasAlgorithm": generation_mode != "portfolio-only", "hasPortfolio": portfolio_source is not None},
    )
    return {
        "id": draft_id,
        "ideaBatchId": str(payload.get("ideaBatchId") or ""),
        "ideaId": str(idea.get("id") or ""),
        "direction": direction,
        "status": "succeeded",
        "message": "Generated component draft.",
        "algorithm": {
            "name": str(idea.get("name") or algorithm_class).replace("_", " ").title(),
            "filename": f"{algorithm_class}.py",
            "className": algorithm_class,
            "description": str(idea.get("description") or direction),
            "content": algorithm_source,
        }
        if generation_mode != "portfolio-only"
        else None,
        "portfolio": {
            "name": f"{str(idea.get('name') or portfolio_class).replace('_', ' ').title()} Portfolio",
            "filename": f"{portfolio_class}.py",
            "className": portfolio_class,
            "description": str(idea.get("suggestedPortfolioBehavior") or "Generated portfolio draft."),
            "content": portfolio_source,
        }
        if portfolio_source
        else None,
        "createdAt": _now(),
        "updatedAt": _now(),
        "provenance": {
            "profile": "trading",
            "adapter": "core.plugins.trading.ui_agent",
            "mode": generation_mode,
        },
    }


def _idea_templates(direction: str) -> list[dict[str, str]]:
    lower = direction.lower()
    primary = (
        {
            "name": "volatility_filtered_mean_reversion",
            "description": "Mean reversion entry gated by volatility regime and benchmark context.",
            "hypothesis": "Short-term dislocations revert more reliably when volatility is elevated but not disorderly.",
        }
        if "mean" in lower or "reversion" in lower
        else {
            "name": "regime_confirmed_momentum",
            "description": "Trend-following signal confirmed by recent volatility and benchmark participation.",
            "hypothesis": "Directional continuation is stronger when trend, volatility, and market breadth agree.",
        }
    )
    return [
        primary,
        {
            "name": "breakout_failure_reversal",
            "description": "Fade failed breakouts after volume-confirmed exhaustion.",
            "hypothesis": "False breakouts create asymmetric reversal opportunities after crowded intraday moves.",
        },
        {
            "name": "adaptive_range_expansion",
            "description": "Trade range expansion only when realized volatility supports follow-through.",
            "hypothesis": "Volatility-adjusted breakouts avoid low-energy chop better than fixed thresholds.",
        },
        {
            "name": "benchmark_relative_strength_filter",
            "description": "Trade the selected symbol only when relative strength versus benchmark confirms the setup.",
            "hypothesis": "Filtering by benchmark-relative behavior reduces idiosyncratic noise and weak signals.",
        },
    ]


def _idea(direction: str, symbols: list[str], index: int, template: dict[str, str]) -> dict[str, Any]:
    idea_id = f"idea_{_digest({'direction': direction, 'index': index, 'name': template['name']})}"
    return {
        "id": idea_id,
        "name": template["name"],
        "description": template["description"],
        "hypothesis": template["hypothesis"],
        "rationale": f"Derived from the requested direction: {direction}",
        "proposedSymbols": symbols,
        "requiredData": ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        "suggestedPortfolioBehavior": "Single-symbol bracket or fixed-fractional portfolio with explicit stops, profit target, transaction costs, and slippage.",
        "risks": ["overfitting", "lookahead leakage", "insufficient trade count", "transaction-cost sensitivity"],
        "complexity": "medium" if index == 1 else "high",
        "score": max(0, 9 - index),
    }


def _algorithm_source(*, class_name: str, description: str, symbols: list[str]) -> str:
    primary = symbols[0] if symbols else "SPY"
    return f'''from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType


class {class_name}(Algorithm):
    crucible_metadata = {{
        "schema_version": "1",
        "role": "algorithm",
        "description": {description!r},
        "tunables": [
            {{"key": "lookback", "type": "integer", "default": 40, "min": 10, "max": 200}},
            {{"key": "z_threshold", "type": "number", "default": 1.5, "min": 0.5, "max": 4.0}},
        ],
        "fixed_parameters": {{"primary_symbol": {primary!r}}},
        "required_symbols": {symbols!r},
        "required_timeframes": ["Minute"],
        "required_fields": ["open", "high", "low", "close", "volume"],
        "warmup_bars": 40,
        "signal_contract": {{"emits": ["BUY", "SELL"], "strength_range": [0, 100]}},
        "statefulness": "rolling price history",
        "dependencies": [],
    }}

    def __init__(self, cfg: dict | None = None, history_length: int = 0):
        cfg = cfg or {{}}
        super().__init__(cfg=cfg, history_length=history_length)
        self.symbol = cfg.get("symbol", {primary!r})
        self.lookback = int(cfg.get("lookback", 40))
        self.z_threshold = float(cfg.get("z_threshold", 1.5))

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        tick = next((item for item in data if item.symbol == self.symbol), None)
        if tick is None:
            return []
        closes = list(self.price_history.get(self.symbol, []))[-self.lookback:]
        if len(closes) < max(5, self.lookback // 2):
            return []
        mean = sum(closes) / len(closes)
        variance = sum((value - mean) ** 2 for value in closes) / len(closes)
        std = variance ** 0.5
        if std <= 0:
            return []
        z_score = (tick.close - mean) / std
        strength = int(min(100, max(1, abs(z_score) / self.z_threshold * 60)))
        if z_score <= -self.z_threshold:
            return [MarketSignal(SignalType.BUY, self.symbol, strength, {{"z_score": z_score}})]
        if z_score >= self.z_threshold:
            return [MarketSignal(SignalType.SELL, self.symbol, strength, {{"z_score": z_score}})]
        return []
'''


def _portfolio_source(*, class_name: str, description: str, symbols: list[str]) -> str:
    return f'''from trading.core.portfolio import Portfolio
from trading.core.classes import PriceData, MarketSignal, TickResults


class {class_name}(Portfolio):
    crucible_metadata = {{
        "schema_version": "1",
        "role": "portfolio",
        "description": {description!r},
        "tunables": {{
            "risk_per_trade": {{"type": "number", "default": 0.01, "min": 0.001, "max": 0.05}},
            "stop_pct": {{"type": "number", "default": 5.0, "min": 0.5, "max": 20.0}},
            "profit_pct": {{"type": "number", "default": 10.0, "min": 0.5, "max": 20.0}},
        }},
        "fixed_parameters": {{"symbols": {symbols!r}}},
        "signal_requirements": {{"accepted_signal_types": ["BUY", "SELL"], "uses_signal_strength": True}},
        "order_contract": {{"order_types": ["MARKET", "BRACKET"], "position_sizing": "fixed fractional", "exit_logic": "stop/profit target"}},
        "risk_controls": {{"max_risk_per_trade": "5%"}},
        "statefulness": "cash, positions, pending orders, history",
        "dependencies": [],
    }}

    def process_tick_market_signals_logic(
        self,
        signals: list[MarketSignal],
        tick: list[PriceData],
    ) -> TickResults:
        return TickResults(orders=[], output_tick=None, metadata={{"signals_seen": len(signals)}})
'''


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _symbols(payload: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    values = _list(payload.get("symbols")) or _list(constraints.get("symbols"))
    return values or ["SPY"]


def _list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    return []


def _class_name(value: str, *, suffix: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", value)
    base = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not base:
        base = "AgentGenerated"
    return base if base.endswith(suffix) else f"{base}{suffix}"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    progress: Progress | None, progress_pct: int, message: str, metadata: dict[str, Any] | None = None
) -> None:
    if progress is not None:
        progress(progress_pct, message, metadata)


def _emit_progress(progress_pct: int, message: str, metadata: dict[str, Any] | None = None) -> None:
    print(
        json.dumps(
            {
                "type": "progress",
                "progressPct": progress_pct,
                "message": message,
                "metadata": metadata or {},
                "time": _now(),
            },
            default=str,
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trading UI agent wrapper")
    parser.add_argument("command", choices=["idea-batch", "component-draft"])
    args = parser.parse_args(argv)
    payload = json.loads(sys.stdin.read() or "{}")
    result = (
        create_idea_batch(payload, _emit_progress)
        if args.command == "idea-batch"
        else create_component_draft(payload, _emit_progress)
    )
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
