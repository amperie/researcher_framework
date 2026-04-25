"""Trading research adapter scaffold.

This adapter defines the integration boundary for an algorithmic trading
platform. It intentionally avoids embedding a backtest engine in the researcher
repo; wire the platform-specific imports and calls here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from core.plugins.base import ResearchAdapter
from core.utils.logger import get_logger

log = get_logger(__name__)


class TradingAdapter(ResearchAdapter):
    """ResearchAdapter implementation boundary for trading experiments."""

    def validate_environment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        platform = profile.get("platform") or {}
        package = platform.get("package")
        source_path = platform.get("source_path")

        package_available = bool(package and importlib.util.find_spec(package))
        source_exists = bool(source_path and Path(source_path).expanduser().exists())

        return {
            "package": package,
            "package_available": package_available,
            "source_path": source_path,
            "source_exists": source_exists,
        }

    def build_context(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return {
            "data_sources": profile.get("data_sources") or [],
            "base_classes": profile.get("base_classes") or [],
            "risk_constraints": profile.get("risk_constraints") or {},
            "evaluation": profile.get("evaluation") or {},
        }

    def prepare_experiment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        proposals = state.get("proposals") or []
        artifacts = []

        for proposal in proposals:
            data_source = proposal.get("data_source") or _first_name(profile.get("data_sources") or [])
            artifacts.append({
                "artifact_id": f"{proposal.get('name', 'unknown')}_backtest_config",
                "artifact_type": "backtest_config",
                "proposal_name": proposal.get("name", "unknown"),
                "data_source": data_source,
                "universe": proposal.get("universe") or [],
                "timeframe": proposal.get("timeframe"),
                "start": proposal.get("start"),
                "end": proposal.get("end"),
                "transaction_cost_bps": proposal.get("transaction_cost_bps"),
                "slippage_bps": proposal.get("slippage_bps"),
                "risk_constraints": profile.get("risk_constraints") or {},
            })

        return {
            "experiment_artifacts": artifacts,
            "errors": list(state.get("errors") or []),
        }

    def execute_experiment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a trading backtest.

        Replace this method with calls into your trading platform's backtest
        engine. The returned dict should include normalized metrics such as
        sharpe_ratio, max_drawdown, annual_return, turnover, and win_rate.
        """
        raise NotImplementedError(
            "TradingAdapter.execute_experiment is a scaffold. Wire it to the "
            "algorithmic trading platform backtest engine before running the "
            "trading profile end to end."
        )

    def summarize_result(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        results = state.get("experiment_results") or []
        primary = (profile.get("evaluation") or {}).get("primary_metric")
        return {
            "primary_metric": primary,
            "results": [
                {
                    "proposal_name": result.get("proposal_name"),
                    "primary_metric_value": (result.get("metrics") or {}).get(primary) if primary else None,
                    "metrics": result.get("metrics") or {},
                    "risk": {
                        "max_drawdown": (result.get("metrics") or {}).get("max_drawdown"),
                        "turnover": (result.get("metrics") or {}).get("turnover"),
                        "exposure": (result.get("metrics") or {}).get("exposure"),
                    },
                }
                for result in results
            ],
        }


def get_adapter() -> TradingAdapter:
    return TradingAdapter()


def _first_name(items: list[dict[str, Any]]) -> str:
    return items[0].get("name", "") if items else ""
