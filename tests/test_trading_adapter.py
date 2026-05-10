from __future__ import annotations

from pathlib import Path

from core.graph.nodes.validate import _build_contract_test
from core.plugins.trading.adapter import (
    TradingAdapter,
    _build_hpo_config,
    _build_runtime_config,
    _resolve_alpaca_credentials,
    _summarize_variant_results,
    _variant_specs,
)


def _profile(tmp_path: Path) -> dict:
    platform_root = tmp_path / "trading_guy"
    return {
        "name": "trading",
        "platform": {"package": "trading", "source_path": str(platform_root)},
        "execution": {
            "defaults": {
                "mode": "backtest",
                "symbol": "SPY",
                "cash": 100000,
                "history_length": 120,
                "stop_pct": 5.0,
                "profit_pct": 10.0,
                "tx_cost": 0.0,
                "data_provider": "trading.data_providers.alpaca_data_provider.AlpacaDataProvider",
                "portfolio": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
                "order_manager": "trading.core.om.backtesting_om.BacktestingOrderManager",
                "alpaca_account": "paper",
                "alpaca_account_path": "accounts.yaml",
                "timeframe": "Minute",
                "adjustment": "split",
                "limit": 5000,
                "market_hours_only": True,
                "log_to_mlflow": False,
                "walk_forward": {"optimization_window_days": 90},
            },
            "mass_test": {"max_variants_per_proposal": 3},
        },
        "storage": {"mlflow_experiment": "trading_agent_experiments"},
    }


def test_prepare_experiment_builds_runtime_artifact(tmp_path: Path):
    profile = _profile(tmp_path)
    adapter = TradingAdapter()
    impl_path = tmp_path / "ExampleAlgo.py"
    impl_path.write_text("class ExampleAlgo:\n    pass\n", encoding="utf-8")
    (tmp_path / "trading_guy").mkdir()
    (tmp_path / "trading_guy" / "accounts.yaml").write_text(
        "paper:\n  api_key: test_key\n  secret_key: test_secret\n",
        encoding="utf-8",
    )

    state = {
        "proposals": [{
            "name": "momentum_probe",
            "description": "Test momentum",
            "mode": "walk-forward",
            "symbol": "SPY",
            "history_length": 200,
            "hyperparameters": {"lookback": 15},
        }],
        "implementations": [{
            "proposal_name": "momentum_probe",
            "class_name": "ExampleAlgo",
            "script_path": str(impl_path),
        }],
        "validation_results": [{"proposal_name": "momentum_probe", "passed": True}],
    }

    result = adapter.prepare_experiment(profile, state)
    artifact = result["experiment_artifacts"][0]
    assert artifact["proposal_name"] == "momentum_probe"
    assert artifact["mode"] == "walk-forward"
    assert artifact["runtime_config"]["algorithm"]["params"]["lookback"] == 15
    assert Path(artifact["config_path"]).exists()


def test_build_runtime_config_uses_profile_defaults(tmp_path: Path):
    profile = _profile(tmp_path)
    (tmp_path / "trading_guy").mkdir()
    (tmp_path / "trading_guy" / "accounts.yaml").write_text(
        "paper:\n  api_key: test_key\n  secret_key: test_secret\n",
        encoding="utf-8",
    )
    proposal = {"name": "mean_reversion_probe", "description": "Test mean reversion"}
    implementation = {"class_name": "MeanReversionProbe", "script_path": "x.py"}

    cfg = _build_runtime_config(profile, proposal, implementation)
    assert cfg["mode"] == "backtest"
    assert cfg["data_provider"]["implementation"].endswith("AlpacaDataProvider")
    assert cfg["data_provider"]["params"]["symbols"] == ["SPY"]
    assert cfg["data_provider"]["params"]["api_key"] == "test_key"
    assert cfg["data_provider"]["params"]["limit"] == 5000
    assert cfg["portfolio"]["implementation"] == "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    assert cfg["portfolio"]["params"]["symbol"] == "SPY"
    assert "stop_pct" in cfg["hpo"]["portfolio_param_keys"]
    assert "profit_pct" in cfg["hpo"]["portfolio_param_keys"]
    assert "stop_pct" in cfg["hpo"]["search_space"]
    assert "profit_pct" in cfg["hpo"]["search_space"]
    assert cfg["optimization"]["target"] == "joint"
    assert cfg["generated_algorithm"]["class_name"] == "MeanReversionProbe"


def test_variant_specs_caps_and_keeps_base(tmp_path: Path):
    profile = _profile(tmp_path)
    proposal = {
        "experiment_variants": [
            {"name": "agg_5", "overrides": {"aggregation": {"enabled": True, "aggregation_period_minutes": 5}}},
            {"name": "agg_15", "overrides": {"aggregation": {"enabled": True, "aggregation_period_minutes": 15}}},
            {"name": "wf_short", "overrides": {"walk_forward": {"trading_window_days": 15}}},
        ]
    }

    specs = _variant_specs(profile, proposal)
    assert specs[0]["name"] == "base"
    assert len(specs) == 3
    assert specs[1]["name"] == "agg_5"


def test_summarize_variant_results_aggregates_metrics():
    summary = _summarize_variant_results([
        {"variant_name": "base", "metrics": {"sharpe_ratio": 1.0, "max_drawdown_pct": -8.0, "total_trades": 10}, "report": "base"},
        {"variant_name": "agg_5", "metrics": {"sharpe_ratio": 2.0, "max_drawdown_pct": -12.0, "total_trades": 12}, "report": "agg"},
    ])
    metrics = summary["metrics"]
    assert metrics["sharpe_ratio"] == 1.5
    assert metrics["max_drawdown_pct"] == -12.0
    assert metrics["total_trades"] == 22
    assert metrics["variant_count"] == 2


def test_trading_contract_test_includes_platform_root(tmp_path: Path):
    profile = _profile(tmp_path)
    source = _build_contract_test(
        profile=profile,
        contract_test="trading_algorithm",
        script_path=str(tmp_path / "Algo.py"),
        class_name="Algo",
        expected_feature_set_name="algo",
    )
    assert "PLATFORM_ROOT" in source
    assert "Signals before the final bar" in source


def test_resolve_alpaca_credentials_prefers_env_when_present(tmp_path: Path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setenv("ALPACA_API_KEY", "env_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env_secret")

    creds = _resolve_alpaca_credentials(
        profile=profile,
        proposal={},
        defaults=profile["execution"]["defaults"],
    )

    assert creds == {"api_key": "env_key", "secret_key": "env_secret"}


def test_build_hpo_config_preserves_explicit_search_space_and_keys(tmp_path: Path):
    profile = _profile(tmp_path)
    proposal = {
        "name": "macd_probe",
        "hpo": {
            "num_samples": 80,
            "algorithm_param_keys": ["fast_period"],
            "portfolio_param_keys": ["stop_pct", "profit_pct"],
            "search_space": {
                "fast_period": {"type": "randint", "low": 4, "high": 40},
                "stop_pct": {"type": "uniform", "low": 1.0, "high": 8.0},
                "profit_pct": {"type": "uniform", "low": 2.0, "high": 20.0},
            },
        },
    }
    algorithm_params = {"symbol": "SPY", "history_length": 120, "fast_period": 12}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}

    hpo = _build_hpo_config(
        profile,
        proposal,
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert hpo["num_samples"] == 80
    assert hpo["algorithm_param_keys"] == ["fast_period"]
    assert hpo["portfolio_param_keys"] == ["stop_pct", "profit_pct"]
    assert hpo["search_space"]["fast_period"]["high"] == 40


def test_build_hpo_config_infers_wide_search_space_for_backtest(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "lookback": 15, "threshold": 1.25}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}

    hpo = _build_hpo_config(
        profile,
        {"name": "mean_reversion"},
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert "lookback" in hpo["algorithm_param_keys"]
    assert hpo["search_space"]["lookback"]["type"] == "randint"
    assert "threshold" in hpo["algorithm_param_keys"]
    assert hpo["search_space"]["threshold"]["type"] == "uniform"
    assert "stop_pct" in hpo["portfolio_param_keys"]
    assert "profit_pct" in hpo["portfolio_param_keys"]


def test_build_hpo_config_algorithm_target_excludes_portfolio_tuning(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "lookback": 15, "threshold": 1.25}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}

    hpo = _build_hpo_config(
        profile,
        {"name": "mean_reversion", "optimization_target": "algorithm"},
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert hpo["optimization_target"] == "algorithm"
    assert sorted(hpo["algorithm_param_keys"]) == ["lookback", "threshold"]
    assert hpo["portfolio_param_keys"] == []
    assert "stop_pct" not in hpo["search_space"]
    assert "profit_pct" not in hpo["search_space"]


def test_build_hpo_config_portfolio_target_excludes_algorithm_tuning(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "lookback": 15, "threshold": 1.25}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}

    hpo = _build_hpo_config(
        profile,
        {"name": "mean_reversion", "optimization_target": "portfolio"},
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert hpo["optimization_target"] == "portfolio"
    assert hpo["algorithm_param_keys"] == []
    assert sorted(hpo["portfolio_param_keys"]) == ["profit_pct", "stop_pct"]
    assert set(hpo["search_space"]) == {"stop_pct", "profit_pct"}


def test_build_hpo_config_filters_walk_forward_keys_by_target(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "lookback": 15}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}
    proposal = {
        "name": "wf_probe",
        "optimization_target": "portfolio",
        "hpo": {
            "algorithm_param_keys": ["lookback"],
            "portfolio_param_keys": ["stop_pct", "profit_pct"],
            "search_space": {
                "lookback": {"type": "randint", "low": 5, "high": 30},
                "stop_pct": {"type": "uniform", "low": 1.0, "high": 8.0},
                "profit_pct": {"type": "uniform", "low": 2.0, "high": 20.0},
            },
        },
    }

    hpo = _build_hpo_config(
        profile,
        proposal,
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "walk-forward",
    )

    assert hpo["optimization_target"] == "portfolio"
    assert hpo["algorithm_param_keys"] == []
    assert sorted(hpo["portfolio_param_keys"]) == ["profit_pct", "stop_pct"]
    assert set(hpo["search_space"]) == {"stop_pct", "profit_pct"}


def test_build_hpo_config_omits_non_continuous_inferred_params(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {
        "symbol": "SPY",
        "history_length": 120,
        "lookback": 15,
        "use_filter": True,
        "regime": "trend",
    }
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}

    hpo = _build_hpo_config(
        profile,
        {"name": "mean_reversion", "optimization_target": "algorithm"},
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert hpo["algorithm_param_keys"] == ["lookback"]
    assert set(hpo["search_space"]) == {"lookback"}


def test_build_hpo_config_converts_numeric_choice_space_to_ranges(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "fast_period": 12, "threshold": 1.25}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}
    proposal = {
        "name": "macd_probe",
        "hpo": {
            "algorithm_param_keys": ["fast_period", "threshold"],
            "search_space": {
                "fast_period": {"type": "choice", "values": [4, 8, 16, 32]},
                "threshold": {"type": "choice", "values": [0.5, 1.0, 2.0]},
            },
        },
    }

    hpo = _build_hpo_config(
        profile,
        proposal,
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert hpo["search_space"]["fast_period"] == {"type": "randint", "low": 4, "high": 33}
    assert hpo["search_space"]["threshold"] == {"type": "uniform", "low": 0.5, "high": 2.0}


def test_build_hpo_config_drops_nonnumeric_choice_space(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "lookback": 15, "regime": "trend"}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}
    proposal = {
        "name": "macd_probe",
        "optimization_target": "algorithm",
        "hpo": {
            "algorithm_param_keys": ["lookback", "regime"],
            "search_space": {
                "lookback": {"type": "randint", "low": 5, "high": 30},
                "regime": {"type": "choice", "values": ["trend", "mean_revert"]},
            },
        },
    }

    hpo = _build_hpo_config(
        profile,
        proposal,
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert hpo["algorithm_param_keys"] == ["lookback"]
    assert set(hpo["search_space"]) == {"lookback"}
