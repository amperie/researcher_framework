from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.graph.nodes.validate import _build_contract_test
from core.plugins.trading.adapter import (
    TradingAdapter,
    _should_log_to_mlflow,
    _build_hpo_config,
    _build_alpaca_data_provider_params,
    _build_tune_run_config,
    _format_hpo_progress_message,
    _build_runtime_config,
    _normalize_runtime_config,
    _patch_trading_color_logger_pickling,
    _backtest_objective_with_mlflow_policy,
    _run_backtest_local_with_mlflow_policy,
    _resolve_alpaca_credentials,
    _safe_best_tune_result,
    _summarize_variant_results,
    _trial_status_counts,
    _trading_python,
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
                "start_date": "2023-01-01",
                "end_date": "2025-12-31",
                "limit": 350000,
                "market_hours_only": True,
                "aggregation_enabled": False,
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


def test_execute_experiment_delegates_to_shared_external_runner(tmp_path: Path):
    profile = _profile(tmp_path)
    artifact = {
        "artifact_id": "momentum_probe_trading_runtime",
        "proposal_name": "momentum_probe",
        "class_name": "ExampleAlgo",
        "script_path": str(tmp_path / "ExampleAlgo.py"),
        "config_path": str(tmp_path / "runtime.yaml"),
        "runtime_config": {"mode": "backtest", "hpo": {}, "data_provider": {"params": {"timeframe": "Minute"}}},
        "variant_specs": [{"name": "base", "overrides": {}}],
    }
    state = {
        "experiment_artifacts": [artifact],
        "proposals": [{"name": "momentum_probe", "description": "Test momentum"}],
    }
    cfg = SimpleNamespace(experiment_timeout_seconds=30, trading_python="python")

    with patch("core.plugins.trading.adapter.get_config", return_value=cfg):
        with patch("core.plugins.trading.adapter.run_task", return_value={"proposal_name": "momentum_probe", "metrics": {"sharpe_ratio": 1.0}}) as call_task:
            result = TradingAdapter().execute_experiment(profile, state)

    assert result["experiment_results"][0]["proposal_name"] == "momentum_probe"
    assert result["experiment_results"][0]["metrics"]["sharpe_ratio"] == 1.0
    assert result["experiment_results"][0]["artifacts"]["runtime_config_path"] == str(tmp_path / "runtime.yaml")
    assert result["experiment_results"][0]["artifacts"]["results_json_path"].endswith(".json")
    assert call_task.call_args.args[0]["task_path"] == "core.plugins.trading.tasks.run_trading_artifact"
    assert call_task.call_args.args[0]["python"] == "python"
    assert call_task.call_args.args[0]["cwd"] == str(tmp_path / "trading_guy")
    assert "output_dir" not in call_task.call_args.args[0]["payload"]


def test_external_runtime_spec_exposes_trading_runner_settings(tmp_path: Path):
    profile = _profile(tmp_path)
    cfg = SimpleNamespace(experiment_timeout_seconds=30, trading_python="python")
    (tmp_path / "trading_guy").mkdir()

    with patch("core.plugins.trading.adapter.get_config", return_value=cfg):
        spec = TradingAdapter().external_runtime_spec(profile, "validate")

    assert spec["python"] == "python"
    assert spec["plugin_name"] == "trading"
    assert str((tmp_path / "trading_guy").resolve()) in spec["pythonpath_entries"]


def test_trading_python_prefers_project_venv_interpreter(tmp_path: Path):
    profile = _profile(tmp_path)
    venv_python = tmp_path / "trading_guy" / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")
    cfg = SimpleNamespace(trading_python="uv run python")

    with patch("core.plugins.trading.adapter.get_config", return_value=cfg):
        python_path = _trading_python(profile)

    assert Path(python_path) == venv_python.resolve()


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
    assert cfg["data_provider"]["params"]["limit"] >= 304000
    assert cfg["portfolio"]["implementation"] == "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    assert cfg["portfolio"]["params"]["symbol"] == "SPY"
    assert cfg["aggregation"]["enabled"] is False
    assert "stop_pct" in cfg["hpo"]["portfolio_param_keys"]
    assert "profit_pct" in cfg["hpo"]["portfolio_param_keys"]
    assert "stop_pct" in cfg["hpo"]["search_space"]
    assert "profit_pct" in cfg["hpo"]["search_space"]
    assert cfg["optimization"]["target"] == "joint"
    assert cfg["generated_algorithm"]["class_name"] == "MeanReversionProbe"
    assert cfg["analysis"]["mlflow_policy"] == {
        "annualized_return_threshold": 0.0,
        "sample_negative_rate": 20,
    }
    assert cfg["algorithm"]["params"]["history_length"] >= 120


def test_build_runtime_config_adds_macro_symbols_and_dual_switch_portfolio(tmp_path: Path):
    profile = _profile(tmp_path)
    (tmp_path / "trading_guy").mkdir()
    (tmp_path / "trading_guy" / "accounts.yaml").write_text(
        "paper:\n  api_key: test_key\n  secret_key: test_secret\n",
        encoding="utf-8",
    )
    impl_path = tmp_path / "SwitchAlgo.py"
    impl_path.write_text(
        "class SwitchAlgo:\n"
        "    def read(self):\n"
        "        return 'VIX T10Y2Y'\n",
        encoding="utf-8",
    )
    proposal = {
        "name": "macro_switch",
        "description": "Switch using VIX and yield curve regime filters",
        "symbol": "SPY",
        "symbols": ["SPY", "UPRO", "SPXU"],
        "tradable_symbols": ["UPRO", "SPXU"],
    }
    implementation = {"class_name": "SwitchAlgo", "script_path": str(impl_path)}

    cfg = _build_runtime_config(profile, proposal, implementation)

    assert cfg["data_provider"]["params"]["symbols"] == ["SPY", "UPRO", "SPXU", "VIX", "T10Y2Y"]
    assert cfg["portfolio"]["implementation"] == "trading.core.pf.dual_symbol_switch_portfolio.DualSymbolSwitchPortfolio"
    assert cfg["portfolio"]["params"]["upro_symbol"] == "UPRO"
    assert cfg["portfolio"]["params"]["spxu_symbol"] == "SPXU"


def test_build_hpo_config_rebalances_misclassified_portfolio_keys(tmp_path: Path):
    profile = _profile(tmp_path)
    proposal = {
        "name": "macd_probe",
        "hpo": {
            "algorithm_param_keys": ["fast_period"],
            "portfolio_param_keys": ["position_size_base", "stop_pct"],
            "search_space": {
                "fast_period": {"type": "randint", "low": 4, "high": 40},
                "position_size_base": {"type": "randint", "low": 10, "high": 1000},
                "stop_pct": {"type": "uniform", "low": 1.0, "high": 8.0},
            },
        },
    }
    algorithm_params = {"symbol": "SPY", "history_length": 120, "fast_period": 12, "position_size_base": 100}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}

    hpo = _build_hpo_config(
        profile,
        proposal,
        algorithm_params,
        portfolio_params,
        profile["execution"]["defaults"],
        "backtest",
    )

    assert "position_size_base" in hpo["algorithm_param_keys"]
    assert "position_size_base" not in hpo["portfolio_param_keys"]


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
    assert "test_generated_algorithm_reconfigure_contract" in source
    assert "super().reconfigure(new_params)" in source


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
    assert hpo["search_space"]["fast_period"] == {"type": "randint", "low": 4, "high": 40}
    assert hpo["search_space"]["stop_pct"] == {"type": "uniform", "low": 0.5, "high": 20.0}
    assert hpo["search_space"]["profit_pct"] == {"type": "uniform", "low": 0.5, "high": 20.0}


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
    assert hpo["search_space"]["lookback"] == {"type": "randint", "low": 5, "high": 46}
    assert "threshold" in hpo["algorithm_param_keys"]
    assert hpo["search_space"]["threshold"]["type"] == "uniform"
    assert "stop_pct" in hpo["portfolio_param_keys"]
    assert "profit_pct" in hpo["portfolio_param_keys"]
    assert hpo["search_space"]["stop_pct"] == {"type": "uniform", "low": 0.5, "high": 20.0}
    assert hpo["search_space"]["profit_pct"] == {"type": "uniform", "low": 0.5, "high": 20.0}


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


def test_build_hpo_config_normalizes_lower_upper_bounds(tmp_path: Path):
    profile = _profile(tmp_path)
    algorithm_params = {"symbol": "SPY", "history_length": 120, "fast_period": 12, "threshold": 1.25}
    portfolio_params = {"symbol": "SPY", "cash": 100000.0, "stop_pct": 5.0, "profit_pct": 10.0}
    proposal = {
        "name": "macd_probe",
        "optimization_target": "algorithm",
        "hpo": {
            "algorithm_param_keys": ["fast_period", "threshold"],
            "search_space": {
                "fast_period": {"type": "randint", "lower": 4, "upper": 40},
                "threshold": {"type": "uniform", "min": 0.5, "max": 2.0},
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

    assert hpo["search_space"]["fast_period"] == {"type": "randint", "lower": 4, "upper": 40, "low": 4, "high": 40}
    assert hpo["search_space"]["threshold"] == {"type": "uniform", "min": 0.5, "max": 2.0, "low": 0.5, "high": 2.0}


def test_build_alpaca_data_provider_params_normalizes_timeframe_aliases(tmp_path: Path):
    profile = _profile(tmp_path)

    params = _build_alpaca_data_provider_params(
        profile=profile,
        proposal={"timeframe": "5min"},
        defaults=profile["execution"]["defaults"],
        symbol_ctx={"expected_symbols": ["SPY"]},
        algorithm_params={"history_length": 120},
    )
    assert params["timeframe"] == "Minute"


def test_backtest_analysis_uses_trading_guy_full_mlflow_pipeline(monkeypatch):
    class FakeMetrics:
        annualized_return = 12.5

    class FakeEngine:
        def __init__(self, portfolio, order_manager):
            self.logged = False

        def calculate_metrics(self):
            return FakeMetrics()

        def run_full_analysis(self, **kwargs):
            assert kwargs["log_to_mlflow"] is True
            assert kwargs["artifact_paths"] == ["runtime.yaml", "algorithm_config.json", "portfolio_config.json"]
            assert kwargs["run_name"] == "demo_run"
            return {"metrics": FakeMetrics(), "report": "ok", "trades": []}

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class FakeImportModule:
        def __call__(self, name):
            if name == "trading.engines.backtest_engine":
                class BacktestingEngine:
                    def __init__(self, cfg, dp, al, om, pf):
                        self.pf = pf

                    def run(self):
                        return None
                return type("M", (), {"BacktestingEngine": BacktestingEngine})
            if name == "trading.analysis.analysis_engine":
                return type("M", (), {"AnalysisEngine": FakeEngine})
            raise AssertionError(name)

    monkeypatch.setattr("core.plugins.trading.adapter.importlib.import_module", FakeImportModule())

    result = _run_backtest_local_with_mlflow_policy(
        backtest_cfg={
            "experiment_name": "exp",
            "starting_cash": 100000.0,
            "run_name": "demo_run",
            "description": "desc",
            "config_artifact_path": "runtime.yaml",
            "git_tags": {},
            "benchmark_paths": {},
        },
        alg_cfg={"symbol": "SPY", "history_length": 120},
        pf_cfg={"symbol": "SPY"},
        dp_cfg={"symbols": ["SPY"], "timeframe": "Minute", "limit": 1000},
        algorithm_class=Dummy,
        portfolio_class=Dummy,
        data_provider_class=Dummy,
        order_manager_class=Dummy,
        mlflow_policy={"annualized_return_threshold": 0.0, "sample_negative_rate": 20},
        config_artifact_paths=["algorithm_config.json", "portfolio_config.json"],
    )

    assert result["report"] == "ok"


def test_should_log_to_mlflow_always_logs_positive_annualized_return():
    should_log = _should_log_to_mlflow(
        annualized_return=0.01,
        run_name="positive_run",
        params={"fast_period": 12},
        sample_negative_rate=20,
        annualized_return_threshold=0.0,
    )

    assert should_log is True


def test_backtest_objective_returns_finite_metric_on_failure():
    result = _backtest_objective_with_mlflow_policy(
        config={},
        symbol="SPY",
        algorithm_class=object,
        portfolio_class=object,
        data_provider_class=object,
        order_manager_class=object,
        base_algorithm_config={},
        base_portfolio_config={},
        base_data_provider_config={},
        base_backtest_config={},
        algorithm_param_keys=[],
        portfolio_param_keys=[],
        mlflow_policy={},
    )

    assert result["_metric"] == -1_000_000_000.0
    assert "_trial_error" in result


def test_safe_best_tune_result_falls_back_when_ray_has_no_best_trial():
    class BrokenResults:
        def get_best_result(self, **kwargs):
            raise RuntimeError("no valid trials")

    best = _safe_best_tune_result(BrokenResults())

    assert best.config == {}


def test_trial_status_counts_groups_ray_statuses():
    trials = [
        SimpleNamespace(status="PENDING"),
        SimpleNamespace(status="RUNNING"),
        SimpleNamespace(status="TERMINATED"),
        SimpleNamespace(status="ERROR"),
        SimpleNamespace(status="PAUSED"),
        SimpleNamespace(status="RESTORING"),
    ]

    counts = _trial_status_counts(trials)

    assert counts == {
        "queued": 1,
        "running": 1,
        "done": 1,
        "errored": 1,
        "paused": 1,
        "other": 1,
    }


def test_format_hpo_progress_message_includes_status_and_best_metric():
    trials = [
        SimpleNamespace(status="RUNNING", last_result={"_metric": 0.25}),
        SimpleNamespace(status="TERMINATED", last_result={"_metric": 1.5}),
        SimpleNamespace(status="ERROR", last_result={"_metric": float("nan")}),
    ]

    message = _format_hpo_progress_message(
        run_name="demo_run",
        trials=trials,
        total_samples=5,
        metric_name="_metric",
        done=False,
    )

    assert "run_name=demo_run" in message
    assert "done=1/5" in message
    assert "running=1" in message
    assert "errored=1" in message
    assert "best__metric=1.500000" in message


def test_build_tune_run_config_filters_unsupported_kwargs():
    class FakeRunConfig:
        def __init__(self, *, verbose=None):
            self.verbose = verbose

    fake_tune = SimpleNamespace(RunConfig=FakeRunConfig, Callback=type("Callback", (), {}))

    run_config = _build_tune_run_config(
        tune=fake_tune,
        run_name="demo_run",
        total_samples=10,
        metric_name="_metric",
        progress_report_interval_seconds=5,
        log_progress=True,
    )

    assert isinstance(run_config, FakeRunConfig)
    assert run_config.verbose == 1


def test_patch_trading_color_logger_pickling_recovers_missing_logger_state(monkeypatch):
    import logging
    import sys
    from types import ModuleType

    logger_module = ModuleType("utils.logger")

    class FakeColorLogger:
        def __init__(self, logger):
            self._logger = logger

        def __getattr__(self, name):
            return getattr(self._logger, name)

    logger_module.ColorLogger = FakeColorLogger
    monkeypatch.setitem(sys.modules, "utils.logger", logger_module)

    _patch_trading_color_logger_pickling()

    wrapped = FakeColorLogger(logging.getLogger("demo"))
    state = wrapped.__getstate__()
    restored = FakeColorLogger.__new__(FakeColorLogger)
    restored.__setstate__(state)

    assert restored.name == "demo"
    assert getattr(FakeColorLogger, "_research_pickle_patch", False) is True


def test_should_log_to_mlflow_samples_negative_runs_deterministically():
    profile = _profile(Path("."))
    sampled = 0
    total = 200
    for idx in range(total):
        if _should_log_to_mlflow(
            annualized_return=-1.0,
            run_name=f"negative_run_{idx}",
            params={"fast_period": idx + 1},
            sample_negative_rate=20,
            annualized_return_threshold=0.0,
        ):
            sampled += 1

    assert 3 <= sampled <= 20

    with patch("core.plugins.trading.adapter._resolve_alpaca_credentials", return_value={"api_key": "k", "secret_key": "s"}):
        params = _build_alpaca_data_provider_params(
            profile=profile,
            proposal={"timeframe": "1h"},
            defaults=profile["execution"]["defaults"],
            symbol_ctx={"expected_symbols": ["SPY"]},
            algorithm_params={"history_length": 120},
        )
    assert params["timeframe"] == "Hour"


def test_normalize_runtime_config_repairs_stale_runtime_artifacts():
    runtime_cfg = {
        "data_provider": {
            "params": {
                "timeframe": "5min",
            },
        },
        "hpo": {
            "search_space": {
                "fast_period": {"type": "randint", "lower": 4, "upper": 40},
                "threshold": {"type": "uniform", "min": 0.5, "max": 2.0},
            },
        },
    }

    normalized = _normalize_runtime_config(runtime_cfg)

    assert normalized["data_provider"]["params"]["timeframe"] == "Minute"
    assert normalized["hpo"]["search_space"]["fast_period"]["low"] == 4
    assert normalized["hpo"]["search_space"]["fast_period"]["high"] == 40
    assert normalized["hpo"]["search_space"]["threshold"]["low"] == 0.5
    assert normalized["hpo"]["search_space"]["threshold"]["high"] == 2.0
