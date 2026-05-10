"""Trading research adapter backed by the local ``trading_guy`` project."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any
from uuid import uuid4

import yaml

from configs.config import get_config, resolve_dev_path
from core.plugins.base import ResearchAdapter
from core.utils.logger import get_logger

log = get_logger(__name__)


class TradingAdapter(ResearchAdapter):
    """ResearchAdapter implementation for the local trading framework."""

    def validate_environment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        platform_root = _platform_root(profile)
        package_name = str((profile.get("platform") or {}).get("package") or "trading")
        provider_name = _default_provider(profile)
        provider_uses_local_data = "alpaca" not in provider_name.lower()
        validation: dict[str, Any] = {
            "package": package_name,
            "source_path": str(platform_root),
            "source_exists": platform_root.exists(),
            "run_py_exists": (platform_root / "run.py").exists(),
            "data_dir_exists": (platform_root / "data").exists(),
            "default_data_provider": provider_name,
            "default_data_exists": (
                _platform_data_path(profile, _default_data_path(profile)).exists()
                if provider_uses_local_data
                else True
            ),
        }
        try:
            with _sys_path(platform_root):
                importlib.import_module(package_name)
            validation["package_available"] = True
        except Exception as exc:
            validation["package_available"] = False
            validation["import_error"] = str(exc)
        return validation

    def build_context(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        data_sources = []
        for item in profile.get("data_sources") or []:
            enriched = dict(item)
            rel_path = str(item.get("path") or "")
            if rel_path:
                enriched["resolved_path"] = str(_platform_data_path(profile, rel_path))
            data_sources.append(enriched)
        return {
            "data_sources": data_sources,
            "base_classes": profile.get("base_classes") or [],
            "risk_constraints": profile.get("risk_constraints") or {},
            "evaluation": profile.get("evaluation") or {},
            "execution": profile.get("execution") or {},
        }

    def prepare_experiment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        proposals = list(state.get("proposals") or [])
        implementations = {
            str(item.get("proposal_name") or ""): item
            for item in (state.get("implementations") or [])
            if item.get("proposal_name")
        }
        validation_by_name = {
            str(item.get("proposal_name") or ""): item
            for item in (state.get("validation_results") or [])
            if item.get("proposal_name")
        }
        artifacts: list[dict[str, Any]] = []
        errors = list(state.get("errors") or [])
        cfg_dir = resolve_dev_path("dev/experiments/trading/runtime_configs")
        cfg_dir.mkdir(parents=True, exist_ok=True)

        for proposal in proposals:
            proposal_name = str(proposal.get("name") or "unknown")
            implementation = implementations.get(proposal_name)
            validation = validation_by_name.get(proposal_name) or {}
            if not implementation or not implementation.get("script_path"):
                errors.append(f"prepare_experiment: {proposal_name} has no generated implementation")
                continue
            if validation.get("passed") is False:
                errors.append(f"prepare_experiment: {proposal_name} did not pass validation")
                continue

            runtime_cfg = _build_runtime_config(profile, proposal, implementation)
            cfg_path = cfg_dir / f"{implementation.get('class_name', proposal_name)}.yaml"
            cfg_path.write_text(yaml.safe_dump(runtime_cfg, sort_keys=False), encoding="utf-8")

            artifacts.append({
                "artifact_id": f"{proposal_name}_trading_runtime",
                "artifact_type": "trading_runtime_config",
                "proposal_name": proposal_name,
                "class_name": implementation.get("class_name", ""),
                "script_path": implementation.get("script_path", ""),
                "config_path": str(cfg_path),
                "mode": runtime_cfg.get("mode", "backtest"),
                "runtime_config": runtime_cfg,
                "variant_specs": _variant_specs(profile, proposal),
                "data_path": runtime_cfg.get("data_provider", {}).get("path", ""),
            })

        return {
            "experiment_artifacts": artifacts,
            "errors": errors,
        }

    def execute_experiment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        artifacts = list(state.get("experiment_artifacts") or [])
        errors = list(state.get("errors") or [])
        results: list[dict[str, Any]] = []
        output_dir = resolve_dev_path("dev/experiments/trading/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        platform_root = _platform_root(profile)

        with _sys_path(platform_root):
            for artifact in artifacts:
                proposal_name = str(artifact.get("proposal_name") or "unknown")
                class_name = str(artifact.get("class_name") or proposal_name)
                script_path = str(artifact.get("script_path") or "")
                runtime_cfg = dict(artifact.get("runtime_config") or {})
                variant_specs = list(artifact.get("variant_specs") or [{"name": "base", "overrides": {}}])
                try:
                    variant_results = [
                        self._run_variant(
                            profile=profile,
                            proposal_name=proposal_name,
                            class_name=class_name,
                            script_path=script_path,
                            runtime_cfg=runtime_cfg,
                            variant_spec=variant_spec,
                        )
                        for variant_spec in variant_specs
                    ]
                    summary = _summarize_variant_results(variant_results)
                    experiment_id = str(uuid4())
                    result_path = output_dir / f"{class_name}_{experiment_id[:8]}.json"
                    result_payload = {
                        "experiment_id": experiment_id,
                        "proposal_name": proposal_name,
                        "class_name": class_name,
                        "mode": runtime_cfg.get("mode", "backtest"),
                        "metrics": summary["metrics"],
                        "variants": variant_results,
                    }
                    result_path.write_text(json.dumps(result_payload, indent=2, default=_json_default), encoding="utf-8")
                    results.append({
                        "experiment_id": experiment_id,
                        "proposal_name": proposal_name,
                        "proposal": next((item for item in (state.get("proposals") or []) if item.get("name") == proposal_name), {}),
                        "metrics": summary["metrics"],
                        "execution_config": _public_runtime_config(runtime_cfg),
                        "artifacts": {
                            "runtime_config_path": artifact.get("config_path", ""),
                            "results_json_path": str(result_path),
                            "variant_count": len(variant_results),
                            "runtime_config": _public_runtime_config(runtime_cfg),
                        },
                        "variant_results": variant_results,
                        "report": summary.get("report", ""),
                    })
                except Exception as exc:
                    log.error("TradingAdapter.execute_experiment | %s failed: %s", proposal_name, exc, exc_info=True)
                    errors.append(f"execute_experiment: {proposal_name} failed: {exc}")

        return {
            "experiment_results": results,
            "errors": errors,
        }

    def summarize_result(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        results = list(state.get("experiment_results") or [])
        primary_metric = str(((profile.get("evaluation") or {}).get("primary_metric")) or "sharpe_ratio")
        ranked = sorted(
            results,
            key=lambda item: float((item.get("metrics") or {}).get(primary_metric, float("-inf")) or float("-inf")),
            reverse=True,
        )
        return {
            "primary_metric": primary_metric,
            "best_result": ranked[0] if ranked else {},
            "results": [
                {
                    "proposal_name": item.get("proposal_name"),
                    "primary_metric_value": (item.get("metrics") or {}).get(primary_metric),
                    "metrics": item.get("metrics") or {},
                    "variant_count": len(item.get("variant_results") or []),
                }
                for item in ranked
            ],
        }

    def _run_variant(
        self,
        *,
        profile: dict[str, Any],
        proposal_name: str,
        class_name: str,
        script_path: str,
        runtime_cfg: dict[str, Any],
        variant_spec: dict[str, Any],
    ) -> dict[str, Any]:
        trading = _load_trading_runtime()
        variant_name = str(variant_spec.get("name") or "base")
        merged_cfg = _deep_merge(runtime_cfg, dict(variant_spec.get("overrides") or {}))
        algorithm_cls = _load_algorithm_class(script_path, class_name)
        mode = str(merged_cfg.get("mode") or "backtest")

        if mode == "walk-forward":
            built = _build_runtime_components(trading, merged_cfg, algorithm_cls)
            engine = trading["WalkForwardEngine"](
                cfg=merged_cfg,
                dp=built["data_provider"],
                al=built["algorithm"],
                om=built["order_manager"],
                pf=built["portfolio"],
            )
            run_output = engine.run()
            period_results = list(run_output.get("periods") or [])
            aggregate = dict(run_output.get("aggregate") or {})
            metrics = _walk_forward_metrics(period_results, aggregate)
            report = json.dumps(aggregate, indent=2)
            raw_output = {
                "aggregate": aggregate,
                "periods": [_serialize(item) for item in period_results],
            }
        else:
            hpo_result = _run_hpo_backtest(merged_cfg, algorithm_cls)
            metrics = dict(hpo_result["metrics"])
            report = str(hpo_result["report"])
            raw_output = dict(hpo_result["raw_output"])

        return {
            "variant_name": variant_name,
            "mode": mode,
            "metrics": metrics,
            "config": merged_cfg,
            "report": report,
            "raw_output": raw_output,
        }


def get_adapter() -> TradingAdapter:
    return TradingAdapter()


def _platform_root(profile: dict[str, Any]) -> Path:
    source_path = str(((profile.get("platform") or {}).get("source_path")) or "../trading_guy")
    return Path(source_path).expanduser().resolve()


class _sys_path:
    def __init__(self, *paths: Path):
        self.paths = [str(path.resolve()) for path in paths if path]
        self.added: list[str] = []

    def __enter__(self):
        for path in reversed(self.paths):
            if path not in sys.path:
                sys.path.insert(0, path)
                self.added.append(path)
        return self

    def __exit__(self, exc_type, exc, tb):
        for path in self.added:
            while path in sys.path:
                sys.path.remove(path)
        return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _platform_data_path(profile: dict[str, Any], rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return _platform_root(profile) / rel_path


def _default_data_path(profile: dict[str, Any]) -> str:
    return str((((profile.get("execution") or {}).get("defaults") or {}).get("data_path")) or "data/SPY_5min.csv")


def _default_portfolio(profile: dict[str, Any]) -> str:
    return str((((profile.get("execution") or {}).get("defaults") or {}).get("portfolio")) or "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio")


def _default_provider(profile: dict[str, Any]) -> str:
    return str((((profile.get("execution") or {}).get("defaults") or {}).get("data_provider")) or "trading.data_providers.test_data_provider.TestDataProvider")


def _default_order_manager(profile: dict[str, Any]) -> str:
    return str((((profile.get("execution") or {}).get("defaults") or {}).get("order_manager")) or "trading.core.om.backtesting_om.BacktestingOrderManager")


def _optimization_target(proposal: dict[str, Any]) -> str:
    raw = str(proposal.get("optimization_target") or proposal.get("optimize_target") or "joint").strip().lower()
    alias_map = {
        "algorithm": "algorithm",
        "algorithm_only": "algorithm",
        "algo": "algorithm",
        "portfolio": "portfolio",
        "portfolio_only": "portfolio",
        "joint": "joint",
        "both": "joint",
        "co_optimize": "joint",
        "co-optimize": "joint",
    }
    return alias_map.get(raw, "joint")


def _build_runtime_config(profile: dict[str, Any], proposal: dict[str, Any], implementation: dict[str, Any]) -> dict[str, Any]:
    execution_cfg = profile.get("execution") or {}
    defaults = execution_cfg.get("defaults") or {}
    mode = str(proposal.get("mode") or defaults.get("mode") or "backtest")
    symbol = str(proposal.get("symbol") or proposal.get("universe") or defaults.get("symbol") or "SPY")
    history_length = int(proposal.get("history_length") or defaults.get("history_length") or 200)
    run_name = f"{proposal.get('name', implementation.get('class_name', 'strategy'))}_{mode}"
    algorithm_params = {
        "symbol": symbol,
        "history_length": history_length,
        **dict(proposal.get("hyperparameters") or {}),
    }
    portfolio_params = {
        "symbol": symbol,
        "cash": float(proposal.get("cash") or defaults.get("cash") or 100000),
        "keep_history": True,
        "stop_pct": float(proposal.get("stop_pct") or defaults.get("stop_pct") or 5.0),
        "profit_pct": float(proposal.get("profit_pct") or defaults.get("profit_pct") or 10.0),
        "tx_cost": float(proposal.get("tx_cost") or defaults.get("tx_cost") or 0.0),
    }
    data_provider_impl = str(proposal.get("data_provider") or defaults.get("data_provider") or _default_provider(profile))
    data_provider = _build_data_provider_params(
        profile=profile,
        proposal=proposal,
        defaults=defaults,
        symbol=symbol,
        provider_implementation=data_provider_impl,
    )

    runtime_cfg: dict[str, Any] = {
        "mode": mode,
        "algorithm": {
            "implementation": "__generated__",
            "params": algorithm_params,
        },
        "portfolio": {
            "implementation": _portfolio_implementation_for_mode(profile, proposal, mode),
            "params": portfolio_params,
        },
        "order_manager": {
            "implementation": str(proposal.get("order_manager") or defaults.get("order_manager") or _default_order_manager(profile)),
            "params": dict(proposal.get("order_manager_params") or defaults.get("order_manager_params") or {}),
        },
        "data_provider": {
            "implementation": data_provider_impl,
            "params": data_provider,
        },
        "analysis": {
            "enabled": True,
            "log_to_mlflow": bool(defaults.get("log_to_mlflow", False)),
            "experiment_name": str(((profile.get("storage") or {}).get("mlflow_experiment")) or "trading_agent_experiments"),
            "run_name": run_name,
            "description": str(proposal.get("description") or ""),
            "benchmarks": dict(proposal.get("benchmarks") or defaults.get("benchmarks") or {}),
        },
        "aggregation": {
            "enabled": bool(proposal.get("aggregation_enabled") or defaults.get("aggregation_enabled") or False),
            "aggregation_period_minutes": int(proposal.get("aggregation_period_minutes") or defaults.get("aggregation_period_minutes") or 5),
        },
        "walk_forward": _deep_merge(
            dict(defaults.get("walk_forward") or {}),
            dict(proposal.get("walk_forward") or {}),
        ),
        "hpo": _build_hpo_config(profile, proposal, algorithm_params, portfolio_params, defaults, mode),
        "optimization": {
            "target": _optimization_target(proposal),
        },
        "mlflow": {"enabled": bool(defaults.get("log_to_mlflow", False))},
        "state_store": {"enabled": False},
        "logging": {},
        "generated_algorithm": {
            "class_name": implementation.get("class_name", ""),
            "script_path": implementation.get("script_path", ""),
        },
    }
    return runtime_cfg


def _portfolio_implementation_for_mode(profile: dict[str, Any], proposal: dict[str, Any], mode: str) -> str:
    if mode == "backtest":
        return "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    return str(proposal.get("portfolio") or _default_portfolio(profile))


def _build_hpo_config(
    profile: dict[str, Any],
    proposal: dict[str, Any],
    algorithm_params: dict[str, Any],
    portfolio_params: dict[str, Any],
    defaults: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    base = dict(defaults.get("hpo") or {})
    proposal_hpo = dict(proposal.get("hpo") or {})
    optimization_target = _optimization_target(proposal)

    if mode != "backtest":
        return _filter_hpo_config(
            _deep_merge(base, proposal_hpo),
            optimization_target=optimization_target,
            algorithm_params=algorithm_params,
            portfolio_params=portfolio_params,
        )

    search_space, algorithm_keys, portfolio_keys = _resolve_backtest_hpo_space(
        proposal=proposal,
        algorithm_params=algorithm_params,
        portfolio_params=portfolio_params,
        optimization_target=optimization_target,
    )
    hpo_cfg = _deep_merge(base, proposal_hpo)
    hpo_cfg["objective_metric"] = str(
        proposal_hpo.get("objective_metric")
        or proposal_hpo.get("optimization_metric")
        or "annualized_return"
    )
    hpo_cfg["num_samples"] = int(proposal_hpo.get("num_samples") or proposal_hpo.get("n_trials") or 50)
    hpo_cfg["max_concurrent_trials"] = int(proposal_hpo.get("max_concurrent_trials") or 8)
    hpo_cfg["search_space"] = search_space
    hpo_cfg["algorithm_param_keys"] = algorithm_keys
    hpo_cfg["portfolio_param_keys"] = portfolio_keys
    hpo_cfg["optimization_target"] = optimization_target
    return hpo_cfg


def _resolve_backtest_hpo_space(
    *,
    proposal: dict[str, Any],
    algorithm_params: dict[str, Any],
    portfolio_params: dict[str, Any],
    optimization_target: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    proposal_hpo = dict(proposal.get("hpo") or {})
    search_space = _normalize_search_space(dict(proposal_hpo.get("search_space") or {}))
    algorithm_keys = [str(item) for item in (proposal_hpo.get("algorithm_param_keys") or []) if item]
    portfolio_keys = [str(item) for item in (proposal_hpo.get("portfolio_param_keys") or []) if item]

    tunable_params = proposal_hpo.get("tunable_params") or {}
    if not search_space and isinstance(tunable_params, dict):
        for key, spec in tunable_params.items():
            if isinstance(spec, dict):
                normalized_spec = _normalize_search_spec(str(key), dict(spec))
                if normalized_spec:
                    search_space[str(key)] = normalized_spec

    if not search_space:
        search_space = _infer_wide_search_space(algorithm_params, portfolio_params)

    if not algorithm_keys:
        algorithm_keys = [key for key in search_space if key in algorithm_params]
    if not portfolio_keys:
        portfolio_keys = [key for key in search_space if key in portfolio_params]

    if optimization_target in {"portfolio", "joint"}:
        for required_key in ("stop_pct", "profit_pct"):
            if required_key not in search_space:
                search_space[required_key] = _default_portfolio_search_space(required_key, float(portfolio_params.get(required_key, 0.0) or 0.0))
            if required_key not in portfolio_keys:
                portfolio_keys.append(required_key)

    algorithm_keys = [key for key in algorithm_keys if key in search_space]
    portfolio_keys = [key for key in portfolio_keys if key in search_space]
    return _filter_search_space_for_target(
        search_space,
        algorithm_keys=algorithm_keys,
        portfolio_keys=portfolio_keys,
        optimization_target=optimization_target,
    )


def _infer_wide_search_space(algorithm_params: dict[str, Any], portfolio_params: dict[str, Any]) -> dict[str, Any]:
    search_space: dict[str, Any] = {}
    for key, value in algorithm_params.items():
        if key in {"symbol", "history_length"}:
            continue
        spec = _infer_search_spec_from_value(key, value)
        if spec:
            search_space[key] = spec
    for key in ("stop_pct", "profit_pct"):
        search_space[key] = _default_portfolio_search_space(key, float(portfolio_params.get(key, 0.0) or 0.0))
    return search_space


def _infer_search_spec_from_value(key: str, value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        low = max(1, int(value * 0.5))
        high = max(low + 2, int(value * 2.0) + 1)
        return {"type": "randint", "low": low, "high": high}
    if isinstance(value, float):
        low = max(0.0001, float(value) * 0.25 if value > 0 else 0.0001)
        high = max(low * 1.5, float(value) * 4.0 if value > 0 else 5.0)
        return {"type": "uniform", "low": round(low, 6), "high": round(high, 6)}
    return None


def _default_portfolio_search_space(key: str, value: float) -> dict[str, Any]:
    if key == "stop_pct":
        low = min(max(value * 0.4, 0.25), 4.0)
        high = max(value * 2.5, 12.0)
        return {"type": "uniform", "low": round(low, 4), "high": round(high, 4)}
    if key == "profit_pct":
        low = min(max(value * 0.5, 0.5), 8.0)
        high = max(value * 3.0, 20.0)
        return {"type": "uniform", "low": round(low, 4), "high": round(high, 4)}
    return {"type": "uniform", "low": 0.0, "high": max(value * 2.0, 1.0)}


def _filter_search_space_for_target(
    search_space: dict[str, Any],
    *,
    algorithm_keys: list[str],
    portfolio_keys: list[str],
    optimization_target: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    if optimization_target == "algorithm":
        allowed_keys = list(dict.fromkeys(algorithm_keys))
        return (
            {key: value for key, value in search_space.items() if key in allowed_keys},
            allowed_keys,
            [],
        )
    if optimization_target == "portfolio":
        allowed_keys = list(dict.fromkeys(portfolio_keys))
        return (
            {key: value for key, value in search_space.items() if key in allowed_keys},
            [],
            allowed_keys,
        )
    return (
        {key: value for key, value in search_space.items() if key in set(dict.fromkeys(algorithm_keys + portfolio_keys))},
        list(dict.fromkeys(algorithm_keys)),
        list(dict.fromkeys(portfolio_keys)),
    )


def _filter_hpo_config(
    hpo_cfg: dict[str, Any],
    *,
    optimization_target: str,
    algorithm_params: dict[str, Any],
    portfolio_params: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(hpo_cfg)
    search_space = _normalize_search_space(dict(cfg.get("search_space") or {}))
    algorithm_keys = [str(item) for item in (cfg.get("algorithm_param_keys") or []) if str(item) in algorithm_params]
    portfolio_keys = [str(item) for item in (cfg.get("portfolio_param_keys") or []) if str(item) in portfolio_params]
    filtered_space, filtered_algorithm_keys, filtered_portfolio_keys = _filter_search_space_for_target(
        search_space,
        algorithm_keys=algorithm_keys,
        portfolio_keys=portfolio_keys,
        optimization_target=optimization_target,
    )
    cfg["search_space"] = filtered_space
    cfg["algorithm_param_keys"] = filtered_algorithm_keys
    cfg["portfolio_param_keys"] = filtered_portfolio_keys
    cfg["optimization_target"] = optimization_target
    return cfg


def _normalize_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, spec in search_space.items():
        normalized_spec = _normalize_search_spec(str(key), spec)
        if normalized_spec:
            normalized[str(key)] = normalized_spec
    return normalized


def _normalize_search_spec(key: str, spec: Any) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    spec_type = str(spec.get("type") or "").strip().lower()
    if spec_type != "choice":
        return dict(spec)

    values = list(spec.get("values") or [])
    if not values:
        return None
    if any(isinstance(item, bool) for item in values):
        return None

    numeric_values = [item for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if len(numeric_values) != len(values):
        return None

    low = min(numeric_values)
    high = max(numeric_values)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        return {"type": "randint", "low": int(low), "high": int(high) + 1}

    if float(low) == float(high):
        base = float(low)
        low = max(0.0001, base * 0.8 if base > 0 else 0.0001)
        high = max(low * 1.25, base * 1.2 if base > 0 else 1.0)
    return {"type": "uniform", "low": round(float(low), 6), "high": round(float(high), 6)}


def _build_data_provider_params(
    *,
    profile: dict[str, Any],
    proposal: dict[str, Any],
    defaults: dict[str, Any],
    symbol: str,
    provider_implementation: str,
) -> dict[str, Any]:
    if "alpaca" in provider_implementation.lower():
        return _build_alpaca_data_provider_params(
            profile=profile,
            proposal=proposal,
            defaults=defaults,
            symbol=symbol,
        )

    data_path = str(proposal.get("data_path") or proposal.get("dataset_path") or defaults.get("data_path") or "data/SPY_5min.csv")
    data_provider = {
        "provider": provider_implementation,
        "path": data_path,
        "truncate": int(proposal.get("truncate") or 0),
    }
    if proposal.get("start_date"):
        data_provider["start_date"] = proposal["start_date"]
    if proposal.get("end_date"):
        data_provider["end_date"] = proposal["end_date"]
    return data_provider


def _build_alpaca_data_provider_params(
    *,
    profile: dict[str, Any],
    proposal: dict[str, Any],
    defaults: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    symbols = proposal.get("symbols") or proposal.get("universe") or defaults.get("symbols") or [symbol]
    if isinstance(symbols, str):
        symbols = [symbols]
    elif isinstance(symbols, (tuple, set)):
        symbols = list(symbols)
    elif not isinstance(symbols, list):
        symbols = [symbol]

    creds = _resolve_alpaca_credentials(profile=profile, proposal=proposal, defaults=defaults)
    params: dict[str, Any] = {
        "provider": "alpaca",
        "api_key": creds.get("api_key", ""),
        "secret_key": creds.get("secret_key", ""),
        "symbols": [str(item) for item in symbols if item],
        "timeframe": str(proposal.get("timeframe") or defaults.get("timeframe") or "Minute"),
        "adjustment": str(proposal.get("adjustment") or defaults.get("adjustment") or "split"),
        "market_hours_only": bool(proposal.get("market_hours_only", defaults.get("market_hours_only", True))),
    }
    if proposal.get("start_date"):
        params["start_date"] = proposal["start_date"]
    if proposal.get("end_date"):
        params["end_date"] = proposal["end_date"]
    limit = proposal.get("limit")
    if limit is None:
        limit = defaults.get("limit")
    if limit is not None:
        params["limit"] = int(limit)
    return params


def _resolve_alpaca_credentials(
    *,
    profile: dict[str, Any],
    proposal: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, str]:
    explicit_key = str(proposal.get("api_key") or defaults.get("api_key") or os.getenv("ALPACA_API_KEY") or "").strip()
    explicit_secret = str(proposal.get("secret_key") or defaults.get("secret_key") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
    if explicit_key and explicit_secret:
        return {"api_key": explicit_key, "secret_key": explicit_secret}

    account_name = str(
        proposal.get("alpaca_account")
        or defaults.get("alpaca_account")
        or ""
    ).strip()
    if not account_name:
        return {"api_key": explicit_key, "secret_key": explicit_secret}

    accounts_path = proposal.get("alpaca_account_path") or defaults.get("alpaca_account_path")
    if accounts_path:
        path = Path(str(accounts_path)).expanduser()
        if not path.is_absolute():
            path = _platform_root(profile) / path
    else:
        path = _platform_root(profile) / "accounts.yaml"

    accounts = _load_yaml_file(path)
    entry = accounts.get(account_name) if isinstance(accounts, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"Alpaca account '{account_name}' not found in {path}")

    api_key = str(entry.get("api_key") or explicit_key or "").strip()
    secret_key = str(entry.get("secret_key") or explicit_secret or "").strip()
    if not api_key or not secret_key:
        raise ValueError(f"Alpaca account '{account_name}' in {path} is missing api_key or secret_key")
    return {"api_key": api_key, "secret_key": secret_key}


def _load_yaml_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required YAML file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _variant_specs(profile: dict[str, Any], proposal: dict[str, Any]) -> list[dict[str, Any]]:
    variants = list(proposal.get("experiment_variants") or [])
    if not variants:
        return [{"name": "base", "overrides": {}}]
    max_variants = int((((profile.get("execution") or {}).get("mass_test") or {}).get("max_variants_per_proposal")) or 6)
    normalized: list[dict[str, Any]] = [{"name": "base", "overrides": {}}]
    for idx, variant in enumerate(variants[:max_variants - 1]):
        if not isinstance(variant, dict):
            continue
        normalized.append({
            "name": str(variant.get("name") or f"variant_{idx + 1}"),
            "overrides": dict(variant.get("overrides") or variant),
        })
    return normalized


def _load_trading_runtime() -> dict[str, Any]:
    modules = {
        "BacktestingEngine": ("trading.engines.backtest_engine", "BacktestingEngine"),
        "WalkForwardEngine": ("trading.engines.walk_forward_engine", "WalkForwardEngine"),
        "TickAggregationPassthroughEngine": ("trading.engines.tick_aggregation_passthrough_engine", "TickAggregationPassthroughEngine"),
        "AnalysisEngine": ("trading.analysis.analysis_engine", "AnalysisEngine"),
    }
    runtime: dict[str, Any] = {}
    for name, (module_name, attr) in modules.items():
        module = importlib.import_module(module_name)
        runtime[name] = getattr(module, attr)
    return runtime


def _import_dotted(dotted_path: str):
    module_path, attr = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _load_algorithm_class(script_path: str, class_name: str):
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Generated algorithm script not found: {script_path}")
    module_name = f"generated_trading_{path.stem}_{uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algorithm_cls = getattr(module, class_name, None)
    if algorithm_cls is None:
        raise AttributeError(f"{class_name} not found in {script_path}")
    return algorithm_cls


def _build_runtime_components(trading: dict[str, Any], runtime_cfg: dict[str, Any], algorithm_cls: type[Any]) -> dict[str, Any]:
    data_provider_spec = runtime_cfg.get("data_provider") or {}
    portfolio_spec = runtime_cfg.get("portfolio") or {}
    order_manager_spec = runtime_cfg.get("order_manager") or {}
    algorithm_spec = runtime_cfg.get("algorithm") or {}
    order_manager_cls = _import_dotted(str(order_manager_spec.get("implementation")))
    portfolio_cls = _import_dotted(str(portfolio_spec.get("implementation")))
    data_provider_cls = _import_dotted(str(data_provider_spec.get("implementation")))

    order_manager = order_manager_cls(dict(order_manager_spec.get("params") or {}))
    portfolio = portfolio_cls(dict(portfolio_spec.get("params") or {}), order_manager)
    data_provider = data_provider_cls(dict(data_provider_spec.get("params") or {}))
    algorithm_params = dict(algorithm_spec.get("params") or {})
    history_length = int(algorithm_params.pop("history_length", 0) or 0)
    try:
        algorithm = algorithm_cls(cfg=algorithm_params, history_length=history_length)
    except TypeError:
        algorithm = algorithm_cls(algorithm_params, history_length)
    return {
        "data_provider": data_provider,
        "portfolio": portfolio,
        "order_manager": order_manager,
        "algorithm": algorithm,
    }


def _run_hpo_backtest(runtime_cfg: dict[str, Any], algorithm_cls: type[Any]) -> dict[str, Any]:
    launcher = importlib.import_module("trading.launchers.run_backtest_ray")
    order_manager_cls = _import_dotted(str((runtime_cfg.get("order_manager") or {}).get("implementation")))
    data_provider_cls = _import_dotted(str((runtime_cfg.get("data_provider") or {}).get("implementation")))
    portfolio_cls = _import_dotted("trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio")

    algorithm_params = dict(((runtime_cfg.get("algorithm") or {}).get("params")) or {})
    portfolio_params = dict(((runtime_cfg.get("portfolio") or {}).get("params")) or {})
    data_provider_params = dict(((runtime_cfg.get("data_provider") or {}).get("params")) or {})
    hpo_cfg = dict(runtime_cfg.get("hpo") or {})
    analysis_cfg = dict(runtime_cfg.get("analysis") or {})

    starting_cash = float(portfolio_params.get("cash") or 100000.0)
    base_pf_cfg = {key: value for key, value in portfolio_params.items() if key not in {"cash", "keep_history"}}
    base_backtest_cfg = {
        "starting_cash": starting_cash,
        "experiment_name": analysis_cfg.get("experiment_name", "trading_agent_experiments"),
        "run_name": analysis_cfg.get("run_name", "Backtest_HPO"),
        "description": analysis_cfg.get("description", ""),
        "symbol": base_pf_cfg.get("symbol", algorithm_params.get("symbol", "SPY")),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
    }

    best_config = launcher.tune_backtest_hyperparameters(
        symbol=base_backtest_cfg["symbol"],
        algorithm_class=algorithm_cls,
        portfolio_class=portfolio_cls,
        data_provider_class=data_provider_cls,
        order_manager_class=order_manager_cls,
        base_algorithm_config=dict(algorithm_params),
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=data_provider_params,
        base_backtest_config=base_backtest_cfg,
        search_space=_parse_search_space_config(hpo_cfg.get("search_space") or {}),
        algorithm_param_keys=list(hpo_cfg.get("algorithm_param_keys") or []),
        portfolio_param_keys=list(hpo_cfg.get("portfolio_param_keys") or []),
        num_samples=int(hpo_cfg.get("num_samples") or 50),
        max_concurrent_trials=int(hpo_cfg.get("max_concurrent_trials") or 8),
    )

    best_algorithm_cfg = dict(algorithm_params)
    for key in hpo_cfg.get("algorithm_param_keys") or []:
        if key in best_config:
            best_algorithm_cfg[key] = best_config[key]

    best_portfolio_cfg = dict(base_pf_cfg)
    for key in hpo_cfg.get("portfolio_param_keys") or []:
        if key in best_config:
            best_portfolio_cfg[key] = best_config[key]

    result = launcher.run_backtest_local(
        backtest_cfg=base_backtest_cfg,
        alg_cfg=best_algorithm_cfg,
        pf_cfg=best_portfolio_cfg,
        dp_cfg=data_provider_params,
        algorithm_class=algorithm_cls,
        portfolio_class=portfolio_cls,
        data_provider_class=data_provider_cls,
        order_manager_class=order_manager_cls,
    )
    metrics_obj = result["metrics"]
    metrics = asdict(metrics_obj) if is_dataclass(metrics_obj) else dict(metrics_obj)
    report = result.get("report") or ""
    raw_output = {
        "best_config": dict(best_config),
        "best_algorithm_config": best_algorithm_cfg,
        "best_portfolio_config": best_portfolio_cfg,
        "hpo_config": hpo_cfg,
        "result_summary": {
            "trade_count": len(result.get("trades") or []),
        },
    }
    return {
        "metrics": metrics,
        "report": report,
        "raw_output": raw_output,
    }


def _parse_search_space_config(search_space: dict[str, Any]) -> dict[str, Any]:
    module = importlib.import_module("utils.utils")
    return module.parse_search_space(search_space)


def _walk_forward_metrics(period_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    metric_rows = [asdict(item["metrics"]) for item in period_results if item.get("metrics") is not None and is_dataclass(item.get("metrics"))]
    if not metric_rows:
        return {
            "sharpe_ratio": float(aggregate.get("mean_sharpe_ratio", 0.0) or 0.0),
            "win_rate": float(aggregate.get("mean_win_rate", 0.0) or 0.0),
            "total_return_pct": float(aggregate.get("mean_return_pct", 0.0) or 0.0),
            "total_trades": float(aggregate.get("total_trades", 0.0) or 0.0),
            "num_periods": float(aggregate.get("num_periods", 0.0) or 0.0),
        }
    return {
        "sharpe_ratio": _safe_mean(row.get("sharpe_ratio", 0.0) for row in metric_rows),
        "sortino_ratio": _safe_mean(row.get("sortino_ratio", 0.0) for row in metric_rows),
        "max_drawdown_pct": min(float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in metric_rows),
        "annualized_return": _safe_mean(row.get("annualized_return", 0.0) for row in metric_rows),
        "win_rate": _safe_mean(row.get("win_rate", 0.0) for row in metric_rows),
        "profit_factor": _safe_mean(row.get("profit_factor", 0.0) for row in metric_rows),
        "total_trades": sum(float(row.get("total_trades", 0.0) or 0.0) for row in metric_rows),
        "num_periods": float(aggregate.get("num_periods", len(metric_rows)) or len(metric_rows)),
        "mean_return_pct": float(aggregate.get("mean_return_pct", 0.0) or 0.0),
        "periods_adopted_new_params": float(aggregate.get("periods_adopted_new_params", 0.0) or 0.0),
    }


def _summarize_variant_results(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not variant_results:
        return {"metrics": {}, "report": ""}
    if len(variant_results) == 1:
        single = variant_results[0]
        return {"metrics": dict(single.get("metrics") or {}), "report": str(single.get("report") or "")}

    metric_names = sorted({
        key
        for item in variant_results
        for key, value in (item.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    summary_metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [float((item.get("metrics") or {}).get(name, 0.0) or 0.0) for item in variant_results]
        if name == "max_drawdown_pct":
            summary_metrics[name] = min(values)
        elif name == "total_trades":
            summary_metrics[name] = sum(values)
        else:
            summary_metrics[name] = _safe_mean(values)
    sharpe_values = [float((item.get("metrics") or {}).get("sharpe_ratio", 0.0) or 0.0) for item in variant_results]
    summary_metrics["variant_count"] = len(variant_results)
    summary_metrics["best_variant_sharpe_ratio"] = max(sharpe_values)
    summary_metrics["worst_variant_sharpe_ratio"] = min(sharpe_values)
    summary_metrics["sharpe_ratio_std"] = statistics.pstdev(sharpe_values) if len(sharpe_values) > 1 else 0.0
    best_variant = max(
        variant_results,
        key=lambda item: float((item.get("metrics") or {}).get("sharpe_ratio", float("-inf")) or float("-inf")),
    )
    return {
        "metrics": summary_metrics,
        "report": str(best_variant.get("report") or ""),
    }


def _safe_mean(values) -> float:
    seq = [float(value or 0.0) for value in values]
    return sum(seq) / len(seq) if seq else 0.0


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _public_runtime_config(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    return _redact_sensitive_fields(runtime_cfg)


def _redact_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"api_key", "secret_key", "password", "token"}:
                redacted[key] = "***redacted***"
            else:
                redacted[key] = _redact_sensitive_fields(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value
