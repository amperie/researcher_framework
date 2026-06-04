"""Trading research adapter backed by the local ``trading_guy`` project."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
import hashlib
import inspect
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from uuid import uuid4

import yaml

from configs.config import get_config, resolve_dev_path
from core.plugins.base import ResearchAdapter
from core.plugins.execution import run_task
from core.utils import terminal_progress
from core.utils.logger import get_logger

log = get_logger(__name__)
_PLUGIN_NAME = "trading"
_PLUGIN_LOGGER_PREFIXES = [
    "core.plugins.trading",
    "core.plugins.task_runner",
    "core.plugins.job_runner",
]
_RUN_TRADING_ARTIFACT_TASK = "core.plugins.trading.tasks.run_trading_artifact"


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
            "trading_python": _trading_python(profile),
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

    def external_runtime_spec(self, profile: dict[str, Any], purpose: str) -> dict[str, Any]:
        return {
            "python": _trading_python(profile),
            "cwd": str(_platform_root(profile)),
            "pythonpath_entries": [str(path) for path in _trading_pythonpath_entries(profile)],
            "plugin_name": _PLUGIN_NAME,
            "logger_prefixes": list(_PLUGIN_LOGGER_PREFIXES),
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
        proposals_by_name = {
            str(item.get("name") or ""): item
            for item in (state.get("proposals") or [])
            if item.get("name")
        }
        total_hpo_trials = _total_hpo_trials(artifacts)
        completed_hpo_trials = 0
        terminal_progress.configure_hpo(total_hpo_trials)
        for artifact in artifacts:
            proposal_name = str(artifact.get("proposal_name") or "unknown")
            artifact_trials = _artifact_hpo_trials(artifact)
            try:
                script_path = str(artifact.get("script_path") or "")
                script_source = Path(script_path).read_text(encoding="utf-8") if script_path and Path(script_path).exists() else ""
                result = run_task(
                    {
                        "task_path": _RUN_TRADING_ARTIFACT_TASK,
                        "payload": {
                            "profile": profile,
                            "artifact": artifact,
                            "proposal": proposals_by_name.get(proposal_name, {}),
                            "script_source": script_source,
                        },
                        "python": _trading_python(profile),
                        "timeout": int(get_config().experiment_timeout_seconds),
                        "plugin_name": _PLUGIN_NAME,
                        "logger_prefixes": list(_PLUGIN_LOGGER_PREFIXES),
                        "cwd": str(_platform_root(profile)),
                        "pythonpath_entries": [str(path) for path in _trading_pythonpath_entries(profile)],
                        "env": {
                            "RESEARCH_PROGRESS_BRIDGE": "1",
                            "RESEARCH_HPO_TRIAL_OFFSET": str(completed_hpo_trials),
                            "RESEARCH_HPO_TRIAL_TOTAL": str(total_hpo_trials),
                        },
                        "job_id": f"trading_{proposal_name}",
                        "job_dir": str(resolve_dev_path(f"dev/experiments/trading/sync_tasks/{proposal_name}")),
                    },
                    profile,
                    "execute_experiment",
                    default_runner="sync",
                )
                experiment_id = str(result.get("experiment_id") or uuid4())
                result_path = output_dir / f"{result.get('class_name', proposal_name)}_{experiment_id[:8]}.json"
                result_artifacts = dict(result.get("artifacts") or {})
                runtime_config_path = str(artifact.get("config_path") or "")
                result["artifacts"] = {
                    **result_artifacts,
                    "runtime_config_path": runtime_config_path,
                    "results_json_path": str(result_path),
                    "variant_count": len(result.get("variant_results") or []),
                    "runtime_config": _public_runtime_config(dict(artifact.get("runtime_config") or {})),
                }
                result_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
                results.append(result)
                completed_hpo_trials += artifact_trials
                terminal_progress.update_hpo(
                    done=completed_hpo_trials,
                    running=0,
                    total=total_hpo_trials,
                    message=f"{proposal_name} complete",
                )
            except Exception as exc:
                log.error("TradingAdapter.execute_experiment | %s failed: %s", proposal_name, exc, exc_info=True)
                errors.append(f"execute_experiment: {proposal_name} failed: {exc}")
                completed_hpo_trials += artifact_trials
                terminal_progress.update_hpo(
                    done=completed_hpo_trials,
                    running=0,
                    total=total_hpo_trials,
                    message=f"{proposal_name} failed",
                )

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
        config_artifact_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        trading = _load_trading_runtime()
        variant_name = str(variant_spec.get("name") or "base")
        merged_cfg = _normalize_runtime_config(_deep_merge(runtime_cfg, dict(variant_spec.get("overrides") or {})))
        algorithm_cls = _load_algorithm_class(script_path, class_name)
        mode = str(merged_cfg.get("mode") or "backtest")
        mlflow_capture = _install_mlflow_capture()

        try:
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
                _log_mlflow_config_artifacts(
                    config_artifact_paths or [],
                    capture=mlflow_capture,
                    extra_params={
                        "algorithm_implementation": str((merged_cfg.get("algorithm") or {}).get("implementation") or ""),
                        "portfolio_implementation": str((merged_cfg.get("portfolio") or {}).get("implementation") or ""),
                        "data_provider_implementation": str((merged_cfg.get("data_provider") or {}).get("implementation") or ""),
                        "order_manager_implementation": str((merged_cfg.get("order_manager") or {}).get("implementation") or ""),
                    },
                )
            else:
                hpo_result = _run_hpo_backtest(
                    merged_cfg,
                    algorithm_cls,
                    config_artifact_paths=config_artifact_paths or [],
                    mlflow_capture=mlflow_capture,
                )
                metrics = dict(hpo_result["metrics"])
                report = str(hpo_result["report"])
                raw_output = dict(hpo_result["raw_output"])
        finally:
            _restore_mlflow_capture(mlflow_capture)

        return {
            "variant_name": variant_name,
            "mode": mode,
            "metrics": metrics,
            "config": merged_cfg,
            "report": report,
            "raw_output": raw_output,
            "mlflow": {
                "run_id": mlflow_capture.get("run_id", ""),
                "run_url": mlflow_capture.get("run_url", ""),
                "tracking_uri": mlflow_capture.get("tracking_uri", ""),
                "experiment_name": mlflow_capture.get("experiment_name", ""),
                "experiment_id": mlflow_capture.get("experiment_id", ""),
            },
        }


def get_adapter() -> TradingAdapter:
    return TradingAdapter()


def _artifact_hpo_trials(artifact: dict[str, Any]) -> int:
    runtime_cfg = dict(artifact.get("runtime_config") or {})
    variant_specs = list(artifact.get("variant_specs") or [{"name": "base", "overrides": {}}])
    total = 0
    for variant_spec in variant_specs:
        merged_cfg = _normalize_runtime_config(_deep_merge(runtime_cfg, dict(variant_spec.get("overrides") or {})))
        if str(merged_cfg.get("mode") or "backtest") == "walk-forward":
            continue
        total += int((merged_cfg.get("hpo") or {}).get("num_samples") or 50)
    return total


def _total_hpo_trials(artifacts: list[dict[str, Any]]) -> int:
    return sum(_artifact_hpo_trials(dict(artifact or {})) for artifact in artifacts)


def _platform_root(profile: dict[str, Any]) -> Path:
    source_path = str(((profile.get("platform") or {}).get("source_path")) or "../trading_guy")
    return Path(source_path).expanduser().resolve()


def _trading_python(profile: dict[str, Any]) -> str:
    execution_cfg = profile.get("execution") or {}
    configured = execution_cfg.get("python")
    if configured:
        return str(configured)
    platform_root = _platform_root(profile)
    candidates = [
        platform_root / ".venv" / "Scripts" / "python.exe",
        platform_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(get_config().trading_python)


def _trading_pythonpath_entries(profile: dict[str, Any]) -> list[Path]:
    platform_root = _platform_root(profile)
    researcher_root = _researcher_root()
    entries = [platform_root, researcher_root]
    unique: list[Path] = []
    seen: set[str] = set()
    for entry in entries:
        key = str(entry.resolve())
        if key not in seen:
            unique.append(entry.resolve())
            seen.add(key)
    return unique


def _researcher_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_generated_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_researcher_root() / path).resolve()


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
    symbol_ctx = _resolve_symbol_context(proposal, implementation, defaults)
    symbol = symbol_ctx["primary_symbol"]
    run_name = f"{proposal.get('name', implementation.get('class_name', 'strategy'))}_{mode}"
    algorithm_params = {
        "symbol": symbol,
        "symbols": list(symbol_ctx["expected_symbols"]),
        "tradable_symbols": list(symbol_ctx["tradable_symbols"]),
        "macro_symbols": list(symbol_ctx["macro_symbols"]),
        **dict(proposal.get("hyperparameters") or {}),
    }
    algorithm_params["history_length"] = _effective_history_length(
        algorithm_params,
        fallback=int(proposal.get("history_length") or defaults.get("history_length") or 200),
    )
    portfolio_impl = _portfolio_implementation_for_mode(profile, proposal, mode, symbol_ctx)
    portfolio_params = _build_portfolio_params(
        proposal=proposal,
        defaults=defaults,
        primary_symbol=symbol,
        portfolio_implementation=portfolio_impl,
        symbol_ctx=symbol_ctx,
    )
    data_provider_impl = str(proposal.get("data_provider") or defaults.get("data_provider") or _default_provider(profile))
    data_provider = _build_data_provider_params(
        profile=profile,
        proposal=proposal,
        defaults=defaults,
        symbol_ctx=symbol_ctx,
        algorithm_params=algorithm_params,
        provider_implementation=data_provider_impl,
    )

    runtime_cfg: dict[str, Any] = {
        "mode": mode,
        "algorithm": {
            "implementation": "__generated__",
            "params": algorithm_params,
        },
        "portfolio": {
            "implementation": portfolio_impl,
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
            "mlflow_policy": {
                "annualized_return_threshold": float(defaults.get("mlflow_annualized_return_threshold", 0.0)),
                "sample_negative_rate": int(defaults.get("mlflow_sample_negative_rate", 20) or 20),
            },
        },
        "aggregation": {
            "enabled": bool(proposal.get("aggregation_enabled", defaults.get("aggregation_enabled", False))),
            "aggregation_period_minutes": int(proposal.get("aggregation_period_minutes") or defaults.get("aggregation_period_minutes") or 1),
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
    return _normalize_runtime_config(_ensure_runtime_data_sufficiency(runtime_cfg))


def _portfolio_implementation_for_mode(
    profile: dict[str, Any],
    proposal: dict[str, Any],
    mode: str,
    symbol_ctx: dict[str, Any],
) -> str:
    explicit = str(proposal.get("portfolio") or "").strip()
    if explicit:
        return explicit
    if mode == "backtest":
        if len(symbol_ctx.get("tradable_symbols") or []) >= 2:
            return "trading.core.pf.dual_symbol_switch_portfolio.DualSymbolSwitchPortfolio"
        return "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    return _default_portfolio(profile)


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
        or base.get("objective_metric")
        or "annualized_return"
    )
    hpo_cfg["num_samples"] = int(proposal_hpo.get("num_samples") or proposal_hpo.get("n_trials") or base.get("num_samples") or 50)
    hpo_cfg["max_concurrent_trials"] = int(proposal_hpo.get("max_concurrent_trials") or base.get("max_concurrent_trials") or 8)
    hpo_cfg["smoke_trials"] = int(proposal_hpo.get("smoke_trials") or base.get("smoke_trials") or 3)
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
    proposal_hpo = proposal.get("hpo") if isinstance(proposal.get("hpo"), dict) else {}
    search_space = _normalize_search_space(proposal_hpo.get("search_space") or {})
    algorithm_keys = [str(item) for item in (proposal_hpo.get("algorithm_param_keys") or []) if item]
    portfolio_keys = [str(item) for item in (proposal_hpo.get("portfolio_param_keys") or []) if item]

    tunable_params = proposal_hpo.get("tunable_params") or {}
    if not search_space:
        search_space = _normalize_search_space(tunable_params)

    if not search_space:
        search_space = _infer_wide_search_space(algorithm_params, portfolio_params)
    else:
        search_space = _expand_backtest_search_space(
            search_space,
            algorithm_params=algorithm_params,
            portfolio_params=portfolio_params,
        )

    if not algorithm_keys:
        algorithm_keys = [key for key in search_space if key in algorithm_params]
    if not portfolio_keys:
        portfolio_keys = [key for key in search_space if key in portfolio_params]

    algorithm_keys, portfolio_keys = _rebalance_hpo_keys(
        algorithm_keys=algorithm_keys,
        portfolio_keys=portfolio_keys,
        search_space=search_space,
        algorithm_params=algorithm_params,
        portfolio_params=portfolio_params,
    )

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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _semantic_search_spec(key, value)
    return None


def _default_portfolio_search_space(key: str, value: float) -> dict[str, Any]:
    if key == "stop_pct":
        return {"type": "uniform", "low": 0.5, "high": 20.0}
    if key == "profit_pct":
        return {"type": "uniform", "low": 0.5, "high": 20.0}
    return {"type": "uniform", "low": 0.0, "high": max(value * 2.0, 1.0)}


def _semantic_search_spec(key: str, value: Any = None, spec: dict[str, Any] | None = None) -> dict[str, Any] | None:
    name = str(key or "").lower()
    raw = dict(spec or {})
    if raw.get("type") == "choice":
        return _normalize_search_spec(key, raw)
    if _is_percentile_key(name):
        low, high = _bounded_float_bounds(raw, 1.0, 99.0)
        return {"type": "uniform", "low": low, "high": high}
    if "quantile" in name:
        low, high = _bounded_float_bounds(raw, 0.01, 0.99)
        return {"type": "uniform", "low": low, "high": high}
    if "correlation" in name:
        low, high = _bounded_float_bounds(raw, 0.0, 1.0)
        return {"type": "uniform", "low": low, "high": high}
    if "zscore" in name or "z_score" in name:
        low, high = _signed_float_bounds(value, raw, default=(-4.0, 4.0))
        return {"type": "uniform", "low": low, "high": high}
    if "ratio" in name:
        low, high = _bounded_float_bounds(raw, 0.1, 5.0)
        return {"type": "uniform", "low": low, "high": high}
    if any(token in name for token in ("fraction", "position_size", "risk_fraction", "base_position")):
        low, high = _bounded_float_bounds(raw, 0.001, 1.0)
        return {"type": "uniform", "low": low, "high": high}
    if _is_window_key(name):
        low, high = _integer_window_bounds(name, value, raw)
        return {"type": "randint", "low": low, "high": high}
    if isinstance(value, int) and not isinstance(value, bool):
        base = max(1, abs(int(value)))
        default_low = max(1, base // 3)
        default_high = min(501, max(base * 3, base + 5) + 1)
        low, high = _bounded_int_bounds(raw, default_low, default_high, 1, 501)
        return {"type": "randint", "low": low, "high": high}
    if isinstance(value, float) or raw:
        low, high = _positive_float_bounds(value, raw)
        return {"type": "uniform", "low": low, "high": high}
    return None


def _is_percentile_key(name: str) -> bool:
    return (
        "percentile" in name
        or name.endswith("_pct")
        or name.endswith("_percent")
    ) and name not in {"stop_pct", "profit_pct", "tx_cost"}


def _is_window_key(name: str) -> bool:
    return any(token in name for token in (
        "window", "lookback", "period", "bars", "hold", "lock", "confirmation",
        "forward", "rips", "ma_", "_ma", "sma", "ema",
    ))


def _integer_window_bounds(name: str, value: Any, raw: dict[str, Any]) -> tuple[int, int]:
    if "rips" in name:
        return _bounded_int_bounds(raw, 5, 121, 5, 121)
    if "slow" in name and ("ma" in name or "sma" in name or "ema" in name):
        return _bounded_int_bounds(raw, 10, 501, 10, 501)
    if "fast" in name and ("ma" in name or "sma" in name or "ema" in name):
        return _bounded_int_bounds(raw, 2, 101, 2, 101)
    if any(token in name for token in ("confirmation", "lock", "hold", "forward")):
        return _bounded_int_bounds(raw, 1, 201, 1, 201)
    if "atr" in name or "rsi" in name:
        return _bounded_int_bounds(raw, 2, 101, 2, 101)
    base = _numeric_seed(value, raw, default=20.0)
    low = max(2, int(base / 3))
    high = min(501, max(low + 5, int(base * 3) + 1))
    return _bounded_int_bounds(raw, low, high, 2, 501)


def _bounded_int_bounds(
    raw: dict[str, Any],
    default_low: int,
    default_high: int,
    min_low: int,
    max_high: int,
) -> tuple[int, int]:
    if raw.get("low") is not None and raw.get("high") is not None:
        low = max(min_low, int(raw["low"]))
        high = min(max_high, int(raw["high"]))
        if low < high:
            return low, high
    return default_low, default_high


def _bounded_float_bounds(raw: dict[str, Any], default_low: float, default_high: float) -> tuple[float, float]:
    if raw.get("low") is not None and raw.get("high") is not None:
        low = max(default_low, float(raw["low"]))
        high = min(default_high, float(raw["high"]))
        if low < high:
            return round(low, 6), round(high, 6)
    return default_low, default_high


def _signed_float_bounds(value: Any, raw: dict[str, Any], *, default: tuple[float, float]) -> tuple[float, float]:
    if raw.get("low") is not None and raw.get("high") is not None:
        low = max(default[0], float(raw["low"]))
        high = min(default[1], float(raw["high"]))
        if low < high:
            return round(low, 6), round(high, 6)
    seed = _numeric_seed(value, raw, default=0.0)
    if seed < 0:
        return default[0], -0.25
    if seed > 0:
        return 0.25, default[1]
    return default


def _positive_float_bounds(value: Any, raw: dict[str, Any]) -> tuple[float, float]:
    if raw.get("low") is not None and raw.get("high") is not None:
        low = max(0.0001, float(raw["low"]))
        high = min(100.0, float(raw["high"]))
        if low < high:
            return round(low, 6), round(high, 6)
    seed = max(0.0001, _numeric_seed(value, raw, default=1.0))
    return round(max(0.0001, seed / 3), 6), round(min(100.0, max(seed * 3, seed + 0.1)), 6)


def _numeric_seed(value: Any, raw: dict[str, Any], *, default: float) -> float:
    for candidate in (value, raw.get("value"), raw.get("default")):
        try:
            if candidate is not None:
                return float(candidate)
        except Exception:
            pass
    return float(default)


def _expand_backtest_search_space(
    search_space: dict[str, Any],
    *,
    algorithm_params: dict[str, Any],
    portfolio_params: dict[str, Any],
) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, spec in search_space.items():
        if _is_hpo_excluded_key(key):
            continue
        if key in {"stop_pct", "profit_pct"}:
            expanded[key] = _default_portfolio_search_space(key, float(portfolio_params.get(key, 0.0) or 0.0))
            continue
        value = algorithm_params.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            normalized = _semantic_search_spec(key, value, spec)
            if normalized:
                expanded[key] = normalized
            continue
        if isinstance(value, float):
            normalized = _semantic_search_spec(key, value, spec)
            if normalized:
                expanded[key] = normalized
            continue
        normalized = _semantic_search_spec(key, None, spec)
        if normalized:
            expanded[key] = normalized
    return expanded


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


def _normalize_runtime_config(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = _deep_merge({}, runtime_cfg)
    hpo_cfg = dict(cfg.get("hpo") or {})
    hpo_cfg["search_space"] = _normalize_search_space(hpo_cfg.get("search_space") or {})
    cfg["hpo"] = hpo_cfg

    data_provider = dict(cfg.get("data_provider") or {})
    provider_params = dict(data_provider.get("params") or {})
    if "timeframe" in provider_params:
        provider_params["timeframe"] = _normalize_alpaca_timeframe(str(provider_params.get("timeframe") or ""))
    data_provider["params"] = provider_params
    cfg["data_provider"] = data_provider
    return cfg


def _normalize_search_space(search_space: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, spec in _search_space_entries(search_space):
        if _is_hpo_excluded_key(key):
            continue
        normalized_spec = _normalize_search_spec(str(key), spec)
        if normalized_spec is not None:
            normalized[str(key)] = normalized_spec
    return normalized


def _is_hpo_excluded_key(key: str) -> bool:
    name = str(key or "").strip().lower()
    return name in {
        "symbol",
        "symbols",
        "tradable_symbols",
        "timeframe",
        "adjustment",
        "start_date",
        "end_date",
        "data_provider",
        "portfolio",
        "order_manager",
        "api_key",
        "secret_key",
        "alpaca_account",
        "alpaca_account_path",
    }


def _search_space_entries(search_space: Any) -> list[tuple[str, Any]]:
    if isinstance(search_space, dict):
        return [(str(key), spec) for key, spec in search_space.items() if key]
    if not isinstance(search_space, list):
        return []
    entries: list[tuple[str, Any]] = []
    for item in search_space:
        if isinstance(item, str) and item.strip():
            entries.append((item.strip(), {}))
            continue
        if not isinstance(item, dict):
            continue
        key = item.get("key") or item.get("name") or item.get("param") or item.get("parameter") or item.get("parameter_name")
        if not key:
            continue
        spec = item.get("spec") or item.get("space") or item.get("search_space")
        if not isinstance(spec, dict):
            spec = {k: v for k, v in item.items() if k not in {"key", "name", "param", "parameter", "parameter_name"}}
        entries.append((str(key), spec))
    return entries


def _normalize_search_spec(key: str, spec: Any) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    if not spec:
        return {}
    spec_type = str(spec.get("type") or "").strip().lower()
    if spec_type in {"randint", "uniform", "loguniform"}:
        normalized = dict(spec)
        if "low" not in normalized:
            if "lower" in normalized:
                normalized["low"] = normalized["lower"]
            elif "min" in normalized:
                normalized["low"] = normalized["min"]
        if "high" not in normalized:
            if "upper" in normalized:
                normalized["high"] = normalized["upper"]
            elif "max" in normalized:
                normalized["high"] = normalized["max"]
        if "low" not in normalized or "high" not in normalized:
            return None
        return normalized
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
    symbol_ctx: dict[str, Any],
    algorithm_params: dict[str, Any],
    provider_implementation: str,
) -> dict[str, Any]:
    if "alpaca" in provider_implementation.lower():
        return _build_alpaca_data_provider_params(
            profile=profile,
            proposal=proposal,
            defaults=defaults,
            symbol_ctx=symbol_ctx,
            algorithm_params=algorithm_params,
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
    symbol_ctx: dict[str, Any],
    algorithm_params: dict[str, Any],
) -> dict[str, Any]:
    creds = _resolve_alpaca_credentials(profile=profile, proposal=proposal, defaults=defaults)
    params: dict[str, Any] = {
        "provider": "alpaca",
        "api_key": creds.get("api_key", ""),
        "secret_key": creds.get("secret_key", ""),
        "symbols": [str(item) for item in (symbol_ctx.get("expected_symbols") or []) if item],
        "timeframe": _normalize_alpaca_timeframe(str(proposal.get("timeframe") or defaults.get("timeframe") or "Minute")),
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
    required_limit = _minimum_alpaca_limit(params, algorithm_params)
    if required_limit > int(params.get("limit") or 0):
        params["limit"] = required_limit
    return params


def _build_portfolio_params(
    *,
    proposal: dict[str, Any],
    defaults: dict[str, Any],
    primary_symbol: str,
    portfolio_implementation: str,
    symbol_ctx: dict[str, Any],
) -> dict[str, Any]:
    params = {
        "cash": float(proposal.get("cash") or defaults.get("cash") or 100000),
        "keep_history": True,
        "stop_pct": float(proposal.get("stop_pct") or defaults.get("stop_pct") or 5.0),
        "profit_pct": float(proposal.get("profit_pct") or defaults.get("profit_pct") or 10.0),
        "tx_cost": float(proposal.get("tx_cost") or defaults.get("tx_cost") or 0.0),
    }
    if portfolio_implementation.endswith("DualSymbolSwitchPortfolio"):
        tradable_symbols = list(symbol_ctx.get("tradable_symbols") or [])
        params["upro_symbol"] = tradable_symbols[0] if len(tradable_symbols) > 0 else primary_symbol
        params["spxu_symbol"] = tradable_symbols[1] if len(tradable_symbols) > 1 else primary_symbol
        if proposal.get("holding_period_hours") is not None:
            params["holding_period_hours"] = float(proposal.get("holding_period_hours"))
        return params
    params["symbol"] = primary_symbol
    return params


def _resolve_symbol_context(
    proposal: dict[str, Any],
    implementation: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    explicit_symbols = _normalize_symbol_list(proposal.get("symbols") or proposal.get("universe") or defaults.get("symbols"))
    macro_symbols = _normalize_symbol_list(proposal.get("macro_symbols"))
    tradable_symbols = _normalize_symbol_list(
        proposal.get("tradable_symbols")
        or [proposal.get("long_symbol"), proposal.get("short_symbol")]
    )
    primary_symbol = str(proposal.get("symbol") or defaults.get("symbol") or "SPY")
    if not tradable_symbols:
        tradable_symbols = [primary_symbol]

    detected_macro = _detect_macro_symbols(proposal, implementation)
    macro_symbols = list(dict.fromkeys(macro_symbols + detected_macro))
    expected_symbols = list(dict.fromkeys(explicit_symbols + tradable_symbols + macro_symbols))
    if not expected_symbols:
        expected_symbols = [primary_symbol]
    return {
        "primary_symbol": primary_symbol,
        "tradable_symbols": tradable_symbols,
        "macro_symbols": macro_symbols,
        "expected_symbols": expected_symbols,
    }


def _normalize_symbol_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (tuple, set)):
        value = list(value)
    if not isinstance(value, list):
        return [str(value)] if value else []
    return [str(item) for item in value if item]


def _detect_macro_symbols(proposal: dict[str, Any], implementation: dict[str, Any]) -> list[str]:
    text_parts = [
        str(proposal.get("name") or ""),
        str(proposal.get("description") or ""),
        json.dumps(proposal.get("hyperparameters") or {}, sort_keys=True),
        json.dumps(proposal.get("hpo") or {}, sort_keys=True),
    ]
    script_path = str(implementation.get("script_path") or "")
    if script_path and Path(script_path).exists():
        try:
            text_parts.append(Path(script_path).read_text(encoding="utf-8"))
        except Exception:
            pass
    haystack = "\n".join(text_parts).lower()
    macro_symbols: list[str] = []
    if "vix" in haystack:
        macro_symbols.append("VIX")
    if "t10y2y" in haystack or "yield_curve" in haystack or "yield curve" in haystack:
        macro_symbols.append("T10Y2Y")
    return macro_symbols


def _effective_history_length(algorithm_params: dict[str, Any], fallback: int) -> int:
    blacklist = {
        "cash",
        "limit",
        "position_size_base",
        "tx_cost",
        "stop_pct",
        "profit_pct",
        "stop_loss_bps_normal",
        "stop_loss_bps_high_vix",
        "hold_time_minutes",
        "holding_period_hours",
    }
    candidates = [max(1, int(fallback or 0))]
    for key, value in algorithm_params.items():
        if key in blacklist:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        text = str(key).lower()
        if any(token in text for token in ("period", "window", "lookback", "history", "warmup", "bars", "length")):
            candidates.append(int(value))
    slow = algorithm_params.get("macd_slow_period")
    signal = algorithm_params.get("macd_signal_period")
    if isinstance(slow, int) and isinstance(signal, int):
        candidates.append(int(slow) + int(signal))
    return max(candidates) + 5


def _minimum_alpaca_limit(data_provider_params: dict[str, Any], algorithm_params: dict[str, Any]) -> int:
    history_length = max(0, int(algorithm_params.get("history_length") or 0))
    date_range_limit = _date_range_bar_count(data_provider_params)
    return max(history_length, date_range_limit, int(data_provider_params.get("limit") or 0))


def _date_range_bar_count(data_provider_params: dict[str, Any]) -> int:
    start_raw = data_provider_params.get("start_date")
    end_raw = data_provider_params.get("end_date")
    if not start_raw or not end_raw:
        return 0
    try:
        start_day = date.fromisoformat(str(start_raw)[:10])
        end_day = date.fromisoformat(str(end_raw)[:10])
    except Exception:
        return 0
    if end_day < start_day:
        return 0
    trading_days = 0
    current = start_day
    while current <= end_day:
        if current.weekday() < 5:
            trading_days += 1
        current += timedelta(days=1)
    return trading_days * _bars_per_trading_day(data_provider_params)


def _rebalance_hpo_keys(
    *,
    algorithm_keys: list[str],
    portfolio_keys: list[str],
    search_space: dict[str, Any],
    algorithm_params: dict[str, Any],
    portfolio_params: dict[str, Any],
) -> tuple[list[str], list[str]]:
    algorithm_set = list(dict.fromkeys(algorithm_keys))
    portfolio_set = list(dict.fromkeys(portfolio_keys))
    for key in search_space:
        if key in algorithm_params and key not in algorithm_set:
            algorithm_set.append(key)
        if key in portfolio_params and key not in portfolio_set:
            portfolio_set.append(key)
    portfolio_only_names = {"stop_pct", "profit_pct", "tx_cost", "cash", "holding_period_hours", "min_signal_strength", "symbol", "upro_symbol", "spxu_symbol"}
    for key in list(portfolio_set):
        if key in algorithm_params and key not in portfolio_params:
            portfolio_set.remove(key)
            if key not in algorithm_set:
                algorithm_set.append(key)
    for key in list(search_space):
        if key in portfolio_only_names and key not in portfolio_set:
            portfolio_set.append(key)
            if key in algorithm_set and key not in algorithm_params:
                algorithm_set.remove(key)
    algorithm_set = [key for key in algorithm_set if key not in portfolio_set or key in algorithm_params]
    portfolio_set = [key for key in portfolio_set if key in portfolio_params or key in portfolio_only_names]
    return algorithm_set, portfolio_set


def _ensure_runtime_data_sufficiency(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = _deep_merge({}, runtime_cfg)
    provider_impl = str(((cfg.get("data_provider") or {}).get("implementation")) or "")
    if "alpaca" not in provider_impl.lower():
        return cfg

    algorithm_params = dict(((cfg.get("algorithm") or {}).get("params")) or {})
    effective_history_length = _effective_history_length(
        algorithm_params,
        fallback=int(algorithm_params.get("history_length") or 0),
    )
    algorithm_params["history_length"] = effective_history_length
    cfg.setdefault("algorithm", {})["params"] = algorithm_params

    data_provider = dict(cfg.get("data_provider") or {})
    params = dict(data_provider.get("params") or {})
    required_limit = _minimum_alpaca_limit(params, algorithm_params)
    current_limit = int(params.get("limit") or 0)
    if required_limit > current_limit:
        log.info(
            "Trading runtime | increasing Alpaca limit from %s to %s",
            current_limit or 0,
            required_limit,
        )
        params["limit"] = required_limit
    data_provider["params"] = params
    cfg["data_provider"] = data_provider

    mode = str(cfg.get("mode") or "backtest").strip().lower()
    if mode != "walk-forward":
        return cfg

    required_limit = _minimum_walk_forward_limit(cfg)
    if required_limit > current_limit:
        log.info(
            "Trading runtime | increasing Alpaca limit for walk-forward from %s to %s",
            current_limit or 0,
            required_limit,
        )
        params["limit"] = required_limit
    data_provider["params"] = params
    cfg["data_provider"] = data_provider
    return cfg


def _minimum_walk_forward_limit(runtime_cfg: dict[str, Any]) -> int:
    algorithm_params = dict(((runtime_cfg.get("algorithm") or {}).get("params")) or {})
    walk_forward = dict(runtime_cfg.get("walk_forward") or {})
    data_provider = dict(((runtime_cfg.get("data_provider") or {}).get("params")) or {})

    history_length = max(0, int(algorithm_params.get("history_length") or 0))
    optimization_days = max(1, int(walk_forward.get("optimization_window_days") or 0))
    trading_days = max(1, int(walk_forward.get("trading_window_days") or 0))
    warmup_days = max(5, history_length // max(1, _bars_per_trading_day(data_provider)))
    buffer_days = max(5, int(walk_forward.get("step_days") or 0))
    total_days = optimization_days + trading_days + warmup_days + buffer_days
    return history_length + (total_days * _bars_per_trading_day(data_provider))


def _bars_per_trading_day(data_provider_params: dict[str, Any]) -> int:
    timeframe = _normalize_alpaca_timeframe(str(data_provider_params.get("timeframe") or "Minute"))
    market_hours_only = bool(data_provider_params.get("market_hours_only", True))
    if timeframe == "Minute":
        return 78 if market_hours_only else 390
    if timeframe == "Hour":
        return 7 if market_hours_only else 24
    if timeframe == "Day":
        return 1
    if timeframe == "Week":
        return 1
    if timeframe == "Month":
        return 1
    return 78 if market_hours_only else 390


def _normalize_alpaca_timeframe(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Minute"
    normalized = raw.lower().replace(" ", "")
    alias_map = {
        "1": "Minute",
        "1m": "Minute",
        "1min": "Minute",
        "min": "Minute",
        "minute": "Minute",
        "5m": "Minute",
        "5min": "Minute",
        "15m": "Minute",
        "15min": "Minute",
        "30m": "Minute",
        "30min": "Minute",
        "60m": "Hour",
        "60min": "Hour",
        "1h": "Hour",
        "hour": "Hour",
        "1d": "Day",
        "day": "Day",
        "1w": "Week",
        "week": "Week",
        "1mo": "Month",
        "month": "Month",
    }
    return alias_map.get(normalized, raw)


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
    _patch_trading_color_logger_pickling()
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
    _patch_trading_color_logger_pickling()
    module_path, attr = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _patch_trading_color_logger_pickling() -> None:
    try:
        logger_module = importlib.import_module("utils.logger")
    except Exception:
        return
    color_logger_cls = getattr(logger_module, "ColorLogger", None)
    if color_logger_cls is None or getattr(color_logger_cls, "_research_pickle_patch", False):
        return

    def __getstate__(self):
        base_logger = self.__dict__.get("_logger")
        logger_name = getattr(base_logger, "name", None)
        if not logger_name:
            logger_name = getattr(self, "_logger_name", "") or ""
        return {"logger_name": str(logger_name or "")}

    def __setstate__(self, state):
        logger_name = str((state or {}).get("logger_name") or "")
        self._logger = logging.getLogger(logger_name)
        self._logger_name = logger_name

    original_init = getattr(color_logger_cls, "__init__", None)

    def patched_init(self, logger):
        if original_init is not None:
            original_init(self, logger)
        else:
            self._logger = logger
        self._logger_name = getattr(logger, "name", "")

    original_getattr = getattr(color_logger_cls, "__getattr__", None)

    def patched_getattr(self, name):
        base_logger = self.__dict__.get("_logger")
        if base_logger is None:
            logger_name = self.__dict__.get("_logger_name", "")
            base_logger = logging.getLogger(str(logger_name or ""))
            self._logger = base_logger
        if original_getattr is not None:
            return original_getattr(self, name)
        return getattr(base_logger, name)

    color_logger_cls.__init__ = patched_init
    color_logger_cls.__getattr__ = patched_getattr
    color_logger_cls.__getstate__ = __getstate__
    color_logger_cls.__setstate__ = __setstate__
    color_logger_cls._research_pickle_patch = True


def _load_algorithm_class(script_path: str, class_name: str):
    path = _resolve_generated_path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Generated algorithm script not found: {path}")
    module_name = f"generated_trading_{path.stem}_{uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algorithm_cls = getattr(module, class_name, None)
    if algorithm_cls is None:
        raise AttributeError(f"{class_name} not found in {path}")
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
    history_length = _effective_history_length(algorithm_params, fallback=int(algorithm_params.get("history_length") or 0))
    algorithm_params["history_length"] = history_length
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


def _run_hpo_backtest(
    runtime_cfg: dict[str, Any],
    algorithm_cls: type[Any],
    *,
    config_artifact_paths: list[str] | None = None,
    mlflow_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ray = importlib.import_module("ray")
    tune = importlib.import_module("ray.tune")
    optuna_mod = importlib.import_module("ray.tune.search.optuna")
    hpo_cfg = dict(runtime_cfg.get("hpo") or {})
    analysis_cfg = dict(runtime_cfg.get("analysis") or {})
    run_name = str(analysis_cfg.get("run_name") or "Backtest_HPO")
    total_samples = int(hpo_cfg.get("num_samples") or 50)
    max_concurrent_trials = int(hpo_cfg.get("max_concurrent_trials") or 8)
    objective_metric = str(hpo_cfg.get("objective_metric") or "annualized_return")
    log.info(
        "Trading HPO | running in Ray Tune dashboard=http://127.0.0.1:8265 trials=%s max_concurrent=%s run_name=%s",
        total_samples,
        max_concurrent_trials,
        run_name,
    )
    order_manager_cls = _import_dotted(str((runtime_cfg.get("order_manager") or {}).get("implementation")))
    data_provider_cls = _import_dotted(str((runtime_cfg.get("data_provider") or {}).get("implementation")))
    portfolio_cls = _import_dotted(str((runtime_cfg.get("portfolio") or {}).get("implementation")))

    algorithm_params = dict(((runtime_cfg.get("algorithm") or {}).get("params")) or {})
    portfolio_params = dict(((runtime_cfg.get("portfolio") or {}).get("params")) or {})
    data_provider_params = dict(((runtime_cfg.get("data_provider") or {}).get("params")) or {})
    mlflow_policy = dict(analysis_cfg.get("mlflow_policy") or {})

    starting_cash = float(portfolio_params.get("cash") or 100000.0)
    base_pf_cfg = {key: value for key, value in portfolio_params.items() if key not in {"cash", "keep_history"}}
    base_backtest_cfg = {
        "starting_cash": starting_cash,
        "experiment_name": analysis_cfg.get("experiment_name", "trading_agent_experiments"),
        "run_name": analysis_cfg.get("run_name", "Backtest_HPO"),
        "description": analysis_cfg.get("description", ""),
        "symbol": base_pf_cfg.get("symbol", algorithm_params.get("symbol", "SPY")),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
        "tracking_uri": ((runtime_cfg.get("mlflow") or {}).get("tracking_uri")) or "",
        "config_artifact_paths": list(config_artifact_paths or []),
    }

    ray.init(
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
    )
    trainable_with_params = tune.with_parameters(
        _backtest_objective_with_mlflow_policy,
        symbol=base_backtest_cfg["symbol"],
        algorithm_class=algorithm_cls,
        portfolio_class=portfolio_cls,
        data_provider_class=data_provider_cls,
        order_manager_class=order_manager_cls,
        base_algorithm_config=dict(algorithm_params),
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=data_provider_params,
        base_backtest_config=base_backtest_cfg,
        algorithm_param_keys=list(hpo_cfg.get("algorithm_param_keys") or []),
        portfolio_param_keys=list(hpo_cfg.get("portfolio_param_keys") or []),
        mlflow_policy=mlflow_policy,
        config_artifact_paths=list(config_artifact_paths or []),
        objective_metric=objective_metric,
    )
    _run_hpo_smoke_trials(
        search_space=dict(hpo_cfg.get("search_space") or {}),
        algorithm_param_keys=list(hpo_cfg.get("algorithm_param_keys") or []),
        portfolio_param_keys=list(hpo_cfg.get("portfolio_param_keys") or []),
        base_algorithm_config=dict(algorithm_params),
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=data_provider_params,
        base_backtest_config=base_backtest_cfg,
        algorithm_class=algorithm_cls,
        portfolio_class=portfolio_cls,
        data_provider_class=data_provider_cls,
        order_manager_class=order_manager_cls,
        mlflow_policy=mlflow_policy,
        objective_metric=objective_metric,
        smoke_trials=int(hpo_cfg.get("smoke_trials") or 3),
    )
    optuna_search = optuna_mod.OptunaSearch(metric="_metric", mode="max")
    tune_config_kwargs: dict[str, Any] = {
        "metric": "_metric",
        "mode": "max",
        "num_samples": total_samples,
        "max_concurrent_trials": max_concurrent_trials,
        "search_alg": optuna_search,
        "trial_name_creator": _tune_trial_name_creator,
        "trial_dirname_creator": _tune_trial_dirname_creator,
    }
    try:
        supported_tune_config = set(inspect.signature(tune.TuneConfig).parameters)
        tune_config_kwargs = {
            key: value
            for key, value in tune_config_kwargs.items()
            if key in supported_tune_config
        }
    except Exception:
        pass
    tuner = tune.Tuner(
        trainable_with_params,
        param_space=_parse_search_space_config(hpo_cfg.get("search_space") or {}),
        tune_config=tune.TuneConfig(**tune_config_kwargs),
        run_config=_build_tune_run_config(
            tune=tune,
            run_name=run_name,
            total_samples=total_samples,
            metric_name="_metric",
            progress_report_interval_seconds=int(hpo_cfg.get("progress_report_interval_seconds") or 15),
            log_progress=bool(hpo_cfg.get("log_progress", True)),
        ),
    )
    results = tuner.fit()
    best_result = _safe_best_tune_result(results)
    best_config = dict(getattr(best_result, "config", {}) or {})
    try:
        context = ray.get_runtime_context()
        dashboard_url = "http://127.0.0.1:8265"
        if hasattr(context, "gcs_address"):
            log.info(
                "Trading HPO | Ray initialized dashboard=%s gcs_address=%s",
                dashboard_url,
                getattr(context, "gcs_address", ""),
            )
        else:
            log.info("Trading HPO | Ray initialized dashboard=%s", dashboard_url)
    except Exception:
        log.info("Trading HPO | Ray initialized dashboard=http://127.0.0.1:8265")

    best_algorithm_cfg = dict(algorithm_params)
    for key in hpo_cfg.get("algorithm_param_keys") or []:
        if key in best_config:
            best_algorithm_cfg[key] = best_config[key]

    best_portfolio_cfg = dict(base_pf_cfg)
    for key in hpo_cfg.get("portfolio_param_keys") or []:
        if key in best_config:
            best_portfolio_cfg[key] = best_config[key]

    result = _run_backtest_local_with_mlflow_policy(
        backtest_cfg=base_backtest_cfg,
        alg_cfg=best_algorithm_cfg,
        pf_cfg=best_portfolio_cfg,
        dp_cfg=data_provider_params,
        algorithm_class=algorithm_cls,
        portfolio_class=portfolio_cls,
        data_provider_class=data_provider_cls,
        order_manager_class=order_manager_cls,
        mlflow_policy=mlflow_policy,
        config_artifact_paths=list(config_artifact_paths or []),
        mlflow_capture=mlflow_capture,
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


def _run_hpo_smoke_trials(
    *,
    search_space: dict[str, Any],
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
    base_algorithm_config: dict[str, Any],
    base_portfolio_config: dict[str, Any],
    base_data_provider_config: dict[str, Any],
    base_backtest_config: dict[str, Any],
    algorithm_class: type[Any],
    portfolio_class: type[Any],
    data_provider_class: type[Any],
    order_manager_class: type[Any],
    mlflow_policy: dict[str, Any],
    objective_metric: str,
    smoke_trials: int,
) -> None:
    samples = _smoke_trial_configs(search_space, limit=smoke_trials)
    if not samples:
        return
    failures: list[str] = []
    metrics: list[float] = []
    trade_counts: list[int] = []
    for sample in samples:
        try:
            sample = _repair_sampled_hpo_config(sample)
            alg_cfg = {**base_algorithm_config, **{k: sample[k] for k in algorithm_param_keys if k in sample}}
            pf_cfg = {**base_portfolio_config, **{k: sample[k] for k in portfolio_param_keys if k in sample}}
            result = _run_backtest_local_with_mlflow_policy(
                backtest_cfg=base_backtest_config,
                alg_cfg=alg_cfg,
                pf_cfg=pf_cfg,
                dp_cfg=base_data_provider_config,
                algorithm_class=algorithm_class,
                portfolio_class=portfolio_class,
                data_provider_class=data_provider_class,
                order_manager_class=order_manager_class,
                mlflow_policy={**mlflow_policy, "annualized_return_threshold": 1e18},
            )
            metrics.append(_metric_from_result(result["metrics"], objective_metric, fallback=float("nan")))
            trade_counts.append(len(result.get("trades") or []))
        except Exception as exc:
            failures.append(str(exc))
    if len(failures) == len(samples):
        raise ValueError(f"HPO smoke check failed: all {len(samples)} sampled configs crashed; first_error={failures[0]}")
    if trade_counts and max(trade_counts) <= 0:
        raise ValueError(f"HPO smoke check failed: {len(trade_counts)} sampled configs produced zero trades")
    if metrics and not any(math.isfinite(value) for value in metrics):
        raise ValueError(f"HPO smoke check failed: no finite {objective_metric} values")


def _smoke_trial_configs(search_space: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not search_space:
        return []
    configs: list[dict[str, Any]] = []
    fractions = [0.5, 0.2, 0.8, 0.35, 0.65]
    for fraction in fractions[: max(1, limit)]:
        configs.append({key: _sample_spec_midpoint(spec, fraction=fraction) for key, spec in search_space.items()})
    return configs


def _sample_spec_midpoint(spec: dict[str, Any], *, fraction: float) -> Any:
    spec_type = str((spec or {}).get("type") or "").lower()
    values = list((spec or {}).get("values") or [])
    if spec_type == "choice" and values:
        return values[min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))]
    low = float((spec or {}).get("low", 0.0))
    high = float((spec or {}).get("high", low))
    value = low + (high - low) * float(fraction)
    if spec_type == "randint":
        return max(int(low), min(int(high) - 1, int(round(value))))
    return round(value, 6)


def _repair_sampled_hpo_config(config: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(config)
    for low_key, high_key in (
        ("low_quantile", "high_quantile"),
        ("lower_quantile", "upper_quantile"),
        ("vol_low_quantile", "vol_high_quantile"),
        ("percentile_low", "percentile_high"),
        ("birth_death_ratio_lower", "birth_death_ratio_upper"),
        ("ratio_gate_lower_normal", "ratio_gate_upper_normal"),
        ("ratio_gate_lower_degraded", "ratio_gate_upper_degraded"),
        ("h0_percentile_low", "h0_percentile_high"),
        ("vol_persistence_low_pct", "vol_persistence_high_pct"),
    ):
        if low_key in repaired and high_key in repaired:
            low = float(repaired[low_key])
            high = float(repaired[high_key])
            if low >= high:
                midpoint = (low + high) / 2.0
                repaired[low_key] = midpoint * 0.8
                repaired[high_key] = midpoint * 1.2 if midpoint else high + 1.0
    for fast_key, slow_key in (
        ("fast_ma_period", "slow_ma_period"),
        ("momentum_fast_sma", "momentum_slow_sma"),
        ("fast_period", "slow_period"),
    ):
        if fast_key in repaired and slow_key in repaired:
            fast = int(repaired[fast_key])
            slow = int(repaired[slow_key])
            if fast >= slow:
                repaired[fast_key] = max(1, slow // 2)
    return repaired


def _backtest_objective_with_mlflow_policy(
    config: dict[str, Any],
    *,
    symbol: str,
    algorithm_class: type[Any],
    portfolio_class: type[Any],
    data_provider_class: type[Any],
    order_manager_class: type[Any],
    base_algorithm_config: dict[str, Any],
    base_portfolio_config: dict[str, Any],
    base_data_provider_config: dict[str, Any],
    base_backtest_config: dict[str, Any],
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
    mlflow_policy: dict[str, Any],
    config_artifact_paths: list[str] | None = None,
    objective_metric: str = "annualized_return",
) -> dict[str, Any]:
    try:
        config = _repair_sampled_hpo_config(dict(config))
        alg_params = {k: config[k] for k in algorithm_param_keys if k in config}
        alg_cfg = {**base_algorithm_config, **alg_params}
        pf_params = {k: config[k] for k in portfolio_param_keys if k in config}
        pf_cfg = {**base_portfolio_config, **pf_params}
        result = _run_backtest_local_with_mlflow_policy(
            backtest_cfg=base_backtest_config,
            alg_cfg=alg_cfg,
            pf_cfg=pf_cfg,
            dp_cfg=base_data_provider_config,
            algorithm_class=algorithm_class,
            portfolio_class=portfolio_class,
            data_provider_class=data_provider_class,
            order_manager_class=order_manager_class,
            mlflow_policy=mlflow_policy,
            config_artifact_paths=list(config_artifact_paths or []),
        )
        metric_value = _metric_from_result(result["metrics"], objective_metric, fallback=-1_000_000_000.0)
        return {"_metric": metric_value}
    except Exception as exc:
        log.warning("Trading HPO | trial failed: %s", exc, exc_info=True)
        return {"_metric": -1_000_000_000.0, "_trial_error": str(exc)}


def _run_backtest_local_with_mlflow_policy(
    *,
    backtest_cfg: dict[str, Any],
    alg_cfg: dict[str, Any],
    pf_cfg: dict[str, Any],
    dp_cfg: dict[str, Any],
    algorithm_class: type[Any],
    portfolio_class: type[Any],
    data_provider_class: type[Any],
    order_manager_class: type[Any],
    mlflow_policy: dict[str, Any],
    config_artifact_paths: list[str] | None = None,
    mlflow_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experiment_name = backtest_cfg["experiment_name"]
    starting_cash = backtest_cfg["starting_cash"]
    run_name = backtest_cfg["run_name"]
    desc = backtest_cfg["description"]
    config_artifact_path = backtest_cfg.get("config_artifact_path")
    git_tags = backtest_cfg.get("git_tags") or {}
    benchmark_paths = backtest_cfg.get("benchmark_paths") or {}
    tracking_uri = str(backtest_cfg.get("tracking_uri") or "")

    om = order_manager_class()
    effective_history_length = _effective_history_length(alg_cfg, fallback=int(alg_cfg.get("history_length") or 0))
    alg_cfg = {**alg_cfg, "history_length": effective_history_length}
    dp_cfg = dict(dp_cfg)
    minimum_limit = _minimum_alpaca_limit(dp_cfg, alg_cfg)
    if minimum_limit > int(dp_cfg.get("limit") or 0):
        dp_cfg["limit"] = minimum_limit
    try:
        al = algorithm_class(cfg=alg_cfg, history_length=effective_history_length)
    except TypeError:
        al = algorithm_class(alg_cfg, effective_history_length)
    dp = data_provider_class(dp_cfg)
    pf = portfolio_class(pf_cfg, om, starting_cash, {}, True)
    params = backtest_cfg | alg_cfg | pf_cfg | dp_cfg
    params.update({
        "algorithm_implementation": f"{algorithm_class.__module__}.{algorithm_class.__name__}",
        "portfolio_implementation": f"{portfolio_class.__module__}.{portfolio_class.__name__}",
        "data_provider_implementation": f"{data_provider_class.__module__}.{data_provider_class.__name__}",
        "order_manager_implementation": f"{order_manager_class.__module__}.{order_manager_class.__name__}",
    })

    sim = importlib.import_module("trading.engines.backtest_engine").BacktestingEngine({"state_store": {"enabled": False}}, dp, al, om, pf)
    sim.run()

    analysis_engine_cls = importlib.import_module("trading.analysis.analysis_engine").AnalysisEngine
    engine = analysis_engine_cls(sim.pf, pf.om)
    should_log_to_mlflow = _should_log_to_mlflow(
        annualized_return=float(getattr(engine.calculate_metrics(), "annualized_return", 0.0) or 0.0),
        run_name=run_name,
        params=params,
        sample_negative_rate=int(mlflow_policy.get("sample_negative_rate", 20) or 20),
        annualized_return_threshold=float(mlflow_policy.get("annualized_return_threshold", 0.0)),
    )
    results = engine.run_full_analysis(
        experiment_name=experiment_name,
        run_name=run_name,
        description=desc,
        parameters=params,
        tracking_uri=tracking_uri or None,
        log_to_mlflow=should_log_to_mlflow,
        save_charts_locally=False,
        save_report_locally=False,
        tags=git_tags if git_tags else None,
        artifact_paths=_artifact_path_list(config_artifact_path, config_artifact_paths),
        benchmark_paths=benchmark_paths if benchmark_paths else None,
    )
    return results


def _should_log_to_mlflow(
    *,
    annualized_return: float,
    run_name: str,
    params: dict[str, Any],
    sample_negative_rate: int,
    annualized_return_threshold: float,
) -> bool:
    if annualized_return > annualized_return_threshold:
        return True
    rate = max(1, int(sample_negative_rate or 20))
    signature = json.dumps({"run_name": run_name, "params": params}, sort_keys=True, default=_json_default)
    bucket = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8], 16)
    return bucket % rate == 0


def _artifact_path_list(primary_path: str | None, extra_paths: list[str] | None) -> list[str] | None:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in [primary_path, *(extra_paths or [])]:
        path = str(value or "").strip()
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered or None


def _install_mlflow_capture() -> dict[str, Any]:
    capture: dict[str, Any] = {}
    try:
        module = importlib.import_module("utils.mlflow_client")
        client_cls = getattr(module, "MLflowClient", None)
    except Exception:
        return capture
    if client_cls is None:
        return capture

    original_start = getattr(client_cls, "start_run", None)
    if not callable(original_start):
        return capture

    def _wrapped_start(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        try:
            capture["run_id"] = str(getattr(self, "run_id", None) or getattr(getattr(self, "_active_run", None), "info", None).run_id)
        except Exception:
            capture.setdefault("run_id", "")
        try:
            capture["run_url"] = str(self.get_run_url() or "")
        except Exception:
            capture.setdefault("run_url", "")
        capture["tracking_uri"] = str(getattr(self, "tracking_uri", "") or "")
        capture["experiment_name"] = str(getattr(self, "experiment_name", "") or "")
        capture["experiment_id"] = str(getattr(self, "experiment_id", "") or "")
        return result

    capture["_client_cls"] = client_cls
    capture["_original_start"] = original_start
    setattr(client_cls, "start_run", _wrapped_start)
    return capture


def _restore_mlflow_capture(capture: dict[str, Any] | None) -> None:
    if not capture:
        return
    client_cls = capture.get("_client_cls")
    original_start = capture.get("_original_start")
    if client_cls is not None and callable(original_start):
        setattr(client_cls, "start_run", original_start)


def _log_mlflow_config_artifacts(
    artifact_paths: list[str],
    *,
    capture: dict[str, Any] | None,
    extra_params: dict[str, Any] | None = None,
) -> None:
    if not artifact_paths:
        return
    run_id = str((capture or {}).get("run_id") or "")
    if not run_id:
        return
    try:
        mlflow = importlib.import_module("mlflow")
        with mlflow.start_run(run_id=run_id):
            for key, value in dict(extra_params or {}).items():
                if value is None or value == "":
                    continue
                mlflow.log_param(str(key), value)
            for path in artifact_paths:
                if Path(path).is_file():
                    mlflow.log_artifact(path, artifact_path="config")
    except Exception as exc:
        log.warning("Trading MLflow | failed to append config artifacts to run %s: %s", run_id, exc)


def _coerce_finite_metric(value: Any, *, fallback: float) -> float:
    try:
        metric = float(value)
    except Exception:
        return fallback
    if not math.isfinite(metric):
        return fallback
    return metric


def _metric_from_result(metrics: Any, name: str, *, fallback: float) -> float:
    if is_dataclass(metrics):
        value = getattr(metrics, name, None)
    elif isinstance(metrics, dict):
        value = metrics.get(name)
    else:
        value = getattr(metrics, name, None)
    return _coerce_finite_metric(value, fallback=fallback)


def _trial_status_counts(trials: list[Any]) -> dict[str, int]:
    counts = {
        "queued": 0,
        "running": 0,
        "done": 0,
        "errored": 0,
        "paused": 0,
        "other": 0,
    }
    for trial in trials:
        status = str(getattr(trial, "status", "") or "").upper()
        if status == "PENDING":
            counts["queued"] += 1
        elif status == "RUNNING":
            counts["running"] += 1
        elif status == "TERMINATED":
            counts["done"] += 1
        elif status == "ERROR":
            counts["errored"] += 1
        elif status == "PAUSED":
            counts["paused"] += 1
        else:
            counts["other"] += 1
    return counts


def _best_trial_metric(trials: list[Any], metric_name: str) -> float | None:
    best: float | None = None
    for trial in trials:
        result = getattr(trial, "last_result", None) or {}
        if not isinstance(result, dict):
            continue
        try:
            metric = float(result.get(metric_name))
        except Exception:
            continue
        if not math.isfinite(metric):
            continue
        if best is None or metric > best:
            best = metric
    return best


def _format_hpo_progress_message(
    *,
    run_name: str,
    trials: list[Any],
    total_samples: int | None,
    metric_name: str,
    done: bool,
) -> str:
    counts = _trial_status_counts(trials)
    total = int(total_samples or len(trials) or 0)
    best_metric = _best_trial_metric(trials, metric_name)
    best_fragment = "n/a" if best_metric is None else f"{best_metric:.6f}"
    state = "complete" if done else "running"
    return (
        f"Trading HPO progress | run_name={run_name} state={state} "
        f"done={counts['done']}/{total} running={counts['running']} "
        f"queued={counts['queued']} errored={counts['errored']} "
        f"paused={counts['paused']} best_{metric_name}={best_fragment}"
    )


def _short_identifier(value: str, *, prefix: str, max_length: int) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip()).strip("_").lower()
    if not raw:
        raw = prefix
    digest = hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:8]
    base = f"{prefix}_{raw}"
    room = max(1, int(max_length) - len(digest) - 1)
    shortened = base[:room].rstrip("_") or prefix
    return f"{shortened}_{digest}"


def _tune_trial_name_creator(trial: Any) -> str:
    trial_id = str(getattr(trial, "trial_id", "") or getattr(trial, "trial_name", "") or "trial")
    return _short_identifier(trial_id, prefix="trial", max_length=24)


def _tune_trial_dirname_creator(trial: Any) -> str:
    trial_id = str(getattr(trial, "trial_id", "") or getattr(trial, "trial_name", "") or "trial")
    return _short_identifier(trial_id, prefix="t", max_length=20)


def _make_tune_progress_callback(
    *,
    tune: Any,
    run_name: str,
    total_samples: int | None,
    metric_name: str,
    progress_report_interval_seconds: int,
) -> Any | None:
    callback_base = getattr(tune, "Callback", None)
    if callback_base is None:
        return None

    interval = max(1, int(progress_report_interval_seconds or 15))

    class TuneProgressLogger(callback_base):
        def __init__(self) -> None:
            self._last_report_monotonic = 0.0

        def _maybe_report(self, trials: list[Any], *, done: bool, force: bool = False) -> None:
            now = time.monotonic()
            if not force and self._last_report_monotonic and (now - self._last_report_monotonic) < interval:
                return
            self._last_report_monotonic = now
            log.info(
                _format_hpo_progress_message(
                    run_name=run_name,
                    trials=trials,
                    total_samples=total_samples,
                    metric_name=metric_name,
                    done=done,
                )
            )
            counts = _trial_status_counts(trials)
            offset = int(os.environ.get("RESEARCH_HPO_TRIAL_OFFSET") or 0)
            global_total = int(os.environ.get("RESEARCH_HPO_TRIAL_TOTAL") or 0) or (offset + int(total_samples or 0))
            terminal_progress.emit_hpo_update(
                done=offset + counts["done"] + counts["errored"],
                running=counts["running"],
                total=global_total,
                message=f"{run_name} {'complete' if done else 'running'}",
            )

        def on_step_end(self, iteration: int, trials: list[Any], **info: Any) -> None:
            self._maybe_report(trials, done=False)

        def on_trial_error(self, iteration: int, trials: list[Any], trial: Any, **info: Any) -> None:
            trial_id = getattr(trial, "trial_id", "") or getattr(trial, "trial_name", "")
            log.warning("Trading HPO | trial errored run_name=%s trial_id=%s", run_name, trial_id)
            self._maybe_report(trials, done=False, force=True)

        def on_experiment_end(self, trials: list[Any], **info: Any) -> None:
            self._maybe_report(trials, done=True, force=True)

    return TuneProgressLogger()


def _build_tune_run_config(
    *,
    tune: Any,
    run_name: str,
    total_samples: int | None,
    metric_name: str,
    progress_report_interval_seconds: int,
    log_progress: bool,
) -> Any | None:
    run_config_cls = getattr(tune, "RunConfig", None)
    if run_config_cls is None:
        try:
            run_config_cls = getattr(importlib.import_module("ray.air"), "RunConfig", None)
        except Exception:
            run_config_cls = None
    if run_config_cls is None:
        return None

    ray_storage_path = resolve_dev_path(".tmp/ray").resolve()
    ray_storage_path.mkdir(parents=True, exist_ok=True)
    ray_storage_dir = str(ray_storage_path)
    ray_storage_uri = ray_storage_path.as_uri()
    kwargs: dict[str, Any] = {
        "verbose": 1,
        "name": _short_tune_run_name(run_name),
    }
    if log_progress:
        callback = _make_tune_progress_callback(
            tune=tune,
            run_name=run_name,
            total_samples=total_samples,
            metric_name=metric_name,
            progress_report_interval_seconds=progress_report_interval_seconds,
        )
        if callback is not None:
            kwargs["callbacks"] = [callback]

    try:
        supported = set(inspect.signature(run_config_cls).parameters)
        if "storage_path" in supported:
            kwargs["storage_path"] = ray_storage_uri
        elif "local_dir" in supported:
            kwargs["local_dir"] = ray_storage_dir
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
    except Exception:
        kwargs.setdefault("storage_path", ray_storage_uri)
    try:
        return run_config_cls(**kwargs)
    except Exception:
        if "callbacks" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("callbacks", None)
            try:
                return run_config_cls(**retry_kwargs)
            except Exception:
                pass
        if "storage_path" in kwargs and "local_dir" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("local_dir", None)
            retry_kwargs.pop("callbacks", None)
            try:
                return run_config_cls(**retry_kwargs)
            except Exception:
                return None
        return None


def _short_tune_run_name(run_name: str) -> str:
    return _short_identifier(run_name, prefix="hpo", max_length=32)


def _safe_best_tune_result(results: Any):
    try:
        return results.get_best_result(metric="_metric", mode="max")
    except Exception as exc:
        log.warning("Trading HPO | no valid best trial found, falling back to baseline config: %s", exc)
        return type("TuneFallbackResult", (), {"config": {}})()


def _walk_forward_metrics(period_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    metric_rows = [asdict(item["metrics"]) for item in period_results if item.get("metrics") is not None and is_dataclass(item.get("metrics"))]
    if not metric_rows:
        return {
            "annualized_return": float(aggregate.get("annualized_return", 0.0) or aggregate.get("mean_annualized_return", 0.0) or 0.0),
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
