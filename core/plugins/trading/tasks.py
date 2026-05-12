"""Trading task callables for the shared external task runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from core.utils.temp_paths import temporary_directory


def run_trading_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one prepared trading artifact in a trading-capable process."""
    from core.plugins.trading.adapter import (
        TradingAdapter,
        _json_default,
        _normalize_runtime_config,
        _public_runtime_config,
        _summarize_variant_results,
    )

    adapter = TradingAdapter()
    profile = dict(payload.get("profile") or {})
    artifact = dict(payload.get("artifact") or {})
    proposal = dict(payload.get("proposal") or {})

    proposal_name = str(artifact.get("proposal_name") or "unknown")
    class_name = str(artifact.get("class_name") or proposal_name)
    script_source = str(payload.get("script_source") or "")
    runtime_cfg = _normalize_runtime_config(dict(artifact.get("runtime_config") or {}))
    variant_specs = list(artifact.get("variant_specs") or [{"name": "base", "overrides": {}}])

    with temporary_directory(prefix="rf_trading_", category="trading") as tmpdir:
        script_path = Path(tmpdir) / f"{class_name}.py"
        script_path.write_text(script_source, encoding="utf-8")
        config_artifact_paths = _write_repro_bundle(
            root=Path(tmpdir),
            class_name=class_name,
            script_filename=script_path.name,
            script_source=script_source,
            runtime_cfg=runtime_cfg,
        )
        variant_results = [
            adapter._run_variant(
                profile=profile,
                proposal_name=proposal_name,
                class_name=class_name,
                script_path=str(script_path),
                runtime_cfg=runtime_cfg,
                variant_spec=variant_spec,
                config_artifact_paths=config_artifact_paths,
            )
            for variant_spec in variant_specs
        ]
    summary = _summarize_variant_results(variant_results)
    experiment_id = str(artifact.get("experiment_id") or uuid4())
    primary_mlflow = _primary_mlflow_metadata(variant_results)
    result = {
        "experiment_id": experiment_id,
        "proposal_name": proposal_name,
        "class_name": class_name,
        "proposal": proposal,
        "metrics": summary["metrics"],
        "execution_config": _public_runtime_config(runtime_cfg),
        "variant_results": variant_results,
        "report": summary.get("report", ""),
        "mlflow_run_id": primary_mlflow.get("run_id", ""),
        "mlflow_run_url": primary_mlflow.get("run_url", ""),
    }
    result["artifacts"] = {
        "runtime_config_path": str(artifact.get("config_path") or ""),
        "variant_count": len(variant_results),
        "runtime_config": _public_runtime_config(runtime_cfg),
        "config_artifacts_logged": [Path(path).name for path in config_artifact_paths],
    }
    return result


def _write_repro_bundle(
    *,
    root: Path,
    class_name: str,
    script_filename: str,
    script_source: str,
    runtime_cfg: dict[str, Any],
) -> list[str]:
    from core.plugins.trading.adapter import _public_runtime_config

    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    public_cfg = _public_runtime_config(runtime_cfg)
    algorithm_cfg = dict(public_cfg.get("algorithm") or {})
    algorithm_cfg["class_name"] = class_name
    algorithm_cfg["source_file"] = script_filename

    files: dict[str, Any] = {
        script_filename: script_source,
        "runtime_config.json": public_cfg,
        "runtime_config.yaml": public_cfg,
        "algorithm_config.json": algorithm_cfg,
        "portfolio_config.json": public_cfg.get("portfolio") or {},
        "data_provider_config.json": public_cfg.get("data_provider") or {},
        "order_manager_config.json": public_cfg.get("order_manager") or {},
        "analysis_config.json": public_cfg.get("analysis") or {},
        "hpo_config.json": public_cfg.get("hpo") or {},
        "optimization_config.json": public_cfg.get("optimization") or {},
        "generated_algorithm.json": public_cfg.get("generated_algorithm") or {},
    }

    written: list[str] = []
    for filename, payload in files.items():
        path = config_dir / filename
        if filename.endswith(".py"):
            path.write_text(str(payload), encoding="utf-8")
        elif filename.endswith(".yaml"):
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written.append(str(path))
    return written


def _primary_mlflow_metadata(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    for item in variant_results:
        mlflow_meta = dict(item.get("mlflow") or {})
        if mlflow_meta.get("run_id"):
            return mlflow_meta
    return {}
