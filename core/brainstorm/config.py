from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from core.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_BRAINSTORM_CONFIG = Path("configs/brainstorm/default.brainstorm.yaml")


class BrainstormConfigError(ValueError):
    pass


def load_brainstorm_config(path: str | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_BRAINSTORM_CONFIG)
    if not config_path.exists():
        raise BrainstormConfigError(f"Brainstorm config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise BrainstormConfigError(f"Brainstorm config must be a mapping: {config_path}")

    cfg = _default_brainstorm_config()
    cfg = _deep_merge(cfg, raw)
    cfg["path"] = str(config_path)
    cfg["name"] = str(cfg.get("name") or config_path.stem)

    roles = cfg.get("roles") or []
    if not isinstance(roles, list) or not roles:
        raise BrainstormConfigError("Brainstorm config must define at least one role")

    default_model = str(((cfg.get("llm_defaults") or {}).get("model") or "")).strip()
    default_provider = str(((cfg.get("llm_defaults") or {}).get("provider") or "")).strip()
    normalized_roles: list[dict[str, Any]] = []
    for role in roles:
        if not isinstance(role, dict):
            raise BrainstormConfigError(f"Invalid role entry: {role!r}")
        name = str(role.get("name") or "").strip()
        persona_type = str(role.get("persona_type") or "").strip()
        if not name or not persona_type:
            raise BrainstormConfigError(f"Role missing name/persona_type: {role!r}")
        normalized = dict(role)
        normalized["name"] = name
        normalized["persona_type"] = persona_type
        normalized["enabled"] = bool(role.get("enabled", True))
        normalized["llm_key"] = str(role.get("llm_key") or "brainstorm")
        normalized["model"] = str(role.get("model") or default_model or "").strip()
        normalized["provider"] = str(role.get("provider") or default_provider or "").strip()
        normalized["tools"] = [str(item) for item in (role.get("tools") or [])]
        normalized["research_budget"] = dict(role.get("research_budget") or {})
        normalized_roles.append(normalized)
    cfg["roles"] = normalized_roles
    log.info("brainstorm.config | Loaded brainstorm config=%r roles=%d", cfg["name"], len(normalized_roles))
    return cfg


def _default_brainstorm_config() -> dict[str, Any]:
    return {
        "name": "default_brainstorm",
        "description": "Default interactive brainstorm setup",
        "llm_defaults": {
            "provider": "",
            "model": "",
        },
        "stop_policy": {
            "max_rounds_per_run": 3,
            "max_messages_per_run": 12,
            "max_seconds_per_run": 90,
            "summary_interval_messages": 4,
            "summary_interval_seconds": 20,
            "pause_after_research_round": True,
        },
        "summary": {
            "print_current_thinking": True,
            "include_evidence": True,
            "include_open_questions": True,
        },
        "execution_handoff": {
            "default_start_node": "propose_experiments",
            "allow_direct_to_implement": True,
        },
        "roles": [],
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
