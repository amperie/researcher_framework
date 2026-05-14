from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from core.tools.research_tool_catalog import ResearchToolCatalogError, resolve_research_tool_refs
from core.utils.logger import get_logger

log = get_logger(__name__)

BRAINSTORM_CONFIG_DIR = Path("configs/brainstorm")
DEFAULT_BRAINSTORM_CONFIG = BRAINSTORM_CONFIG_DIR / "default.brainstorm.yaml"


class BrainstormConfigError(ValueError):
    pass


def load_brainstorm_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_brainstorm_config_path(path)
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
        try:
            normalized["tools"] = resolve_research_tool_refs(list(role.get("tools") or []), path_key="path")
        except ResearchToolCatalogError as exc:
            raise BrainstormConfigError(str(exc)) from exc
        normalized["research_budget"] = dict(role.get("research_budget") or {})
        normalized["prompt_overrides"] = dict(role.get("prompt_overrides") or {})
        normalized_roles.append(normalized)
    cfg["roles"] = normalized_roles
    log.info("brainstorm.config | Loaded brainstorm config=%r roles=%d", cfg["name"], len(normalized_roles))
    return cfg


def list_brainstorm_configs() -> list[Path]:
    if not BRAINSTORM_CONFIG_DIR.exists():
        return []
    paths = [
        path for path in BRAINSTORM_CONFIG_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    ]
    return sorted(paths, key=lambda item: item.name.lower())


def resolve_brainstorm_config_path(path: str | None = None) -> Path:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        # Backward compatibility for older saved sessions and docs that still
        # reference the previous default brainstorm filename.
        if candidate.name == DEFAULT_BRAINSTORM_CONFIG.name:
            configs = list_brainstorm_configs()
            if len(configs) == 1:
                return configs[0]
        return candidate
    if DEFAULT_BRAINSTORM_CONFIG.exists():
        return DEFAULT_BRAINSTORM_CONFIG
    configs = list_brainstorm_configs()
    if len(configs) == 1:
        return configs[0]
    if not configs:
        raise BrainstormConfigError(f"No brainstorm configs found under: {BRAINSTORM_CONFIG_DIR}")
    names = ", ".join(item.name for item in configs[:5])
    more = f" (+{len(configs) - 5} more)" if len(configs) > 5 else ""
    raise BrainstormConfigError(
        "Multiple brainstorm configs found and no default selected. "
        f"Pass --brainstorm-config/--config. Available: {names}{more}"
    )


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
            "max_items_per_section": 3,
            "evidence_summary_chars": 80,
        },
        "prompts": {
            "role_turn": {
                "system_template": (
                    "You are the {role_name} persona in a brainstorming session.\n"
                    "Role type: {role_type}\n"
                    "Goal: {role_goal}\n"
                    "Style: {role_style}\n"
                    "Respond briefly and concretely. Use at most 4 short bullets. "
                    "Prefer fragments over paragraphs. No preamble, no recap. "
                    "Mention weak assumptions only if material."
                ),
                "human_template": (
                    "Current goal: {current_goal}\n\n"
                    "Current consensus summary:\n{consensus_summary}\n\n"
                    "Recent user notes:\n{user_notes_json}"
                ),
            },
            "consensus": {
                "system_template": (
                    "Summarize the brainstorming session into JSON with keys: "
                    "agreed_points, active_options, rejected_options, objections, assumptions, "
                    "open_questions, next_recommendation, confidence. "
                    "Use arrays for lists. Keep each list short, with terse entries."
                ),
                "human_template": "{consensus_input_json}",
            },
            "plan": {
                "system_template": (
                    "Generate a structured brainstorming plan as JSON with keys: "
                    "research_direction, refined_ideas, proposals, implementation_plans, "
                    "constraints, exclusions, success_criteria, unresolved_questions. "
                    "Keep entries terse and avoid explanatory prose."
                ),
                "human_template": "{plan_input_json}",
            },
            "researcher": {
                "result_line_template": "- {title}: {summary_excerpt}",
                "empty_message": "No evidence gathered.",
                "summary_excerpt_chars": 100,
            },
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
