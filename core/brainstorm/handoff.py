from __future__ import annotations

from typing import Any


def build_execution_handoff(state: dict[str, Any], brainstorm_cfg: dict[str, Any]) -> dict[str, Any]:
    plan = dict(state.get("plan_draft") or {})
    return {
        "profile_name": str(state.get("profile_name") or ""),
        "research_direction": str(plan.get("research_direction") or state.get("current_goal") or ""),
        "refined_ideas": list(plan.get("refined_ideas") or []),
        "proposals": list(plan.get("proposals") or []),
        "implementation_plans": list(plan.get("implementation_plans") or []),
        "proposal_seed_planning_notes": "\n".join(str(item) for item in (plan.get("constraints") or [])),
        "constraints": list(plan.get("constraints") or []),
        "exclusions": list(plan.get("exclusions") or []),
        "source_brainstorm_session_id": str(state.get("session_id") or ""),
        "source_experiment_record_id": str(state.get("source_experiment_record_id") or ""),
        "source_next_step_record_id": str(state.get("source_next_step_record_id") or ""),
        "source_next_step_title": str(state.get("source_next_step_title") or ""),
        "source_proposal_seed_record_id": str(state.get("source_proposal_seed_record_id") or ""),
        "source_proposal_seed_title": str(state.get("source_proposal_seed_title") or ""),
        "root_run_family_id": str(state.get("root_run_family_id") or ""),
        "root_research_direction": str(state.get("root_research_direction") or ""),
        "campaign_id": str(state.get("campaign_id") or ""),
        "campaign_title": str(state.get("campaign_title") or ""),
        "campaign_variant_id": str(state.get("campaign_variant_id") or ""),
        "campaign_variant_title": str(state.get("campaign_variant_title") or ""),
        "campaign_variant_index": int(state.get("campaign_variant_index") or 0),
        "campaign_size": int(state.get("campaign_size") or 0),
        "brainstorm_config_name": str(brainstorm_cfg.get("name") or ""),
    }


def choose_start_node(handoff: dict[str, Any], brainstorm_cfg: dict[str, Any]) -> str:
    if (
        _has_concrete_implementation_plan(handoff.get("implementation_plans") or [])
        and (brainstorm_cfg.get("execution_handoff") or {}).get("allow_direct_to_implement", True)
    ):
        return "implement"
    if handoff.get("proposals"):
        return "plan_implementation"
    return str(((brainstorm_cfg.get("execution_handoff") or {}).get("default_start_node") or "propose_experiments"))


def _has_concrete_implementation_plan(plans: list[dict[str, Any]]) -> bool:
    for item in plans:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name") or "").strip()
        if not class_name or class_name.lower() in {"unknownclass", "generatedplan"}:
            continue
        return True
    return False
