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
        "brainstorm_config_name": str(brainstorm_cfg.get("name") or ""),
    }


def choose_start_node(handoff: dict[str, Any], brainstorm_cfg: dict[str, Any]) -> str:
    if handoff.get("implementation_plans") and (brainstorm_cfg.get("execution_handoff") or {}).get("allow_direct_to_implement", True):
        return "implement"
    if handoff.get("proposals"):
        return "plan_implementation"
    return str(((brainstorm_cfg.get("execution_handoff") or {}).get("default_start_node") or "propose_experiments"))
