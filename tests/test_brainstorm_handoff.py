from __future__ import annotations

from core.brainstorm.handoff import build_execution_handoff, choose_start_node


def test_brainstorm_handoff_prefers_implement_when_plan_exists():
    state = {
        "profile_name": "neuralsignal",
        "session_id": "s1",
        "current_goal": "test",
        "plan_draft": {
            "research_direction": "test",
            "refined_ideas": [],
            "proposals": [{"name": "proposal-a"}],
            "implementation_plans": [{"proposal_name": "proposal-a", "class_name": "DetectorA"}],
            "constraints": [],
            "exclusions": [],
            "success_criteria": [],
            "unresolved_questions": [],
        },
    }
    cfg = {"name": "default", "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True}}

    handoff = build_execution_handoff(state, cfg)

    assert choose_start_node(handoff, cfg) == "implement"
    assert handoff["source_brainstorm_session_id"] == "s1"


def test_brainstorm_handoff_falls_back_to_plan_implementation():
    state = {
        "profile_name": "neuralsignal",
        "session_id": "s1",
        "current_goal": "test",
        "plan_draft": {
            "research_direction": "test",
            "refined_ideas": [],
            "proposals": [{"name": "proposal-a"}],
            "implementation_plans": [],
            "constraints": [],
            "exclusions": [],
            "success_criteria": [],
            "unresolved_questions": [],
        },
    }
    cfg = {"name": "default", "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True}}

    assert choose_start_node(build_execution_handoff(state, cfg), cfg) == "plan_implementation"
