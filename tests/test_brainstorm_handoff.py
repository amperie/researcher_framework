from __future__ import annotations

from core.brainstorm.handoff import build_execution_handoff, choose_start_node


def test_brainstorm_handoff_prefers_implement_when_plan_exists():
    state = {
        "profile_name": "trading_researcher",
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
        "profile_name": "trading_researcher",
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


def test_brainstorm_handoff_skips_direct_implement_for_placeholder_class_name():
    state = {
        "profile_name": "trading_researcher",
        "session_id": "s1",
        "current_goal": "test",
        "plan_draft": {
            "research_direction": "test",
            "refined_ideas": [],
            "proposals": [{"name": "proposal-a"}],
            "implementation_plans": [{"proposal_name": "proposal-a", "class_name": "UnknownClass"}],
            "constraints": [],
            "exclusions": [],
            "success_criteria": [],
            "unresolved_questions": [],
        },
    }
    cfg = {"name": "default", "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True}}

    assert choose_start_node(build_execution_handoff(state, cfg), cfg) == "plan_implementation"


def test_brainstorm_handoff_skips_direct_implement_for_generated_plan_placeholder():
    state = {
        "profile_name": "trading_researcher",
        "session_id": "s1",
        "current_goal": "test",
        "plan_draft": {
            "research_direction": "test",
            "refined_ideas": [],
            "proposals": [{"name": "proposal-a"}],
            "implementation_plans": [{"proposal_name": "proposal-a", "class_name": "GeneratedPlan"}],
            "constraints": [],
            "exclusions": [],
            "success_criteria": [],
            "unresolved_questions": [],
        },
    }
    cfg = {"name": "default", "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True}}

    assert choose_start_node(build_execution_handoff(state, cfg), cfg) == "plan_implementation"


def test_brainstorm_handoff_preserves_imported_lineage():
    state = {
        "profile_name": "trading_researcher",
        "session_id": "s1",
        "current_goal": "test",
        "source_experiment_record_id": "exp-1",
        "source_next_step_record_id": "next_step:1",
        "source_next_step_title": "Try stronger baseline",
        "root_run_family_id": "family-1",
        "root_research_direction": "initial direction",
        "campaign_id": "campaign-1",
        "campaign_variant_index": 2,
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

    handoff = build_execution_handoff(state, cfg)

    assert handoff["source_experiment_record_id"] == "exp-1"
    assert handoff["source_next_step_record_id"] == "next_step:1"
    assert handoff["root_run_family_id"] == "family-1"
    assert handoff["campaign_id"] == "campaign-1"
    assert handoff["campaign_variant_index"] == 2
