from __future__ import annotations

from core.brainstorm.engine import create_brainstorm_state
from core.brainstorm.seeding import resolve_brainstorm_seed


def test_create_brainstorm_state_imports_seeded_context():
    cfg = {
        "name": "test_brainstorm",
        "path": "configs/brainstorm/test.yaml",
        "roles": [],
        "stop_policy": {},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    seed = {
        "research_direction": "follow up on proposal-a",
        "proposals": [{"name": "proposal-a", "rationale": "Keep the core idea but simplify implementation."}],
        "proposal_seed_planning_notes": "Preserve the original dataset.",
        "source_experiment_record_id": "exp-1",
        "source_proposal_seed_record_id": "proposal_seed:1",
        "root_run_family_id": "family-1",
        "root_research_direction": "initial direction",
        "context_artifacts": [{
            "title": "proposal-a",
            "summary": "Prior run looked promising but overfit.",
            "source_type": "experiment_result",
            "metadata": {"record_id": "exp-1"},
        }],
        "import_turn_content": "Imported context from run=exp-1, proposal_seed=proposal_seed:1",
    }

    state = create_brainstorm_state(
        profile_name="test_profile",
        direction="fallback direction",
        brainstorm_cfg=cfg,
        seed=seed,
    )

    assert state["current_goal"] == "follow up on proposal-a"
    assert state["plan_draft"]["proposals"][0]["name"] == "proposal-a"
    assert state["source_experiment_record_id"] == "exp-1"
    assert state["consensus"]["evidence"][0]["title"] == "proposal-a"
    assert "Preserve the original dataset." in state["user_intent_notes"]
    assert state["turn_log"][0]["message_type"] == "seed_import"


def test_resolve_brainstorm_seed_from_source_experiment_imports_proposal(minimal_profile, monkeypatch):
    run_record = {
        "record_id": "exp-1",
        "title": "proposal-a",
        "summary": "Promising run with a weak baseline.",
        "metadata": {
            "research_direction": "test direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "assessment": "promising",
            "campaign_id": "campaign-1",
        },
    }

    monkeypatch.setattr("core.brainstorm.seeding.load_run_record", lambda _profile, _record_id: run_record)
    monkeypatch.setattr(
        "core.brainstorm.seeding.proposal_template_from_run",
        lambda _profile, _run_record: {"name": "proposal-a", "description": "Imported from prior run."},
    )

    seed = resolve_brainstorm_seed(minimal_profile, source_experiment_record_id="exp-1")

    assert seed["research_direction"] == "test direction"
    assert seed["source_experiment_record_id"] == "exp-1"
    assert seed["proposals"][0]["name"] == "proposal-a"
    assert seed["campaign_id"] == "campaign-1"
    assert any(item["source_type"] == "experiment_result" for item in seed["context_artifacts"])

