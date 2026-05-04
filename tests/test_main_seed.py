from __future__ import annotations

from argparse import Namespace

import main


def test_resolve_initial_seed_uses_saved_handoff(minimal_profile, monkeypatch):
    monkeypatch.setattr(
        main,
        "resolve_run_handoff_seed",
        lambda profile, **kwargs: {
            "research_direction": "saved direction",
            "source_next_step_record_id": "run_handoff:1",
            "source_next_step_title": "Saved handoff",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    )

    seed = main._resolve_initial_seed(
        minimal_profile,
        "test_profile",
        Namespace(direction=None, source_experiment="exp-1", handoff="run_handoff:1", proposal_seed=None, next_step=None),
    )

    assert seed["research_direction"] == "saved direction"
    assert seed["source_next_step_record_id"] == "run_handoff:1"


def test_resolve_initial_seed_uses_saved_proposal_seed(minimal_profile, monkeypatch):
    monkeypatch.setattr(
        main,
        "resolve_proposal_seed",
        lambda profile, **kwargs: {
            "research_direction": "seeded direction",
            "proposals": [{"name": "proposal-a"}],
            "proposal_seed_planning_notes": "Keep it minimal.",
            "source_proposal_seed_record_id": "proposal_seed:1",
            "source_proposal_seed_title": "Seeded proposal",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    )

    seed = main._resolve_initial_seed(
        minimal_profile,
        "test_profile",
        Namespace(direction=None, source_experiment="exp-1", handoff=None, proposal_seed="proposal_seed:1", next_step=None),
    )

    assert seed["research_direction"] == "seeded direction"
    assert seed["proposals"][0]["name"] == "proposal-a"
    assert seed["source_proposal_seed_record_id"] == "proposal_seed:1"


def test_resolve_initial_seed_uses_next_step(minimal_profile, monkeypatch):
    monkeypatch.setattr(
        main,
        "resolve_next_step_seed",
        lambda profile, **kwargs: {
            "research_direction": "run the promoted step",
            "source_next_step_record_id": "next_step:1",
            "source_next_step_title": "Promoted step",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    )

    seed = main._resolve_initial_seed(
        minimal_profile,
        "test_profile",
        Namespace(direction=None, source_experiment=None, handoff=None, proposal_seed=None, next_step="next_step:1"),
    )

    assert seed["research_direction"] == "run the promoted step"
    assert seed["source_next_step_record_id"] == "next_step:1"
