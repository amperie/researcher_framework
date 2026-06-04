from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.brainstorm.commands import HELP_TEXT, parse_brainstorm_command
from core.brainstorm.engine import BrainstormEngine, create_brainstorm_state


def test_help_text_mentions_exit():
    assert "exit" in HELP_TEXT
    assert "help" in HELP_TEXT


def test_parse_exit_command():
    assert parse_brainstorm_command("exit")["type"] == "exit"


def test_parse_feedback_command_with_raw_text():
    parsed = parse_brainstorm_command("We should ignore the dataset issue for now")
    assert parsed["type"] == "feedback"
    assert "ignore the dataset issue" in parsed["text"]


def test_approve_plan_marks_plan_approved_but_does_not_start_execution(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [],
        "stop_policy": {},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    state["plan_draft"]["proposals"] = [{"name": "proposal-a"}]
    messages: list[str] = []

    updated = engine.apply_command(state, "approve_plan", emit=messages.append)

    assert updated["approved_plan"] is True
    assert updated["status"] == "awaiting_user"
    assert any("Use `execute`" in message for message in messages)


def test_plan_command_prints_structured_plan(minimal_profile, monkeypatch):
    cfg = {
        "name": "test_brainstorm",
        "roles": [],
        "stop_policy": {},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    messages: list[str] = []

    monkeypatch.setattr(
        engine,
        "_draft_plan",
        lambda current_state: current_state["plan_draft"].update(
            {
                "research_direction": "direction",
                "proposals": [{"name": "proposal-a", "description": "Test proposal"}],
                "constraints": ["keep it simple"],
            }
        ),
    )

    engine.apply_command(state, "plan", emit=messages.append)

    rendered = "".join(messages)
    assert "[Plan]" in rendered
    assert "Proposals:" in rendered
    assert "proposal-a: Test proposal" in rendered
    assert "Constraints:" in rendered


def test_feedback_resets_stale_consensus_and_preserves_seed_context(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [],
        "stop_policy": {},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    state["turn_log"] = [
        {"message_type": "seed_import", "content": "seed"},
        {"message_type": "discussion", "content": "stale"},
    ]
    state["consensus"] = {
        "agreed_points": ["old point"],
        "active_options": [{"name": "old option"}],
        "rejected_options": [],
        "objections": [],
        "assumptions": [],
        "open_questions": ["old question"],
        "next_recommendation": "old next",
        "confidence": "high",
        "evidence": [
            {"title": "Imported run", "summary": "seed context"},
            {"title": "Irrelevant paper", "summary": "stale research"},
        ],
    }
    state["seed_context_artifacts"] = [{"title": "Imported run", "summary": "seed context"}]
    state["plan_draft"]["proposals"] = [{"name": "old proposal"}]
    state["approved_plan"] = True
    engine = BrainstormEngine(minimal_profile, cfg)
    messages: list[str] = []

    updated = engine.apply_command(state, "feedback focus only on macro regime filters", emit=messages.append)

    assert updated["current_goal"] == "focus only on macro regime filters"
    assert updated["consensus"]["agreed_points"] == []
    assert updated["consensus"]["evidence"] == [{"title": "Imported run", "summary": "seed context"}]
    assert updated["plan_draft"]["proposals"] == []
    assert updated["approved_plan"] is False
    assert len(updated["turn_log"]) == 1
    assert any("consensus was cleared" in message for message in messages)


def test_plan_command_normalizes_string_proposals_and_implementation_plans(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [],
        "stop_policy": {},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    messages: list[str] = []
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(
        content=(
            '{"research_direction":"direction",'
            '"proposals":["use macro regime filter"],'
            '"implementation_plans":["{\\"proposal_name\\": \\"proposal_1\\", \\"class_name\\": \\"MacroRegimeFilter\\"}"],'
            '"constraints":[],"exclusions":[],"success_criteria":[],"unresolved_questions":[]}'
        )
    )

    with patch("core.brainstorm.engine.get_llm", return_value=fake_llm):
        engine.apply_command(state, "plan", emit=messages.append)

    assert state["plan_draft"]["proposals"][0]["description"] == "use macro regime filter"
    assert state["plan_draft"]["implementation_plans"][0]["class_name"] == "MacroRegimeFilter"
    assert any("coerced string proposal entry" in error for error in state["errors"])


def test_plan_command_compacts_large_evidence_before_llm(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [{"name": "planner", "persona_type": "planner"}],
        "stop_policy": {},
        "summary": {},
        "prompts": {"plan": {"max_evidence_items": 2, "max_text_chars": 120}},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    huge = "x" * 100_000
    state["consensus"]["evidence"] = [
        {
            "artifact_id": f"a-{idx}",
            "source": "arxiv",
            "title": f"paper {idx}",
            "summary": huge,
            "raw": {"full_text": huge},
            "metadata": {"blob": huge},
        }
        for idx in range(4)
    ]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(
        content='{"research_direction":"direction","proposals":[],"implementation_plans":[]}'
    )

    with patch("core.brainstorm.engine.get_llm", return_value=fake_llm):
        engine.apply_command(state, "plan", emit=lambda _message: None)

    human_prompt = fake_llm.invoke.call_args.args[0][1].content
    assert len(human_prompt) < 5000
    assert "full_text" not in human_prompt
    assert "metadata" not in human_prompt
    assert human_prompt.count("paper ") == 2
