from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.brainstorm.config import load_brainstorm_config
from core.brainstorm.engine import BrainstormEngine, create_brainstorm_state


def test_apply_exit_sets_cancelled(minimal_profile):
    cfg = load_brainstorm_config()
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)

    updated = engine.apply_command(state, "exit")

    assert updated["status"] == "cancelled"


def test_apply_help_emits_help_text(minimal_profile):
    cfg = load_brainstorm_config()
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    messages: list[str] = []

    engine.apply_command(state, "help", emit=messages.append)

    assert any("Available commands" in message for message in messages)


def test_researcher_turn_uses_configured_tools(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [
            {
                "name": "researcher",
                "persona_type": "researcher",
                "enabled": True,
                "llm_key": "brainstorm",
                "tools": ["tests.test_brainstorm_engine.fake_research_tool"],
                "research_budget": {"max_artifacts_per_tool": 2},
            }
        ],
        "stop_policy": {"max_rounds_per_run": 1, "summary_interval_messages": 10, "pause_after_research_round": True},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    messages: list[str] = []

    updated = engine.run_until_pause(state, emit=messages.append)

    assert updated["status"] == "awaiting_user"
    assert any("Synthetic artifact" in message for message in messages)


def fake_research_tool(direction, profile, tool_cfg, state):
    return [{
        "artifact_id": "artifact-1",
        "source": tool_cfg.get("name", "researcher"),
        "source_type": "paper",
        "title": "Synthetic artifact",
        "summary": f"Evidence for {direction}",
        "metadata": {},
        "raw": {},
    }]
