from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.brainstorm.config import load_brainstorm_config
from core.brainstorm.engine import BrainstormEngine, create_brainstorm_state


def _pass_through_scored(artifacts):
    scored = []
    for artifact in artifacts:
        item = dict(artifact)
        item.setdefault("relevance_score", 9)
        item.setdefault("score_threshold", 6)
        scored.append(item)
    return scored


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
                "tools": [{"path": "tests.test_brainstorm_engine.fake_research_tool", "name": "fake_research"}],
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
    engine._score_brainstorm_research = lambda _state, _role, artifacts, _tool_defs: _pass_through_scored(artifacts)  # type: ignore[method-assign]

    updated = engine.run_until_pause(state, emit=messages.append)

    assert updated["status"] == "awaiting_user"
    assert any("Synthetic artifact" in message for message in messages)


def test_researcher_turn_passes_tool_config_and_enforces_max_tools_per_round(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [
            {
                "name": "researcher",
                "persona_type": "researcher",
                "enabled": True,
                "llm_key": "brainstorm",
                "tools": [
                    {"path": "tests.test_brainstorm_engine.fake_research_tool", "name": "first_tool", "n_results": 7},
                    {"path": "tests.test_brainstorm_engine.second_fake_research_tool", "name": "second_tool"},
                ],
                "research_budget": {"max_tools_per_round": 1, "max_artifacts_per_tool": 3},
            }
        ],
        "stop_policy": {"max_rounds_per_run": 1, "summary_interval_messages": 10, "pause_after_research_round": True},
        "summary": {},
        "prompts": {"researcher": {"result_line_template": "- {source}: {summary_excerpt}", "summary_excerpt_chars": 100}},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    engine._score_brainstorm_research = lambda _state, _role, artifacts, _tool_defs: _pass_through_scored(artifacts)  # type: ignore[method-assign]

    updated = engine.run_until_pause(state)

    assert updated["status"] == "awaiting_user"
    researcher_turns = [item for item in updated["turn_log"] if item.get("role_type") == "researcher"]
    assert len(researcher_turns) == 1
    assert "first_tool" in researcher_turns[0]["content"]
    assert "second_tool" not in researcher_turns[0]["content"]


def test_researcher_turn_filters_irrelevant_artifacts(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [
            {
                "name": "researcher",
                "persona_type": "researcher",
                "enabled": True,
                "llm_key": "brainstorm",
                "tools": [{"path": "tests.test_brainstorm_engine.mixed_research_tool", "name": "mixed_tool", "score_threshold": 6}],
                "research_budget": {"max_artifacts_per_tool": 4, "max_evidence_items": 3},
            }
        ],
        "stop_policy": {"max_rounds_per_run": 1, "summary_interval_messages": 10, "pause_after_research_round": True},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        SimpleNamespace(content='{"score": 9, "reason": "relevant"}'),
        SimpleNamespace(content='{"score": 1, "reason": "irrelevant"}'),
    ]

    with patch("core.brainstorm.engine.get_llm", return_value=fake_llm):
        updated = engine.run_until_pause(state)

    researcher_turns = [item for item in updated["turn_log"] if item.get("role_type") == "researcher"]
    assert len(researcher_turns) == 1
    assert "Relevant artifact" in researcher_turns[0]["content"]
    assert "Irrelevant artifact" not in researcher_turns[0]["content"]
    assert [item["title"] for item in researcher_turns[0]["structured_points"]] == ["Relevant artifact"]


def fake_research_tool(direction, profile, tool_cfg, state):
    return [{
        "artifact_id": "artifact-1",
        "source": tool_cfg.get("name", "researcher"),
        "source_type": "paper",
        "title": "Synthetic artifact",
        "summary": f"Evidence for {direction} with n_results={tool_cfg.get('n_results', 'missing')}",
        "metadata": {"tool_cfg": dict(tool_cfg)},
        "raw": {},
    }]


def second_fake_research_tool(direction, profile, tool_cfg, state):
    return [{
        "artifact_id": "artifact-2",
        "source": tool_cfg.get("name", "researcher"),
        "source_type": "paper",
        "title": "Second synthetic artifact",
        "summary": f"Second evidence for {direction}",
        "metadata": {"tool_cfg": dict(tool_cfg)},
        "raw": {},
    }]


def mixed_research_tool(direction, profile, tool_cfg, state):
    return [
        {
            "artifact_id": "artifact-relevant",
            "source": tool_cfg.get("name", "researcher"),
            "source_type": "paper",
            "title": "Relevant artifact",
            "summary": f"Market regime evidence for {direction}",
            "metadata": {"tool_cfg": dict(tool_cfg)},
            "raw": {},
        },
        {
            "artifact_id": "artifact-irrelevant",
            "source": tool_cfg.get("name", "researcher"),
            "source_type": "paper",
            "title": "Irrelevant artifact",
            "summary": "Completely unrelated robotics content",
            "metadata": {"tool_cfg": dict(tool_cfg)},
            "raw": {},
        },
    ]


def test_role_prompt_can_be_overridden_from_config(minimal_profile):
    cfg = load_brainstorm_config()
    cfg["roles"] = [{
        "name": "facilitator",
        "persona_type": "facilitator",
        "enabled": True,
        "llm_key": "brainstorm",
        "model": "",
        "provider": "",
        "tools": [],
        "research_budget": {},
        "prompt_overrides": {
            "system_template": "SYSTEM {role_name} {current_goal}",
            "human_template": "HUMAN {consensus_summary}",
        },
    }]
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="ok")

    with patch("core.brainstorm.engine.get_llm", return_value=fake_llm):
        turn = engine._run_role_turn(state, cfg["roles"][0], round_index=1)

    assert turn["content"] == "ok"
    messages = fake_llm.invoke.call_args.args[0]
    assert messages[0].content == "SYSTEM facilitator direction"
    assert "Goal: direction" in messages[1].content


def test_run_until_pause_emits_role_lifecycle_callbacks(minimal_profile):
    cfg = {
        "name": "test_brainstorm",
        "roles": [
            {
                "name": "facilitator",
                "persona_type": "facilitator",
                "enabled": True,
                "llm_key": "brainstorm",
                "model": "",
                "provider": "",
                "tools": [],
                "research_budget": {},
            }
        ],
        "stop_policy": {"max_rounds_per_run": 1, "summary_interval_messages": 10, "pause_after_research_round": False},
        "summary": {},
        "execution_handoff": {"default_start_node": "propose_experiments", "allow_direct_to_implement": True},
    }
    engine = BrainstormEngine(minimal_profile, cfg)
    state = create_brainstorm_state(profile_name="test_profile", direction="direction", brainstorm_cfg=cfg)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="ok")
    starts: list[tuple[str, int]] = []
    ends: list[tuple[str, int]] = []

    with patch("core.brainstorm.engine.get_llm", return_value=fake_llm):
        updated = engine.run_until_pause(
            state,
            on_role_start=lambda role, round_index: starts.append((str(role.get("name")), round_index)),
            on_role_end=lambda role, round_index: ends.append((str(role.get("name")), round_index)),
        )

    assert updated["status"] == "awaiting_user"
    assert starts == [("facilitator", 1)]
    assert ends == [("facilitator", 1)]
