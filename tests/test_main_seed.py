from __future__ import annotations

from contextlib import contextmanager
from argparse import Namespace
from types import SimpleNamespace

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


def test_parse_args_accepts_config_alias():
    args = main.parse_args(["--mode", "brainstorm", "--profile", "trading", "--config", "configs/brainstorm/x.yaml"])

    assert args.brainstorm_config == "configs/brainstorm/x.yaml"


def test_parse_args_help_command_exits_with_zero(capsys):
    try:
        main.parse_args(["help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected SystemExit for help command")

    captured = capsys.readouterr()
    assert "Configuration-driven research pipeline and interactive brainstorm runner." in captured.out
    assert "Brainstorm mode:" in captured.out


def test_load_brainstorm_config_for_cli_prompts_for_selection(monkeypatch):
    configs = ["configs/brainstorm/a.yaml", "configs/brainstorm/b.yaml"]
    captured_paths: list[str] = []

    def _load(path):
        captured_paths.append(path)
        return {"path": path}

    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    cfg = main._load_brainstorm_config_for_cli(
        profile_name="trading",
        path=None,
        load_brainstorm_config_fn=_load,
        list_brainstorm_configs_fn=lambda: configs,
        error_type=ValueError,
    )

    assert captured_paths == ["configs/brainstorm/b.yaml"]
    assert cfg["path"] == "configs/brainstorm/b.yaml"


def test_default_brainstorm_config_index_prefers_profile_named_yaml():
    index = main._default_brainstorm_config_index(
        profile_name="trading",
        config_paths=[
            "configs/brainstorm/default.neuralsignal.brainstorm.yaml",
            "configs/brainstorm/default.trading.brainstorm.yaml",
        ],
    )

    assert index == 2


def test_main_help_command_prints_help_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(main, "run_periodic_dev_cleanup", lambda: SimpleNamespace(skipped=True, deleted_files=[], deleted_dirs=[], errors=[]))
    monkeypatch.setattr(main.sys, "argv", ["main.py", "help"])

    try:
        main.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected SystemExit for help command")

    captured = capsys.readouterr()
    assert "Examples:" in captured.out
    assert "--config" in captured.out


def test_run_brainstorm_mode_restores_console_logging_before_execute(monkeypatch):
    import core.brainstorm as brainstorm

    activity_log: list[tuple[str, object]] = []

    @contextmanager
    def _fake_raise_console_log_level(level):
        activity_log.append(("enter", level))
        try:
            yield
        finally:
            activity_log.append(("exit", level))

    class _FakeEngine:
        def __init__(self, _profile, _cfg):
            pass

        def run_until_pause(self, state, **_kwargs):
            activity_log.append(("run_until_pause", state.get("status")))
            state["status"] = "awaiting_user"
            return state

        def apply_command(self, state, command, **_kwargs):
            activity_log.append(("apply_command", command))
            state["status"] = "approved_for_execution"
            return state

    def _fake_execute_handoff(*args, **kwargs):
        active_contexts = sum(1 for kind, _ in activity_log if kind == "enter") - sum(1 for kind, _ in activity_log if kind == "exit")
        activity_log.append(("execute_handoff_active_contexts", active_contexts))
        return "implement", {}

    monkeypatch.setattr(main, "temporarily_raise_console_log_level", _fake_raise_console_log_level)
    monkeypatch.setattr(brainstorm, "BrainstormEngine", _FakeEngine)
    monkeypatch.setattr(brainstorm, "load_brainstorm_config", lambda _path=None: {"name": "default"})
    monkeypatch.setattr(brainstorm, "list_brainstorm_configs", lambda: [])
    monkeypatch.setattr(brainstorm, "persist_brainstorm_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brainstorm, "execute_brainstorm_handoff", _fake_execute_handoff)
    monkeypatch.setattr(brainstorm, "resolve_brainstorm_seed", lambda *_args, **_kwargs: {"research_direction": "seeded direction"})
    monkeypatch.setattr(main, "build_initial_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(main, "run_pipeline_graph", lambda *args, **kwargs: {})
    monkeypatch.setattr("builtins.input", lambda prompt="": "execute" if "brainstorm>" in prompt else "")

    main._run_brainstorm_mode(
        Namespace(
            brainstorm_config=None,
            resume_brainstorm=None,
            source_experiment=None,
            handoff=None,
            proposal_seed=None,
            next_step=None,
            direction="seeded direction",
        ),
        "trading",
        {"name": "trading"},
    )

    assert ("execute_handoff_active_contexts", 0) in activity_log
