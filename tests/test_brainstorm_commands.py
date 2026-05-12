from __future__ import annotations

from core.brainstorm.commands import HELP_TEXT, parse_brainstorm_command


def test_help_text_mentions_exit():
    assert "exit" in HELP_TEXT
    assert "help" in HELP_TEXT


def test_parse_exit_command():
    assert parse_brainstorm_command("exit")["type"] == "exit"


def test_parse_feedback_command_with_raw_text():
    parsed = parse_brainstorm_command("We should ignore the dataset issue for now")
    assert parsed["type"] == "feedback"
    assert "ignore the dataset issue" in parsed["text"]
