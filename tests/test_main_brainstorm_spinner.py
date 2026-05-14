from __future__ import annotations

import io

import main


def test_brainstorm_spinner_clears_line_on_stop(monkeypatch):
    stream = io.StringIO()
    spinner = main._BrainstormSpinner()
    spinner._stream = stream

    monkeypatch.setattr(main.time, "sleep", lambda *_args, **_kwargs: None)

    spinner.start("facilitator thinking")
    spinner.stop()

    output = stream.getvalue()
    assert "facilitator thinking" in output
    assert "\r" in output


def test_brainstorm_output_colors_role_prefix_only():
    rendered = main._format_brainstorm_cli_output("[facilitator] Keep the scope tight.\n")

    assert "\x1b[" in rendered
    assert "[facilitator]" in rendered
    assert "Keep the scope tight." in rendered


def test_brainstorm_output_leaves_non_role_lines_unchanged():
    rendered = main._format_brainstorm_cli_output("[Plan]\nDirection: test\n")

    assert rendered == "[Plan]\nDirection: test\n"
