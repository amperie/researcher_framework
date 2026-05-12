from __future__ import annotations

from pathlib import Path

from core.brainstorm.config import load_brainstorm_config


def test_brainstorm_config_applies_default_model_when_role_omits_model(tmp_path: Path):
    path = tmp_path / "brainstorm.yaml"
    path.write_text(
        """
name: test
llm_defaults:
  provider: anthropic
  model: claude-sonnet-4
roles:
  - name: facilitator
    persona_type: facilitator
    enabled: true
  - name: skeptic
    persona_type: skeptic
    enabled: true
    model: claude-opus-4
""".strip(),
        encoding="utf-8",
    )

    cfg = load_brainstorm_config(str(path))

    assert cfg["roles"][0]["model"] == "claude-sonnet-4"
    assert cfg["roles"][1]["model"] == "claude-opus-4"
    assert cfg["roles"][0]["provider"] == "anthropic"
