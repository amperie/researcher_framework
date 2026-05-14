from __future__ import annotations

from pathlib import Path

from core.brainstorm import config as brainstorm_config
from core.brainstorm.config import load_brainstorm_config
from core.tools import research_tool_catalog


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


def test_brainstorm_config_normalizes_prompt_overrides(tmp_path: Path):
    path = tmp_path / "brainstorm.yaml"
    path.write_text(
        """
name: test
roles:
  - name: facilitator
    persona_type: facilitator
    prompt_overrides:
      system_template: "Role {role_name}"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_brainstorm_config(str(path))

    assert cfg["roles"][0]["prompt_overrides"]["system_template"] == "Role {role_name}"
    assert "role_turn" in cfg["prompts"]


def test_brainstorm_config_normalizes_tool_objects_and_legacy_strings(tmp_path: Path):
    path = tmp_path / "brainstorm.yaml"
    path.write_text(
        """
name: test
roles:
  - name: researcher
    persona_type: researcher
    tools:
      - core.tools.research_tools.collect_memory
      - path: core.tools.research_tools.collect_profile_context
        name: profile_context
        include:
          - datasets
""".strip(),
        encoding="utf-8",
    )

    cfg = load_brainstorm_config(str(path))

    assert cfg["roles"][0]["tools"][0]["path"] == "core.tools.research_tools.collect_memory"
    assert cfg["roles"][0]["tools"][0]["name"] == "collect_memory"
    assert cfg["roles"][0]["tools"][1]["name"] == "profile_context"
    assert cfg["roles"][0]["tools"][1]["include"] == ["datasets"]


def test_resolve_brainstorm_config_path_falls_back_from_legacy_default(monkeypatch, tmp_path: Path):
    legacy_default = tmp_path / "default.brainstorm.yaml"
    trading_default = tmp_path / "default.trading.brainstorm.yaml"
    trading_default.write_text("name: trading\nroles: [{name: facilitator, persona_type: facilitator}]\n", encoding="utf-8")

    monkeypatch.setattr(brainstorm_config, "DEFAULT_BRAINSTORM_CONFIG", legacy_default)
    monkeypatch.setattr(brainstorm_config, "BRAINSTORM_CONFIG_DIR", tmp_path)

    resolved = brainstorm_config.resolve_brainstorm_config_path(str(legacy_default))

    assert resolved == trading_default


def test_brainstorm_config_resolves_tool_refs(monkeypatch, tmp_path: Path):
    catalog_dir = tmp_path / "research_tools"
    catalog_dir.mkdir()
    (catalog_dir / "catalog.yaml").write_text(
        """
tools:
  shared_memory:
    path: core.tools.research_tools.collect_memory
    name: memory
    n_results: 7
""".strip(),
        encoding="utf-8",
    )
    path = tmp_path / "brainstorm.yaml"
    path.write_text(
        """
name: test
roles:
  - name: researcher
    persona_type: researcher
    tools:
      - ref: shared_memory
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(research_tool_catalog, "RESEARCH_TOOL_CATALOG_DIR", catalog_dir)

    cfg = load_brainstorm_config(str(path))

    assert cfg["roles"][0]["tools"][0]["path"] == "core.tools.research_tools.collect_memory"
    assert cfg["roles"][0]["tools"][0]["n_results"] == 7
