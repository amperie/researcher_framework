"""Tests for llm/factory.py — get_llm model resolution and provider dispatch."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm.factory import get_llm


# ---------------------------------------------------------------------------
# Shared profile fixture
# ---------------------------------------------------------------------------

PROFILE = {
    "name": "test",
    "llm": {
        "default_model": "claude-profile-default",
        "step_overrides": {
            "research": "claude-fast",
        },
    },
}


def _mock_cfg(**kwargs) -> SimpleNamespace:
    defaults = dict(
        llm_provider="anthropic",
        llm_model=None,
        anthropic_api_key="sk-test",
        openai_api_key=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Model resolution priority
# ---------------------------------------------------------------------------

class TestModelResolution:
    def test_explicit_model_arg_wins(self):
        cfg = _mock_cfg(llm_model="global-model")
        mock_anthropic = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                get_llm(step_name="research", profile=PROFILE, model="explicit-model")

        mock_anthropic.assert_called_once()
        assert mock_anthropic.call_args.kwargs["model"] == "explicit-model"

    def test_step_override_wins_over_profile_default(self):
        cfg = _mock_cfg()
        mock_anthropic = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                get_llm(step_name="research", profile=PROFILE)

        assert mock_anthropic.call_args.kwargs["model"] == "claude-fast"

    def test_profile_default_used_when_no_step_override(self):
        cfg = _mock_cfg()
        mock_anthropic = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                get_llm(step_name="ideate", profile=PROFILE)

        assert mock_anthropic.call_args.kwargs["model"] == "claude-profile-default"

    def test_global_config_model_used_when_no_profile(self):
        cfg = _mock_cfg(llm_model="global-override")
        mock_anthropic = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                get_llm()

        assert mock_anthropic.call_args.kwargs["model"] == "global-override"

    def test_provider_default_used_when_all_none(self):
        cfg = _mock_cfg(llm_model=None)
        mock_anthropic = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                get_llm()

        assert mock_anthropic.call_args.kwargs["model"] == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

class TestProviderDispatch:
    def test_anthropic_provider(self):
        cfg = _mock_cfg(llm_provider="anthropic")
        mock_cls = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_cls):
                result = get_llm()

        mock_cls.assert_called_once()

    def test_openai_provider(self):
        cfg = _mock_cfg(llm_provider="openai", openai_api_key="sk-openai")
        mock_cls = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_openai.ChatOpenAI", mock_cls):
                result = get_llm(provider="openai")

        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "gpt-4o"

    def test_unknown_provider_raises(self):
        cfg = _mock_cfg(llm_provider="unknown_provider")

        with patch("llm.factory.get_config", return_value=cfg):
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm(provider="unknown_provider")

    def test_explicit_provider_overrides_config(self):
        cfg = _mock_cfg(llm_provider="anthropic")
        mock_openai = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_openai.ChatOpenAI", mock_openai):
                get_llm(provider="openai")

        mock_openai.assert_called_once()

    def test_api_key_passed_to_anthropic(self):
        cfg = _mock_cfg(anthropic_api_key="sk-mykey")
        mock_cls = MagicMock()

        with patch("llm.factory.get_config", return_value=cfg):
            with patch("langchain_anthropic.ChatAnthropic", mock_cls):
                get_llm()

        assert mock_cls.call_args.kwargs["api_key"] == "sk-mykey"
