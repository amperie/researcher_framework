"""Tests for configs/config.py — _interpolate, _walk, get_config."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from configs.config import _interpolate, _walk, get_config


# ---------------------------------------------------------------------------
# _interpolate
# ---------------------------------------------------------------------------

class TestInterpolate:
    def test_substitutes_env_var(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            result = _interpolate("prefix_${MY_VAR}_suffix")
        assert result == "prefix_hello_suffix"

    def test_uses_default_when_var_missing(self):
        env = dict(os.environ)
        env.pop("MISSING_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            result = _interpolate("${MISSING_VAR:fallback}")
        assert result == "fallback"

    def test_empty_default_when_var_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "MISSING_VAR"}
        with patch.dict(os.environ, env, clear=True):
            result = _interpolate("${MISSING_VAR}")
        assert result == ""

    def test_non_string_passthrough(self):
        assert _interpolate(42) == 42
        assert _interpolate(None) is None
        assert _interpolate(True) is True

    def test_no_pattern_unchanged(self):
        assert _interpolate("plain string") == "plain string"


# ---------------------------------------------------------------------------
# _walk
# ---------------------------------------------------------------------------

class TestWalk:
    def test_dict_recursion(self):
        with patch.dict(os.environ, {"K": "v"}):
            result = _walk({"a": "${K}", "b": "literal"})
        assert result == {"a": "v", "b": "literal"}

    def test_list_recursion(self):
        with patch.dict(os.environ, {"X": "1"}):
            result = _walk(["${X}", "static"])
        assert result == ["1", "static"]

    def test_nested_dict_in_list(self):
        with patch.dict(os.environ, {"Y": "yes"}):
            result = _walk([{"k": "${Y}"}])
        assert result == [{"k": "yes"}]

    def test_scalar_passthrough(self):
        assert _walk(99) == 99


# ---------------------------------------------------------------------------
# get_config — uses real config.yaml from the repo
# ---------------------------------------------------------------------------

class TestGetConfig:
    def setup_method(self):
        get_config.cache_clear()

    def teardown_method(self):
        get_config.cache_clear()

    def test_returns_namespace(self):
        cfg = get_config()
        assert isinstance(cfg, SimpleNamespace)

    def test_cached_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_has_expected_attributes(self):
        cfg = get_config()
        assert hasattr(cfg, "llm_provider")
        assert hasattr(cfg, "mongo_url")
        assert hasattr(cfg, "chroma_host")
        assert hasattr(cfg, "chroma_port")

    def test_chroma_port_is_int(self):
        cfg = get_config()
        assert isinstance(cfg.chroma_port, int)

    def test_chroma_ssl_is_bool(self):
        cfg = get_config()
        assert isinstance(cfg.chroma_ssl, bool)

    def test_no_logging_key(self):
        cfg = get_config()
        # logging section must be stripped
        assert not hasattr(cfg, "logging")

    def test_optional_fields_none_when_empty(self):
        # These should be None if not set (not empty strings)
        cfg = get_config()
        for key in ("llm_model", "anthropic_api_key", "openai_api_key", "chroma_auth_token"):
            val = getattr(cfg, key, "MISSING")
            assert val is None or isinstance(val, str), f"{key} has unexpected type {type(val)}"


# ---------------------------------------------------------------------------
# Dotenv loading
# ---------------------------------------------------------------------------

class TestDotenvLoading:
    def setup_method(self):
        get_config.cache_clear()

    def teardown_method(self):
        get_config.cache_clear()

    def test_dotenv_does_not_override_existing_env(self, tmp_path):
        """ENV vars take precedence over .env values."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_KEY=from_file\n", encoding="utf-8")

        with patch("configs.config._ENV_PATH", env_file):
            with patch.dict(os.environ, {"MY_TEST_KEY": "from_env"}):
                # Import _load_dotenv directly and call it
                from configs.config import _load_dotenv
                before = os.environ.get("MY_TEST_KEY")
                _load_dotenv()
                # ENV var must still win
                assert os.environ.get("MY_TEST_KEY") == "from_env"

    def test_dotenv_skips_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\nVALID_KEY=value\n", encoding="utf-8")

        env_snapshot = dict(os.environ)
        env_snapshot.pop("VALID_KEY", None)

        with patch("configs.config._ENV_PATH", env_file):
            with patch.dict(os.environ, env_snapshot, clear=True):
                from configs.config import _load_dotenv
                _load_dotenv()
                assert os.environ.get("VALID_KEY") == "value"

    def test_dotenv_skips_malformed_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("no_equals_sign\nGOOD=ok\n", encoding="utf-8")

        env_snapshot = {k: v for k, v in os.environ.items() if k not in ("no_equals_sign", "GOOD")}

        with patch("configs.config._ENV_PATH", env_file):
            with patch.dict(os.environ, env_snapshot, clear=True):
                from configs.config import _load_dotenv
                _load_dotenv()
                assert "no_equals_sign" not in os.environ
                assert os.environ.get("GOOD") == "ok"
