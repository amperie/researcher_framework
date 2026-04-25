"""Tests for utils/profile_loader.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from core.utils.profile_loader import (
    get_primary_dataset,
    get_prompt,
    get_step_datasets,
    list_profiles,
    load_profile,
    load_profile_cached,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PROFILE = {
    "name": "test",
    "description": "desc",
    "pipeline": {"steps": ["research", "ideate"]},
    "llm": {"default_model": "claude-test"},
    "prompts": {
        "research": {"system": "You are a researcher.", "score_system": "Score it."},
        "ideate": {"system": "Brainstorm."},
    },
    "datasets": [
        {"name": "ds1", "description": "Dataset 1"},
        {"name": "ds2", "description": "Dataset 2"},
    ],
}


# ---------------------------------------------------------------------------
# load_profile — uses real files on disk
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_loads_neuralsignal_profile(self):
        """The neuralsignal profile must exist and pass validation."""
        profile = load_profile("neuralsignal")
        assert profile["name"] == "neuralsignal"
        assert "pipeline" in profile
        assert "prompts" in profile
        assert "llm" in profile

    def test_profile_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_profile("nonexistent_profile_xyz")

    def test_missing_required_keys_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.dump({"name": "x", "pipeline": {}}), encoding="utf-8")
        with patch("utils.profile_loader._PROFILES_DIR", tmp_path):
            with pytest.raises(ValueError, match="missing required keys"):
                load_profile("bad")

    def test_missing_name_key_raises(self, tmp_path):
        incomplete = {"pipeline": {}, "llm": {}, "prompts": {}}
        f = tmp_path / "incomplete.yaml"
        f.write_text(yaml.dump(incomplete), encoding="utf-8")
        with patch("utils.profile_loader._PROFILES_DIR", tmp_path):
            with pytest.raises(ValueError, match="missing required keys"):
                load_profile("incomplete")

    def test_valid_profile_returns_dict(self, tmp_path):
        f = tmp_path / "valid.yaml"
        f.write_text(yaml.dump(_VALID_PROFILE), encoding="utf-8")
        with patch("utils.profile_loader._PROFILES_DIR", tmp_path):
            profile = load_profile("valid")
        assert profile["name"] == "test"
        assert profile["pipeline"]["steps"] == ["research", "ideate"]


# ---------------------------------------------------------------------------
# load_profile_cached
# ---------------------------------------------------------------------------

class TestLoadProfileCached:
    def test_returns_same_object_on_repeated_calls(self):
        load_profile_cached.cache_clear()
        p1 = load_profile_cached("neuralsignal")
        p2 = load_profile_cached("neuralsignal")
        assert p1 is p2

    def teardown_method(self):
        load_profile_cached.cache_clear()


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------

class TestListProfiles:
    def test_lists_at_least_neuralsignal(self):
        profiles = list_profiles()
        assert "neuralsignal" in profiles

    def test_returns_sorted_list(self):
        profiles = list_profiles()
        assert profiles == sorted(profiles)

    def test_missing_dir_returns_empty(self):
        with patch("utils.profile_loader._PROFILES_DIR", Path("/nonexistent_path_xyz")):
            result = list_profiles()
        assert result == []


# ---------------------------------------------------------------------------
# get_prompt
# ---------------------------------------------------------------------------

class TestGetPrompt:
    def test_returns_system_prompt(self):
        result = get_prompt(_VALID_PROFILE, "research")
        assert result == "You are a researcher."

    def test_returns_custom_key(self):
        result = get_prompt(_VALID_PROFILE, "research", "score_system")
        assert result == "Score it."

    def test_missing_step_raises_keyerror(self):
        with pytest.raises(KeyError, match="No prompts defined for step"):
            get_prompt(_VALID_PROFILE, "nonexistent_step")

    def test_missing_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="No prompt key"):
            get_prompt(_VALID_PROFILE, "research", "nonexistent_key")

    def test_missing_prompts_section_raises(self):
        profile = {**_VALID_PROFILE, "prompts": None}
        with pytest.raises(KeyError):
            get_prompt(profile, "research")


# ---------------------------------------------------------------------------
# get_step_datasets / get_primary_dataset
# ---------------------------------------------------------------------------

class TestGetStepDatasets:
    def test_returns_datasets_list(self):
        result = get_step_datasets(_VALID_PROFILE)
        assert len(result) == 2
        assert result[0]["name"] == "ds1"

    def test_empty_profile_returns_empty(self):
        assert get_step_datasets({}) == []

    def test_none_datasets_returns_empty(self):
        assert get_step_datasets({"datasets": None}) == []


class TestGetPrimaryDataset:
    def test_returns_first_dataset(self):
        result = get_primary_dataset(_VALID_PROFILE)
        assert result["name"] == "ds1"

    def test_empty_datasets_returns_none(self):
        assert get_primary_dataset({}) is None

    def test_empty_list_returns_none(self):
        assert get_primary_dataset({"datasets": []}) is None
