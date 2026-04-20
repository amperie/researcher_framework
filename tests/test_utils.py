"""Tests for utils/utils.py — extract_json_array, extract_json_object,
load_yaml_section, fmt_value."""
from __future__ import annotations

import pytest

from utils.utils import (
    extract_json_array,
    extract_json_object,
    fmt_value,
    load_yaml_section,
)


# ---------------------------------------------------------------------------
# extract_json_array
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_simple_array(self):
        assert extract_json_array('[1, 2, 3]') == [1, 2, 3]

    def test_array_embedded_in_text(self):
        result = extract_json_array('Here is the result: [{"a": 1}] done.')
        assert result == [{"a": 1}]

    def test_nested_array(self):
        result = extract_json_array('[[1, 2], [3, 4]]')
        assert result == [[1, 2], [3, 4]]

    def test_array_with_quoted_brackets(self):
        # Brackets inside strings should not affect depth counting
        result = extract_json_array('[{"key": "val[ue]"}]')
        assert result == [{"key": "val[ue]"}]

    def test_array_with_escaped_quote(self):
        result = extract_json_array('[{"key": "val\\"ue"}]')
        assert result == [{"key": 'val"ue'}]

    def test_markdown_fenced_array(self):
        text = '```json\n[{"name": "test"}]\n```'
        result = extract_json_array(text)
        assert result == [{"name": "test"}]

    def test_no_array_raises(self):
        with pytest.raises(ValueError, match="No JSON array"):
            extract_json_array("no array here")

    def test_unmatched_bracket_raises(self):
        with pytest.raises(ValueError, match="Unmatched"):
            extract_json_array("[1, 2, 3")

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="failed to parse"):
            extract_json_array("[1, 2, ]")

    def test_empty_array(self):
        assert extract_json_array('[]') == []

    def test_array_of_strings(self):
        result = extract_json_array('["a", "b", "c"]')
        assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_simple_object(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_object_embedded_in_text(self):
        result = extract_json_object('Result: {"score": 7, "reason": "good"} end.')
        assert result == {"score": 7, "reason": "good"}

    def test_nested_object(self):
        result = extract_json_object('{"outer": {"inner": 42}}')
        assert result == {"outer": {"inner": 42}}

    def test_object_with_quoted_braces(self):
        result = extract_json_object('{"key": "val{ue}"}')
        assert result == {"key": "val{ue}"}

    def test_object_with_escaped_quote(self):
        result = extract_json_object('{"key": "val\\"ue"}')
        assert result == {"key": 'val"ue'}

    def test_no_object_raises(self):
        with pytest.raises(ValueError, match="No JSON object"):
            extract_json_object("no object here")

    def test_unmatched_brace_raises(self):
        with pytest.raises(ValueError, match="Unmatched"):
            extract_json_object('{"a": 1')

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="failed to parse"):
            extract_json_object('{"a": ,}')

    def test_empty_object(self):
        assert extract_json_object('{}') == {}


# ---------------------------------------------------------------------------
# load_yaml_section
# ---------------------------------------------------------------------------

class TestLoadYamlSection:
    def test_loads_section_from_real_config(self):
        # The real config.yaml exists in the repo; at minimum we can call it
        result = load_yaml_section("logging")
        assert isinstance(result, dict)

    def test_missing_section_returns_empty(self):
        result = load_yaml_section("nonexistent_section_xyz")
        assert result == {}

    def test_missing_file_returns_empty(self, capsys):
        result = load_yaml_section("anything", config_path="nonexistent_file.yaml")
        assert result == {}
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_malformed_yaml_returns_empty(self, tmp_path, capsys):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [\nunot_closed", encoding="utf-8")
        result = load_yaml_section("key", config_path=str(bad_yaml))
        assert result == {}
        captured = capsys.readouterr()
        assert "Failed to parse" in captured.err

    def test_valid_yaml_section(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("mykey:\n  val: 42\n", encoding="utf-8")
        result = load_yaml_section("mykey", config_path=str(cfg))
        assert result == {"val": 42}


# ---------------------------------------------------------------------------
# fmt_value
# ---------------------------------------------------------------------------

class TestFmtValue:
    def test_string(self):
        assert fmt_value("hello") == "5 chars"

    def test_empty_string(self):
        assert fmt_value("") == "0 chars"

    def test_list(self):
        assert fmt_value([1, 2, 3]) == "3 items"

    def test_empty_list(self):
        assert fmt_value([]) == "0 items"

    def test_dict(self):
        assert fmt_value({"a": 1, "b": 2}) == "2 keys"

    def test_empty_dict(self):
        assert fmt_value({}) == "0 keys"

    def test_int(self):
        result = fmt_value(42)
        assert result == repr(42)

    def test_none(self):
        result = fmt_value(None)
        assert result == repr(None)

    def test_bool(self):
        result = fmt_value(True)
        assert result == repr(True)
