"""Tests for plugins/base.py (ResearchAdapter Protocol) and plugins/loader.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.plugins import ResearchAdapter
from core.plugins.loader import adapter_has, load_adapter


# ---------------------------------------------------------------------------
# ResearchAdapter Protocol
# ---------------------------------------------------------------------------

class TestResearchAdapterProtocol:
    def test_protocol_has_required_methods(self):
        required = {
            "validate_environment",
            "build_context",
            "prepare_experiment",
            "execute_experiment",
            "summarize_result",
        }
        annotations = set(ResearchAdapter.__protocol_attrs__)
        assert required.issubset(annotations)

    def test_conforming_class_is_accepted(self):
        class GoodAdapter:
            def validate_environment(self, profile, state=None):
                return {}
            def build_context(self, profile, state):
                return {}
            def prepare_experiment(self, profile, proposal, implementation, state):
                return {}
            def execute_experiment(self, profile, proposal, implementation, artifact, experiment_id, state):
                return {}
            def summarize_result(self, profile, result):
                return {}

        adapter = GoodAdapter()
        assert adapter_has(adapter, "validate_environment")
        assert adapter_has(adapter, "prepare_experiment")
        assert adapter_has(adapter, "execute_experiment")

    def test_non_conforming_object_missing_methods(self):
        class BadAdapter:
            def validate_environment(self, profile, state=None):
                return {}
            # missing other methods

        adapter = BadAdapter()
        assert not adapter_has(adapter, "prepare_experiment")
        assert not adapter_has(adapter, "execute_experiment")


# ---------------------------------------------------------------------------
# load_adapter
# ---------------------------------------------------------------------------

class TestLoadAdapter:
    def test_missing_experiment_adapter_key_raises(self):
        with pytest.raises(ValueError, match="missing experiment_adapter"):
            load_adapter({})

    def test_empty_experiment_adapter_raises(self):
        with pytest.raises(ValueError):
            load_adapter({"experiment_adapter": ""})

    def test_module_with_get_adapter_returns_instance(self):
        mock_adapter = MagicMock()
        mock_module = MagicMock()
        mock_module.get_adapter.return_value = mock_adapter

        with patch("importlib.import_module", return_value=mock_module):
            result = load_adapter({"experiment_adapter": "some.module"})

        assert result is mock_adapter

    def test_module_without_get_adapter_returns_module(self):
        mock_module = MagicMock(spec=[])  # no get_adapter attribute
        mock_module.__spec__ = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            result = load_adapter({"experiment_adapter": "some.module"})

        assert result is mock_module

    def test_import_error_propagates(self):
        with patch("importlib.import_module", side_effect=ImportError("no module")):
            with pytest.raises(ImportError):
                load_adapter({"experiment_adapter": "nonexistent.module"})


# ---------------------------------------------------------------------------
# adapter_has
# ---------------------------------------------------------------------------

class TestAdapterHas:
    def test_has_callable_method(self):
        adapter = MagicMock()
        adapter.prepare_experiment = lambda: None
        assert adapter_has(adapter, "prepare_experiment")

    def test_missing_method_returns_false(self):
        adapter = object()
        assert not adapter_has(adapter, "nonexistent_method")

    def test_non_callable_attribute_returns_false(self):
        adapter = MagicMock(spec=[])
        adapter.my_attr = "not_callable"
        assert not adapter_has(adapter, "my_attr")

    def test_module_level_function(self):
        module = MagicMock()
        module.create_dataset = lambda: None
        assert adapter_has(module, "create_dataset")
