"""Tests for NeuralSignal subprocess task wrappers."""
from __future__ import annotations

import sys
from types import ModuleType

from plugins.neuralsignal import tasks


def test_automation_config_merges_payload_over_neuralsignal_defaults(monkeypatch):
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"dataset_row_limit": 100, "seed": 43}
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)

    cfg = tasks._automation_config({"dataset_row_limit": 5, "custom": True})

    assert cfg["dataset_row_limit"] == 5
    assert cfg["seed"] == 43
    assert cfg["custom"] is True


def test_create_dataset_uses_public_automation_api_and_defaults(monkeypatch):
    seen = {}
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"dataset_row_limit": 100, "feature_set_configs": [{"name": "zones"}]}

    def create_dataset(cfg, create_dataset):
        seen["cfg"] = cfg
        seen["create_dataset"] = create_dataset
        return ["features.csv"]

    automation.create_dataset = create_dataset
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)

    result = tasks.create_dataset({"dataset_row_limit": 5})

    assert result == {"file_paths": ["features.csv"]}
    assert seen["create_dataset"] is True
    assert seen["cfg"]["dataset_row_limit"] == 5
    assert seen["cfg"]["feature_set_configs"] == [{"name": "zones"}]


def test_create_dataset_balances_target_values(monkeypatch):
    calls = []
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {
        "dataset_row_limit": 100,
        "row_limit": 100,
        "query": {},
        "feature_set_configs": [{"name": "zones"}],
    }

    def create_dataset(cfg, create_dataset):
        calls.append(dict(cfg))
        return ["features.csv"]

    automation.create_dataset = create_dataset
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)

    result = tasks.create_dataset({
        "dataset_row_limit": 50,
        "query": {"split": "train"},
        "balanced_target": {
            "enabled": True,
            "field": "ground_truth",
            "values": [0, 1],
        },
    })

    assert result["file_paths"] == ["features.csv"]
    assert len(calls) == 2
    assert calls[0]["query"] == {"split": "train", "ground_truth": 0}
    assert calls[0]["dataset_row_limit"] == 25
    assert calls[0]["row_limit"] == 25
    assert calls[0]["overwrite_dataset_file"] is True
    assert calls[0]["write_header"] is True
    assert calls[1]["query"] == {"split": "train", "ground_truth": 1}
    assert calls[1]["dataset_row_limit"] == 25
    assert calls[1]["row_limit"] == 25
    assert calls[1]["overwrite_dataset_file"] is False
    assert calls[1]["write_header"] is False
    assert result["balanced_target"]["pulls"][0]["row_limit"] == 25
    assert result["balanced_target"]["pulls"][1]["row_limit"] == 25


def test_create_dataset_balanced_target_distributes_remainder(monkeypatch):
    limits = []
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"dataset_row_limit": 51, "query": {}}

    def create_dataset(cfg, create_dataset):
        limits.append(cfg["dataset_row_limit"])
        return ["features.csv"]

    automation.create_dataset = create_dataset
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)

    tasks.create_dataset({
        "dataset_row_limit": 51,
        "balanced_target": {"enabled": True, "field": "ground_truth", "values": [0, 1]},
    })

    assert limits == [26, 25]


def test_create_s1_model_uses_public_automation_api_and_normalizes_best_model(monkeypatch):
    class Model:
        def __init__(self, auc):
            self.metrics = {"test_auc": auc}
            self.params = {"auc": auc}
            self.artifacts = {"feature_importance": {"f": auc}}

    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"modeling_row_limits": [0]}
    automation.create_s1_model = lambda cfg: [Model(0.61), Model(0.77)]
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)

    result = tasks.create_s1_model({})

    assert result["metrics"] == {"test_auc": 0.77}
    assert result["params"] == {"auc": 0.77}
    assert result["feature_importance"] == {"f": 0.77}


def test_inject_feature_processor_wraps_structural_class_and_restores_real_base(tmp_path, monkeypatch):
    generated = tmp_path / "GeneratedFeatureSet.py"
    generated.write_text(
        """
import sys
import types

class FeatureSetBase:
    def __init__(self, config):
        self.config = config
    def get_feature_set_name(self):
        raise NotImplementedError
    def process_feature_set(self, scan):
        raise NotImplementedError

_module_path = "neuralsignal.core.modules.feature_sets.feature_set_base"
if _module_path not in sys.modules:
    sys.modules[_module_path] = types.ModuleType(_module_path)
sys.modules[_module_path].FeatureSetBase = FeatureSetBase

class GeneratedFeatureSet(FeatureSetBase):
    def get_feature_set_name(self):
        return "generated_feature_set"
    def process_feature_set(self, scan):
        return (["a"], [1.0])
""",
        encoding="utf-8",
    )

    class RealFeatureSetBase:
        default_config = {"output_format": "name_and_value_columns"}

        def __init__(self, config):
            self.config = {**self.default_config, **config}

        def get_config(self):
            return self.config

    class FeatureProcessor:
        def __init__(self, feature_sets=None, feature_set_configs=None):
            self.feature_sets = feature_sets or []

    base_module = ModuleType("neuralsignal.core.modules.feature_sets.feature_set_base")
    base_module.FeatureSetBase = RealFeatureSetBase
    feature_sets_module = ModuleType("neuralsignal.core.modules.feature_sets")
    feature_sets_module.feature_set_base = base_module
    processor_module = ModuleType("neuralsignal.core.modules.feature_sets.feature_processor")
    processor_module.FeatureProcessor = FeatureProcessor

    monkeypatch.setitem(sys.modules, "neuralsignal.core.modules.feature_sets", feature_sets_module)
    monkeypatch.setitem(sys.modules, "neuralsignal.core.modules.feature_sets.feature_set_base", base_module)
    monkeypatch.setitem(sys.modules, "neuralsignal.core.modules.feature_sets.feature_processor", processor_module)

    cfg = {
        "feature_set_class_path": str(generated),
        "feature_set_class_name": "GeneratedFeatureSet",
        "ffn_layer_patterns": ["mlp"],
    }

    tasks._inject_feature_processor(cfg)

    feature_set = cfg["feature_processor"].feature_sets[0]
    assert isinstance(feature_set, RealFeatureSetBase)
    assert feature_set.get_feature_set_name() == "generated_feature_set"
    assert feature_set.process_feature_set({}) == (["a"], [1.0])
    assert cfg["feature_set_configs"] is None
    assert sys.modules["neuralsignal.core.modules.feature_sets.feature_set_base"].FeatureSetBase is RealFeatureSetBase
