"""Tests for NeuralSignal subprocess task wrappers."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from core.plugins.neuralsignal import tasks


def test_automation_config_merges_payload_over_neuralsignal_defaults(monkeypatch):
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"dataset_row_limit": 100, "seed": 43}
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)

    cfg = tasks._automation_config({"dataset_row_limit": 5, "custom": True})

    assert cfg["dataset_row_limit"] == 5
    assert cfg["seed"] == 43
    assert cfg["custom"] is True


def test_automation_config_deep_merges_backend_config(monkeypatch):
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {
        "backend_config": {
            "backend_type": "neuralsignal_v1",
            "mongo_url": "mongodb://default",
            "scan_cache_size": 1000,
            "scan_hd_cache_size": 2000,
            "scan_cache_directory": "E:/cache",
            "cache_scan_on_load": True,
        }
    }
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)

    cfg = tasks._automation_config({"backend_config": {"mongo_url": "mongodb://override"}})

    assert cfg["backend_config"]["mongo_url"] == "mongodb://override"
    assert cfg["backend_config"]["scan_cache_size"] == 1000
    assert cfg["backend_config"]["scan_hd_cache_size"] == 2000
    assert cfg["backend_config"]["scan_cache_directory"] == "E:/cache"
    assert cfg["backend_config"]["cache_scan_on_load"] is True


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
    monkeypatch.setattr(tasks, "_enable_mongo_no_cursor_timeout", lambda: None)

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
    monkeypatch.setattr(tasks, "_enable_mongo_no_cursor_timeout", lambda: None)

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
    monkeypatch.setattr(tasks, "_enable_mongo_no_cursor_timeout", lambda: None)

    tasks.create_dataset({
        "dataset_row_limit": 51,
        "balanced_target": {"enabled": True, "field": "ground_truth", "values": [0, 1]},
    })

    assert limits == [26, 25]


def test_create_dataset_moves_output_to_dataset_dir(tmp_path, monkeypatch):
    source = tmp_path / "features.csv"
    source.write_text("target,a\n1,2\n", encoding="utf-8")
    output_dir = tmp_path / "dev" / "experiments" / "neuralsignal" / "datasets"

    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"dataset_row_limit": 1, "query": {}}
    automation.create_dataset = lambda cfg, create_dataset: [str(source)]
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)
    monkeypatch.setattr(tasks, "_enable_mongo_no_cursor_timeout", lambda: None)

    result = tasks.create_dataset({"dataset_output_dir": str(output_dir)})

    expected = output_dir / "features.csv"
    assert result["file_paths"] == [str(expected)]
    assert expected.read_text(encoding="utf-8") == "target,a\n1,2\n"
    assert not source.exists()


def test_create_dataset_recovers_partial_file_on_cursor_loss(tmp_path, monkeypatch):
    class CursorNotFound(Exception):
        pass

    output_dir = tmp_path / "datasets"
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {
        "file_out": "partial.csv",
        "dataset_output_dir": str(output_dir),
        "balanced_target": {"values": [0, 1]},
    }

    def create_dataset(cfg, create_dataset):
        (tmp_path / "partial.csv").write_text("target,feature\n1,0.2\n", encoding="utf-8")
        raise CursorNotFound("cursor died")

    automation.create_dataset = create_dataset
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)
    monkeypatch.setattr(tasks, "_enable_mongo_no_cursor_timeout", lambda: None)
    monkeypatch.chdir(tmp_path)

    result = tasks.create_dataset({})

    expected = output_dir / "partial.csv"
    assert result["partial"] is True
    assert result["partial_rows"] == 1
    assert result["target_counts"] == {"1": 1}
    assert result["usable_for_model"] is False
    assert result["unusable_reason"] == "Partial dataset is missing target classes: 0"
    assert result["file_paths"] == [str(expected)]
    assert expected.read_text(encoding="utf-8") == "target,feature\n1,0.2\n"


def test_balanced_dataset_continues_next_class_after_strong_partial(tmp_path, monkeypatch):
    class CursorNotFound(Exception):
        pass

    output_dir = tmp_path / "datasets"
    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {
        "dataset_row_limit": 200,
        "row_limit": 200,
        "query": {},
        "file_out": "balanced.csv",
        "dataset_output_dir": str(output_dir),
    }

    def create_dataset(cfg, create_dataset):
        path = tmp_path / cfg["file_out"] if not Path(cfg["file_out"]).is_absolute() else Path(cfg["file_out"])
        if cfg["query"]["ground_truth"] == 0:
            path.write_text("target,feature\n" + ("0,0.1\n" * 100), encoding="utf-8")
            raise CursorNotFound("cursor died")
        with path.open("a", encoding="utf-8") as fh:
            fh.write("1,0.2\n" * 100)
        return [str(path)]

    automation.create_dataset = create_dataset
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)
    monkeypatch.setattr(tasks, "_enable_mongo_no_cursor_timeout", lambda: None)
    monkeypatch.chdir(tmp_path)

    result = tasks.create_dataset({
        "balanced_target": {"enabled": True, "field": "ground_truth", "values": [0, 1]},
    })

    expected = output_dir / "balanced.csv"
    assert result["file_paths"] == [str(expected)]
    assert result["target_counts"] == {"0": 100, "1": 100}
    assert result["usable_for_model"] is True


def test_run_proposal_branch_skips_model_for_single_class_partial(monkeypatch, tmp_path):
    dataset_path = tmp_path / "partial.csv"
    dataset_path.write_text("target,feature\n0,0.1\n", encoding="utf-8")
    monkeypatch.setattr(tasks, "create_dataset", lambda payload: {
        "file_paths": [str(dataset_path)],
        "partial": True,
        "usable_for_model": False,
        "unusable_reason": "Partial dataset is missing target classes: 1",
    })
    monkeypatch.setattr(tasks, "create_s1_model", lambda payload: {"metrics": {"test_auc": 1.0}})

    result = tasks.run_proposal_branch({
        "proposal_name": "p",
        "dataset_config": {},
        "model_config_base": {},
    })

    assert result["dataset_result"]["file_paths"] == [str(dataset_path)]
    assert result["model_result"]["skipped_model_training"] is True
    assert result["model_result"]["error"] == "Partial dataset is missing target classes: 1"


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
    assert result["artifacts"] == {"feature_importance": {"f": 0.77}}
    assert result["model_config"] == {"description": None, "model": None, "tags": None}
    assert result["figure_paths"] == {}


def test_run_proposal_branch_reuses_existing_dataset_and_trains_model(monkeypatch, tmp_path):
    dataset_path = tmp_path / "features.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
    seen = {}

    monkeypatch.setattr(tasks, "create_dataset", lambda payload: {"file_paths": [str(tmp_path / "unexpected.csv")]})

    def _create_model(payload):
        seen["payload"] = dict(payload)
        return {"metrics": {"test_auc": 0.77}, "params": {}, "feature_importance": {}, "artifacts": {}, "model_config": {}, "figure_paths": {}}

    monkeypatch.setattr(tasks, "create_s1_model", _create_model)

    result = tasks.run_proposal_branch({
        "proposal_name": "activation_sparsity",
        "experiment_id": "exp-1",
        "dataset_config": {"dataset": "HaluBench"},
        "model_config_base": {"dataset": "HaluBench"},
        "reused_dataset_artifact": {"dataset_path": str(dataset_path), "memory_record_id": "dataset:abc"},
    })

    assert result["proposal_name"] == "activation_sparsity"
    assert result["experiment_id"] == "exp-1"
    assert result["dataset_result"]["reused_from_memory"] is True
    assert seen["payload"]["dataset_path"] == str(dataset_path)
    assert seen["payload"]["file_out"] == "features.csv"


def test_enable_mongo_no_cursor_timeout_patches_query(monkeypatch):
    backend_module = ModuleType("neuralsignal.backend.mongo_backend")

    class Collection:
        def __init__(self):
            self.calls = []

        def find(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return ["ok"]

    class MongoBackend:
        def __init__(self):
            self.col = Collection()

    backend_module.MongoBackend = MongoBackend
    monkeypatch.setitem(sys.modules, "neuralsignal.backend.mongo_backend", backend_module)

    tasks._enable_mongo_no_cursor_timeout()
    backend = MongoBackend()
    result = backend.query({"x": 1})

    assert result == ["ok"]
    assert backend.col.calls == [({"x": 1}, {"no_cursor_timeout": True})]


def test_create_s1_model_supports_s1model_config_shape(monkeypatch):
    class Model:
        def __init__(self, auc):
            self.config = {
                "metrics": {"test_auc": auc},
                "params": {"auc": auc},
                "artifacts": {"feature_importance": {"f": auc}},
            }

    automation = ModuleType("neuralsignal.automation")
    automation.get_config = lambda: {"modeling_row_limits": [0]}
    automation.create_s1_model = lambda cfg: [Model(0.61), Model(0.77)]
    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)

    result = tasks.create_s1_model({})

    assert result["metrics"] == {"test_auc": 0.77}
    assert result["params"] == {"auc": 0.77}
    assert result["feature_importance"] == {"f": 0.77}
    assert result["artifacts"] == {"feature_importance": {"f": 0.77}}
    assert result["model_config"] == {"description": None, "model": None, "tags": None}
    assert result["figure_paths"] == {}


def test_create_s1_model_returns_config_and_existing_figure_paths(monkeypatch, tmp_path):
    figure_path = tmp_path / "confusion_matrix.png"
    figure_path.write_text("fake image", encoding="utf-8")
    automation = type("Automation", (), {})()
    automation.get_config = lambda: {"seed": 42}

    class Model:
        def __init__(self):
            self.metrics = {"test_auc": 0.81}
            self.params = {"max_depth": 4}
            self.artifacts = {"feature_importance": {"f": 0.81}}
            self.config = {
                "description": "model description",
                "model": "xgboost",
                "tags": ["neuralsignal", "hallucination"],
                "figures": {"confusion_matrix": str(figure_path)},
            }

    automation.create_s1_model = lambda cfg: [Model()]

    monkeypatch.setitem(sys.modules, "neuralsignal.automation", automation)
    monkeypatch.setattr(tasks, "_inject_feature_processor", lambda cfg: None)

    result = tasks.create_s1_model({})

    assert result["model_config"] == {
        "description": "model description",
        "model": "xgboost",
        "tags": ["neuralsignal", "hallucination"],
    }
    assert result["figure_paths"] == {"confusion_matrix": str(figure_path.resolve())}


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
    assert isinstance(feature_set.feature_set, RealFeatureSetBase)
    assert feature_set.get_feature_set_name() == "generated_feature_set"
    assert feature_set.process_feature_set({}) == (["a"], [1.0])
    assert cfg["feature_set_configs"] is None
    assert sys.modules["neuralsignal.core.modules.feature_sets.feature_set_base"].FeatureSetBase is RealFeatureSetBase


def test_scan_shape_wrapper_retries_flat_outputs_as_pass_list():
    class GeneratedFeatureSet:
        def __init__(self):
            self.config = {"output_format": "name_and_value_columns"}

        def get_feature_set_name(self):
            return "generated"

        def process_feature_set(self, scan):
            outputs = scan["outputs"]
            layer_order = scan["layer_order"]
            layer_id = layer_order[0]
            # This mimics generated code that assumes outputs[0][layer_id].
            activation = outputs[0][layer_id]
            return (["feature"], [float(activation)])

    feature_set = tasks._ScanShapeCompatibleFeatureSet(GeneratedFeatureSet())
    result = feature_set.process_feature_set({
        "outputs": {"layer_a": 3.0},
        "layer_order": ["layer_a"],
    })

    assert result == (["feature"], [3.0])


def test_scan_shape_wrapper_raises_for_target_only_features():
    class EmptyFeatureSet:
        def __init__(self):
            self.config = {"output_format": "name_and_value_columns"}

        def get_feature_set_name(self):
            return "empty"

        def process_feature_set(self, scan):
            return ([], [])

    feature_set = tasks._ScanShapeCompatibleFeatureSet(EmptyFeatureSet())

    try:
        feature_set.process_feature_set({"outputs": {"layer_a": 3.0}})
    except RuntimeError as exc:
        assert "returned no feature columns" in str(exc)
    else:
        raise AssertionError("Expected empty feature set to fail loudly")
