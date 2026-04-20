"""Unit tests for DatasetManager using mongomock (no live MongoDB required)."""
import gridfs
import mongomock
import mongomock.gridfs
import pytest

# Patch gridfs to accept mongomock databases for the duration of the test session.
mongomock.gridfs.enable_gridfs_integration()

from utils.dataset_manager import DatasetManager


@pytest.fixture
def dm():
    client = mongomock.MongoClient()
    mgr = DatasetManager.__new__(DatasetManager)
    mgr._client = client
    mgr._db = client["test_datasets"]
    mgr._registry = mgr._db["datasets"]
    mgr._fs = gridfs.GridFS(mgr._db)
    return mgr


_BASE_CONFIG = {"model": "gpt2", "num_layers": 12, "hidden_dim": 768}
_COLUMNS = ["feat_a", "feat_b", "feat_c"]
_VALUES = [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 1. config_hash stability
# ---------------------------------------------------------------------------
def test_config_hash_stable(dm):
    h1 = dm.config_hash({"b": 2, "a": 1})
    h2 = dm.config_hash({"a": 1, "b": 2})
    assert h1 == h2, "hash must be key-order-independent"


# ---------------------------------------------------------------------------
# 2. Volatile keys excluded from hash
# ---------------------------------------------------------------------------
def test_config_hash_volatile_keys_excluded(dm):
    base = {"model": "gpt2"}
    with_volatile = {"model": "gpt2", "experiment_id": "uuid-abc", "script_path": "/tmp/x.py"}
    assert dm.config_hash(base) == dm.config_hash(with_volatile)


# ---------------------------------------------------------------------------
# 3. Different non-volatile value → different hash
# ---------------------------------------------------------------------------
def test_config_hash_differs(dm):
    h1 = dm.config_hash({"model": "gpt2"})
    h2 = dm.config_hash({"model": "gpt-4"})
    assert h1 != h2


# ---------------------------------------------------------------------------
# 4. Save then find_cached — cache hit
# ---------------------------------------------------------------------------
def test_save_and_find_cached(dm):
    dm.save_dataset("exp-1", "my_features", _BASE_CONFIG, _COLUMNS, _VALUES)
    cached = dm.find_cached("my_features", _BASE_CONFIG)
    assert cached is not None
    assert cached["feature_set_name"] == "my_features"
    assert cached["column_names"] == _COLUMNS


# ---------------------------------------------------------------------------
# 5. Different feature_set_name → cache miss
# ---------------------------------------------------------------------------
def test_find_cached_miss(dm):
    dm.save_dataset("exp-1", "my_features", _BASE_CONFIG, _COLUMNS, _VALUES)
    assert dm.find_cached("other_features", _BASE_CONFIG) is None


# ---------------------------------------------------------------------------
# 6. GridFS file deleted → find_cached returns None
# ---------------------------------------------------------------------------
def test_find_cached_missing_gridfs(dm):
    import bson
    entry = dm.save_dataset("exp-1", "my_features", _BASE_CONFIG, _COLUMNS, _VALUES)
    # Delete the GridFS file directly
    dm._fs.delete(bson.ObjectId(entry["gridfs_id"]))
    assert dm.find_cached("my_features", _BASE_CONFIG) is None


# ---------------------------------------------------------------------------
# 7. Single row → DataFrame shape (1, N)
# ---------------------------------------------------------------------------
def test_save_single_row(dm):
    entry = dm.save_dataset("exp-1", "my_features", _BASE_CONFIG, _COLUMNS, _VALUES)
    df = dm.load_dataset(entry)
    assert df.shape == (1, len(_COLUMNS))
    assert list(df.columns) == _COLUMNS


# ---------------------------------------------------------------------------
# 8. Multi-row → correct shape
# ---------------------------------------------------------------------------
def test_save_multi_row(dm):
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    entry = dm.save_dataset("exp-1", "my_features", _BASE_CONFIG, _COLUMNS, rows)
    df = dm.load_dataset(entry)
    assert df.shape == (3, len(_COLUMNS))


# ---------------------------------------------------------------------------
# 9. meld two datasets — combined columns, same index length
# ---------------------------------------------------------------------------
def test_meld_two_datasets(dm):
    cols_a = ["x", "y"]
    cols_b = ["p", "q"]
    scan_ids = ["scan-1"]
    entry_a = dm.save_dataset("exp-1", "feat_a", _BASE_CONFIG, cols_a, [0.1, 0.2], scan_ids)
    entry_b = dm.save_dataset("exp-2", "feat_b", _BASE_CONFIG, cols_b, [0.3, 0.4], scan_ids)
    merged = dm.meld([entry_a, entry_b])
    assert "feat_a__x" in merged.columns
    assert "feat_b__p" in merged.columns
    assert len(merged) == 1


# ---------------------------------------------------------------------------
# 10. meld_by_config_hash matches meld()
# ---------------------------------------------------------------------------
def test_meld_by_config_hash(dm):
    cols_a = ["x", "y"]
    cols_b = ["p", "q"]
    scan_ids = ["scan-1"]
    entry_a = dm.save_dataset("exp-1", "feat_a", _BASE_CONFIG, cols_a, [0.1, 0.2], scan_ids)
    entry_b = dm.save_dataset("exp-2", "feat_b", _BASE_CONFIG, cols_b, [0.3, 0.4], scan_ids)

    h = dm.config_hash(_BASE_CONFIG)
    merged_by_hash = dm.meld_by_config_hash(h)
    merged_direct = dm.meld([entry_a, entry_b])

    assert set(merged_by_hash.columns) == set(merged_direct.columns)
    assert len(merged_by_hash) == len(merged_direct)


# ---------------------------------------------------------------------------
# 11. build_cached_result — all keys expected by experiment_runner_node
# ---------------------------------------------------------------------------
def test_build_cached_result_shape(dm):
    entry = dm.save_dataset("exp-1", "my_features", _BASE_CONFIG, _COLUMNS, _VALUES)
    script = {"proposal_name": "test_proposal", "experiment_config": _BASE_CONFIG}
    result = dm.build_cached_result(entry, script)

    required_keys = {
        "experiment_id", "proposal_name", "stdout", "stderr",
        "success", "raw_results", "dataset_id", "from_cache",
    }
    assert required_keys.issubset(result.keys())
    assert result["success"] is True
    assert result["from_cache"] is True
    assert result["proposal_name"] == "test_proposal"
    assert result["raw_results"]["from_cache"] is True
