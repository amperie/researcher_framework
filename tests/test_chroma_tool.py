"""Tests for tools/chroma_tool.py — ChromaStore."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.tools.chroma_tool import ChromaStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_cfg(**kwargs):
    defaults = dict(
        chroma_host="localhost",
        chroma_port=8000,
        chroma_ssl=False,
        chroma_auth_token=None,
        chroma_collection="test_col",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_store(collection_name="test_col", cfg_kwargs=None) -> tuple[ChromaStore, MagicMock, MagicMock]:
    """Return (store, mock_client, mock_collection)."""
    cfg = _mock_cfg(**(cfg_kwargs or {}))
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch("core.tools.chroma_tool.get_config", return_value=cfg):
        with patch("core.tools.chroma_tool._build_client", return_value=mock_client):
            store = ChromaStore(collection_name=collection_name)
            # Force lazy init
            store._get_collection()

    return store, mock_client, mock_collection


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestChromaStoreInit:
    def test_uses_explicit_collection_name(self):
        cfg = _mock_cfg()
        with patch("core.tools.chroma_tool.get_config", return_value=cfg):
            store = ChromaStore(collection_name="my_col")
        assert store._collection_name == "my_col"

    def test_defaults_to_config_collection(self):
        cfg = _mock_cfg(chroma_collection="cfg_col")
        with patch("core.tools.chroma_tool.get_config", return_value=cfg):
            store = ChromaStore()
        assert store._collection_name == "cfg_col"

    def test_client_is_none_before_first_use(self):
        cfg = _mock_cfg()
        with patch("core.tools.chroma_tool.get_config", return_value=cfg):
            store = ChromaStore()
        assert store._client is None
        assert store._collection is None


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_calls_collection(self):
        store, _, mock_col = _make_store()
        store.upsert("id1", "doc text", {"key": "val"})
        mock_col.upsert.assert_called_once_with(
            ids=["id1"],
            documents=["doc text"],
            metadatas=[{"key": "val"}],
        )


# ---------------------------------------------------------------------------
# query_similar
# ---------------------------------------------------------------------------

class TestQuerySimilar:
    def test_returns_list_of_dicts(self):
        store, _, mock_col = _make_store()
        mock_col.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"m": 1}, {"m": 2}]],
            "distances": [[0.1, 0.2]],
        }

        results = store.query_similar("query text", n_results=2)

        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["document"] == "doc1"
        assert results[0]["distance"] == 0.1
        assert results[1]["id"] == "id2"

    def test_passes_correct_n_results(self):
        store, _, mock_col = _make_store()
        mock_col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }

        store.query_similar("text", n_results=5)

        call_kwargs = mock_col.query.call_args.kwargs
        assert call_kwargs.get("n_results") == 5


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------

class TestGetById:
    def test_returns_record_when_found(self):
        store, _, mock_col = _make_store()
        mock_col.get.return_value = {
            "ids": ["id1"],
            "documents": ["doc text"],
            "metadatas": [{"k": "v"}],
        }

        result = store.get_by_id("id1")

        assert result is not None
        assert result["id"] == "id1"
        assert result["document"] == "doc text"

    def test_returns_none_when_not_found(self):
        store, _, mock_col = _make_store()
        mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        result = store.get_by_id("missing_id")

        assert result is None


# ---------------------------------------------------------------------------
# list_recent
# ---------------------------------------------------------------------------

class TestListRecent:
    def test_returns_n_most_recent(self):
        store, _, mock_col = _make_store()
        mock_col.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "documents": ["d1", "d2", "d3"],
            "metadatas": [
                {"inserted_at": "2024-01-01"},
                {"inserted_at": "2024-01-03"},
                {"inserted_at": "2024-01-02"},
            ],
        }

        results = store.list_recent(2)

        assert len(results) == 2
        # Most recent first
        assert results[0]["id"] == "id2"
        assert results[1]["id"] == "id3"

    def test_empty_collection(self):
        store, _, mock_col = _make_store()
        mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        results = store.list_recent(5)
        assert results == []


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------

class TestPing:
    def test_ping_returns_true_on_success(self):
        cfg = _mock_cfg()
        mock_client = MagicMock()

        with patch("core.tools.chroma_tool.get_config", return_value=cfg):
            with patch("core.tools.chroma_tool._build_client", return_value=mock_client):
                store = ChromaStore()
                result = store.ping()

        assert result is True
        mock_client.heartbeat.assert_called_once()

    def test_ping_returns_false_on_exception(self):
        cfg = _mock_cfg()
        mock_client = MagicMock()
        mock_client.heartbeat.side_effect = Exception("connection refused")

        with patch("core.tools.chroma_tool.get_config", return_value=cfg):
            with patch("core.tools.chroma_tool._build_client", return_value=mock_client):
                store = ChromaStore()
                result = store.ping()

        assert result is False

