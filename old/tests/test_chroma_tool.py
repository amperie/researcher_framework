"""Tests for tools/chroma_tool.py.

ChromaDB network calls are patched throughout — no running server required.
Methods that are not yet implemented (upsert, query_similar, get_by_id,
list_recent) are tested to confirm they raise NotImplementedError; these
tests will need updating once the implementations land.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.chroma_tool import ChromaStore, _build_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_collection(count: int = 0) -> MagicMock:
    col = MagicMock()
    col.count.return_value = count
    return col


def _mock_http_client(collection: MagicMock | None = None) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.return_value = collection or _mock_collection()
    return client


# ---------------------------------------------------------------------------
# _build_client
# ---------------------------------------------------------------------------

class TestBuildClient:
    def test_uses_config_host_and_port(self):
        with patch("tools.chroma_tool.chromadb.HttpClient") as mock_cls, \
             patch("tools.chroma_tool.get_config") as mock_cfg:
            mock_cfg.return_value.chroma_host = "my.server"
            mock_cfg.return_value.chroma_port = 9000
            mock_cfg.return_value.chroma_ssl = False
            mock_cfg.return_value.chroma_auth_token = None

            _build_client()

        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs["host"] == "my.server"
        assert kwargs["port"] == 9000
        assert kwargs["ssl"] is False

    def test_no_auth_settings_when_token_absent(self):
        with patch("tools.chroma_tool.chromadb.HttpClient") as mock_cls, \
             patch("tools.chroma_tool.get_config") as mock_cfg, \
             patch("tools.chroma_tool.Settings") as mock_settings:
            mock_cfg.return_value.chroma_host = "localhost"
            mock_cfg.return_value.chroma_port = 8000
            mock_cfg.return_value.chroma_ssl = False
            mock_cfg.return_value.chroma_auth_token = None

            _build_client()

        # Settings() called with no kwargs when no token
        mock_settings.assert_called_once_with()

    def test_auth_settings_injected_when_token_present(self):
        with patch("tools.chroma_tool.chromadb.HttpClient") as mock_cls, \
             patch("tools.chroma_tool.get_config") as mock_cfg, \
             patch("tools.chroma_tool.Settings") as mock_settings:
            mock_cfg.return_value.chroma_host = "localhost"
            mock_cfg.return_value.chroma_port = 8000
            mock_cfg.return_value.chroma_ssl = False
            mock_cfg.return_value.chroma_auth_token = "secret-token"

            _build_client()

        call_kwargs = mock_settings.call_args.kwargs
        assert call_kwargs["chroma_client_auth_credentials"] == "secret-token"
        assert "chroma_client_auth_provider" in call_kwargs


# ---------------------------------------------------------------------------
# ChromaStore.__init__
# ---------------------------------------------------------------------------

class TestChromaStoreInit:
    def test_uses_provided_collection_name(self):
        with patch("tools.chroma_tool.get_config"):
            store = ChromaStore(collection_name="my_collection")

        assert store._collection_name == "my_collection"

    def test_falls_back_to_config_collection(self):
        with patch("tools.chroma_tool.get_config") as mock_cfg:
            mock_cfg.return_value.chroma_collection = "default_col"
            mock_cfg.return_value.chroma_host = "localhost"
            mock_cfg.return_value.chroma_port = 8000
            store = ChromaStore()

        assert store._collection_name == "default_col"

    def test_client_and_collection_start_as_none(self):
        with patch("tools.chroma_tool.get_config"):
            store = ChromaStore(collection_name="test")

        assert store._client is None
        assert store._collection is None


# ---------------------------------------------------------------------------
# ChromaStore._get_collection (lazy init)
# ---------------------------------------------------------------------------

class TestGetCollection:
    def test_connects_lazily_on_first_call(self):
        col = _mock_collection(count=3)
        client = _mock_http_client(collection=col)

        with patch("tools.chroma_tool.get_config"), \
             patch("tools.chroma_tool._build_client", return_value=client):
            store = ChromaStore(collection_name="exp")
            assert store._collection is None  # not connected yet

            returned = store._get_collection()

        assert returned is col
        assert store._client is client
        client.get_or_create_collection.assert_called_once_with("exp")

    def test_does_not_reconnect_on_subsequent_calls(self):
        col = _mock_collection()
        client = _mock_http_client(collection=col)

        with patch("tools.chroma_tool.get_config"), \
             patch("tools.chroma_tool._build_client", return_value=client) as mock_build:
            store = ChromaStore(collection_name="exp")
            store._get_collection()
            store._get_collection()
            store._get_collection()

        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# ChromaStore.ping
# ---------------------------------------------------------------------------

class TestPing:
    def test_returns_true_when_server_reachable(self):
        client = MagicMock()
        client.heartbeat.return_value = {"nanosecond heartbeat": 123}

        with patch("tools.chroma_tool.get_config"), \
             patch("tools.chroma_tool._build_client", return_value=client):
            store = ChromaStore(collection_name="test")
            result = store.ping()

        assert result is True
        client.heartbeat.assert_called_once()

    def test_returns_false_when_server_unreachable(self):
        with patch("tools.chroma_tool.get_config"), \
             patch("tools.chroma_tool._build_client", side_effect=ConnectionRefusedError("refused")):
            store = ChromaStore(collection_name="test")
            result = store.ping()

        assert result is False

    def test_returns_false_on_any_exception(self):
        with patch("tools.chroma_tool.get_config"), \
             patch("tools.chroma_tool._build_client", side_effect=RuntimeError("boom")):
            store = ChromaStore(collection_name="test")
            result = store.ping()

        assert result is False


# ---------------------------------------------------------------------------
# Unimplemented methods — verify NotImplementedError until TODOs are filled
# ---------------------------------------------------------------------------

class TestUnimplementedMethods:
    """These tests document the current state. Remove/update as each method lands."""

    @pytest.fixture()
    def store(self):
        with patch("tools.chroma_tool.get_config"):
            return ChromaStore(collection_name="test")

    def test_upsert_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.upsert("id-1", "document text", {"key": "value"})

    def test_query_similar_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.query_similar("research direction text", n_results=5)

    def test_get_by_id_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.get_by_id("id-1")

    def test_list_recent_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.list_recent(n=10)
