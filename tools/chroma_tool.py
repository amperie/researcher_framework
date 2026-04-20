"""ChromaDB wrapper for experiment storage and semantic retrieval."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings

from configs.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


def _build_client() -> chromadb.HttpClient:
    cfg = get_config()
    settings_kwargs: dict = {}
    if cfg.chroma_auth_token:
        settings_kwargs = {
            "chroma_client_auth_provider": "chromadb.auth.token_authn.TokenAuthClientProvider",
            "chroma_client_auth_credentials": cfg.chroma_auth_token,
        }
    return chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        ssl=cfg.chroma_ssl,
        settings=Settings(**settings_kwargs) if settings_kwargs else Settings(),
    )


class ChromaStore:
    """Thin wrapper around a remote ChromaDB collection. Lazily connects on first use."""

    def __init__(self, collection_name: str | None = None) -> None:
        cfg = get_config()
        self._collection_name = collection_name or cfg.chroma_collection
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._client = _build_client()
            self._collection = self._client.get_or_create_collection(self._collection_name)
            log.info(
                "ChromaStore | Connected — collection=%r, records=%d",
                self._collection_name, self._collection.count(),
            )
        return self._collection

    def upsert(self, record_id: str, document: str, metadata: dict) -> None:
        col = self._get_collection()
        col.upsert(ids=[record_id], documents=[document], metadatas=[metadata])
        log.debug("ChromaStore.upsert | %r", record_id)

    def query_similar(self, text: str, n_results: int) -> list[dict]:
        col = self._get_collection()
        raw = col.query(query_texts=[text], n_results=n_results)
        return [
            {"id": id_, "document": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                raw["ids"][0], raw["documents"][0],
                raw["metadatas"][0], raw["distances"][0],
            )
        ]

    def get_by_id(self, record_id: str) -> dict | None:
        col = self._get_collection()
        raw = col.get(ids=[record_id])
        if not raw["ids"]:
            return None
        return {"id": raw["ids"][0], "document": raw["documents"][0], "metadata": raw["metadatas"][0]}

    def list_recent(self, n: int) -> list[dict]:
        col = self._get_collection()
        raw = col.get()
        records = [
            {"id": id_, "document": doc, "metadata": meta}
            for id_, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
        records.sort(key=lambda r: r["metadata"].get("inserted_at", ""), reverse=True)
        return records[:n]

    def ping(self) -> bool:
        try:
            _build_client().heartbeat()
            return True
        except Exception:
            return False
