"""ChromaDB wrapper for experiment storage and semantic retrieval.

Connects to a remote ChromaDB HTTP server (default: hp.lan:8000).
Server connection parameters are read from config.yaml / .env via Config.

Persists full experiment records (research summary, code, results, analysis)
as documents with metadata so that past experiments can be retrieved by
semantic similarity when analysing new results or proposing follow-ups.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings

from configs.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


def _build_client() -> chromadb.HttpClient:
    """Construct a ChromaDB HttpClient from the active Config.

    Auth: if chroma_auth_token is set the client sends it as a Bearer token
    via chromadb's built-in token auth provider.
    """
    cfg = get_config()

    settings_kwargs: dict = {}
    if cfg.chroma_auth_token:
        settings_kwargs = {
            "chroma_client_auth_provider": "chromadb.auth.token_authn.TokenAuthClientProvider",
            "chroma_client_auth_credentials": cfg.chroma_auth_token,
        }
        log.debug("ChromaDB auth token configured")

    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        ssl=cfg.chroma_ssl,
        settings=Settings(**settings_kwargs) if settings_kwargs else Settings(),
    )
    log.debug("ChromaDB HttpClient created — %s://%s:%d",
              "https" if cfg.chroma_ssl else "http", cfg.chroma_host, cfg.chroma_port)
    return client


class ChromaStore:
    """Thin wrapper around a remote ChromaDB collection.

    Lazily connects on first use so that import-time failures (e.g. server
    unreachable) don't crash unrelated parts of the app.

    Args:
        collection_name: ChromaDB collection to use. Defaults to Config.chroma_collection.
    """

    def __init__(self, collection_name: str | None = None) -> None:
        cfg = get_config()
        self._collection_name = collection_name or cfg.chroma_collection
        self._client: chromadb.HttpClient | None = None
        self._collection = None
        log.info("ChromaStore initialised — server=%s:%d, collection=%r",
                 cfg.chroma_host, cfg.chroma_port, self._collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self):
        """Return the collection, connecting lazily on first call."""
        if self._collection is None:
            log.debug("ChromaStore | Connecting to remote server")
            self._client = _build_client()
            self._collection = self._client.get_or_create_collection(self._collection_name)
            log.info("ChromaStore | Connected — collection=%r, existing_records=%d",
                     self._collection_name, self._collection.count())
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, record_id: str, document: str, metadata: dict) -> None:
        """Insert or update an experiment record.

        Args:
            record_id: Unique identifier (experiment_id / UUID).
            document:  Full-text representation of the experiment for embedding.
            metadata:  Flat dict of scalar values (mlflow_run_id, metrics, etc.).
                       ChromaDB requires all values to be str, int, float, or bool.

        TODO: call self._get_collection().upsert(...).
        """
        log.info("ChromaStore.upsert — record_id=%r, doc_len=%d, metadata_keys=%s",
                 record_id, len(document), list(metadata.keys()))
        # col = self._get_collection()
        # col.upsert(ids=[record_id], documents=[document], metadatas=[metadata])
        # log.debug("ChromaStore.upsert | Complete for record_id=%r", record_id)
        raise NotImplementedError("ChromaStore.upsert is not yet implemented")

    def query_similar(self, text: str, n_results: int) -> list[dict]:
        """Return the n most semantically similar experiment records.

        Args:
            text:      Query text (e.g. the current research direction).
            n_results: Number of results to return.

        Returns:
            List of dicts, each with keys: id, document, metadata, distance.

        TODO: call self._get_collection().query(...) and reshape the response.
        """
        preview = text[:80] + ("…" if len(text) > 80 else "")
        log.info("ChromaStore.query_similar — query=%r, n_results=%d", preview, n_results)
        # col = self._get_collection()
        # raw = col.query(query_texts=[text], n_results=n_results)
        # results = [
        #     {"id": id_, "document": doc, "metadata": meta, "distance": dist}
        #     for id_, doc, meta, dist in zip(
        #         raw["ids"][0], raw["documents"][0],
        #         raw["metadatas"][0], raw["distances"][0],
        #     )
        # ]
        # log.debug("ChromaStore.query_similar | %d matches, closest distance=%s",
        #           len(results), results[0]["distance"] if results else "n/a")
        # return results
        raise NotImplementedError("ChromaStore.query_similar is not yet implemented")

    def get_by_id(self, record_id: str) -> dict | None:
        """Fetch a single record by its exact ID.

        Returns:
            Dict with keys id, document, metadata, or None if not found.

        TODO: call self._get_collection().get(ids=[record_id]).
        """
        log.debug("ChromaStore.get_by_id — record_id=%r", record_id)
        # col = self._get_collection()
        # raw = col.get(ids=[record_id])
        # if not raw["ids"]:
        #     log.warning("ChromaStore.get_by_id | Record not found: %r", record_id)
        #     return None
        # return {"id": raw["ids"][0], "document": raw["documents"][0],
        #         "metadata": raw["metadatas"][0]}
        raise NotImplementedError("ChromaStore.get_by_id is not yet implemented")

    def list_recent(self, n: int) -> list[dict]:
        """Return the n most recently inserted experiment records.

        Sorts client-side by the 'inserted_at' timestamp stored in each
        record's metadata (set by db_logger_node at upsert time).

        Args:
            n: Number of records to return.

        Returns:
            List of dicts with keys: id, document, metadata.

        TODO: call self._get_collection().get(), sort by metadata['inserted_at'] desc,
              slice to n.
        """
        log.debug("ChromaStore.list_recent — n=%d", n)
        # col = self._get_collection()
        # raw = col.get()
        # total = len(raw["ids"])
        # log.debug("ChromaStore.list_recent | Total records in collection: %d", total)
        # records = [
        #     {"id": id_, "document": doc, "metadata": meta}
        #     for id_, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        # ]
        # records.sort(key=lambda r: r["metadata"].get("inserted_at", ""), reverse=True)
        # return records[:n]
        raise NotImplementedError("ChromaStore.list_recent is not yet implemented")

    def ping(self) -> bool:
        """Return True if the remote ChromaDB server is reachable.

        Useful for health-checks at startup before running the full pipeline.
        """
        try:
            client = _build_client()
            client.heartbeat()
            log.info("ChromaStore.ping | Server reachable — %s:%d",
                     get_config().chroma_host, get_config().chroma_port)
            return True
        except Exception as exc:
            log.warning("ChromaStore.ping | Server unreachable: %s", exc)
            return False
