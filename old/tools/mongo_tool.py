"""MongoDB snapshot query tool.

Provides balanced dataset retrieval from a neuralsignal MongoDB collection.
Each scan document stored by the neuralsignal SDK contains:
    - input / output / ground_truth / metadata / detections (lightweight fields)
    - inputs / outputs  (GridFS ObjectIds pointing at serialised tensors — heavy)

Tensor fields (``inputs``/``outputs``) are included by default since most callers
need them for featurisation.  Pass ``include_tensors=False`` for lightweight
metadata-only queries.  The neuralsignal MongoBackend
is not reused here because it drags in torch / GridFS deserialisation which is
expensive and unnecessary for metadata queries.

Typical usage
-------------
    from tools.mongo_tool import MongoSnapshotStore

    store = MongoSnapshotStore(db="my_app", collection="hallucination_scans")
    docs = store.fetch_balanced(n=200)          # 100 GT=0, 100 GT=1
    docs = store.fetch_balanced(
        n=200,
        extra_query={"metadata.model_name": "flan-t5-large"},
    )
    counts = store.class_counts()               # {"0": 1420, "1": 983}
    raw    = store.query({"metadata.foo": "bar"}, limit=50)
"""
from __future__ import annotations

import math
import random
from typing import Any

import pymongo
from bson import ObjectId

from configs.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)

# Fields that hold serialised tensor data as GridFS ObjectIds.
# Excluded by default to avoid returning opaque binary references.
_TENSOR_FIELDS = ("inputs", "outputs")


def _serialize_doc(doc: dict) -> dict:
    """Convert BSON types that are not JSON-serialisable to plain Python types.

    - ObjectId  → str
    - bytes     → '<bytes len=N>'  (summary; callers should not request tensors)
    """
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, bytes):
            out[k] = f"<bytes len={len(v)}>"
        elif isinstance(v, dict):
            out[k] = _serialize_doc(v)
        elif isinstance(v, list):
            out[k] = [_serialize_doc(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


class MongoSnapshotStore:
    """Thin pymongo client for querying neuralsignal scan snapshots.

    Connects lazily on first use.

    Args:
        db:         MongoDB database name.
        collection: MongoDB collection name.
        mongo_url:  Optional connection URI override. Falls back to
                    ``Config.mongo_url`` (from configs/config.yaml / .env).
    """

    def __init__(
        self,
        db: str,
        collection: str,
        mongo_url: str | None = None,
    ) -> None:
        cfg = get_config()
        self._url = mongo_url or cfg.mongo_url
        self._db_name = db
        self._col_name = collection
        self._client: pymongo.MongoClient | None = None
        self._col = None
        log.info(
            "MongoSnapshotStore initialised — db=%r, collection=%r, url=%r",
            db, collection, self._url,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_col(self):
        """Return the pymongo Collection, connecting lazily on first call."""
        if self._col is None:
            log.debug("MongoSnapshotStore | Connecting to %r", self._url)
            self._client = pymongo.MongoClient(
                self._url,
                serverSelectionTimeoutMS=5_000,
            )
            # Trigger a real connection attempt so errors surface immediately.
            self._client.admin.command("ping")
            self._col = self._client[self._db_name][self._col_name]
            log.info(
                "MongoSnapshotStore | Connected — db=%r, collection=%r",
                self._db_name, self._col_name,
            )
        return self._col

    @staticmethod
    def _projection(include_tensors: bool) -> dict | None:
        """Build a MongoDB projection that strips tensor fields unless requested."""
        if include_tensors:
            return None
        return {field: 0 for field in _TENSOR_FIELDS}

    def _fetch_class(
        self,
        ground_truth: int,
        n: int,
        extra_query: dict | None,
        include_tensors: bool,
    ) -> list[dict]:
        """Fetch up to *n* documents with the given ground_truth value."""
        query: dict = {"ground_truth": ground_truth}
        if extra_query:
            query.update(extra_query)

        proj = self._projection(include_tensors)
        cursor = self._get_col().find(query, proj).limit(n)
        docs = [_serialize_doc(d) for d in cursor]
        log.debug(
            "MongoSnapshotStore | ground_truth=%d → requested %d, got %d",
            ground_truth, n, len(docs),
        )
        return docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_balanced(
        self,
        n: int,
        extra_query: dict | None = None,
        include_tensors: bool = True,
        shuffle: bool = True,
    ) -> list[dict]:
        """Return a balanced dataset of *n* scan documents.

        Retrieves ``ceil(n/2)`` documents with ``ground_truth=1`` and
        ``floor(n/2)`` documents with ``ground_truth=0`` so that odd values
        of *n* always produce a dataset of exactly *n* items (one extra from
        class 1) when both classes have enough documents.

        If a class has fewer documents than requested, all available documents
        are returned for that class and a warning is logged.  The result may
        therefore contain fewer than *n* items.

        Args:
            n:               Total number of documents to return.
            extra_query:     Additional MongoDB filter merged into both class
                             queries (e.g. ``{"metadata.model": "flan-t5"}``).
            include_tensors: When True (default) the ``inputs`` and ``outputs``
                             GridFS reference fields are included in results.
                             Set to False for lightweight metadata-only queries.
            shuffle:         Shuffle the combined list before returning.

        Returns:
            List of plain-Python dicts.  ObjectIds are converted to strings.
        """
        n_pos = math.ceil(n / 2)   # ground_truth = 1
        n_neg = math.floor(n / 2)  # ground_truth = 0

        log.info(
            "MongoSnapshotStore.fetch_balanced | n=%d (pos=%d, neg=%d), "
            "extra_query=%s, include_tensors=%s",
            n, n_pos, n_neg, extra_query, include_tensors,
        )

        pos_docs = self._fetch_class(1, n_pos, extra_query, include_tensors)
        neg_docs = self._fetch_class(0, n_neg, extra_query, include_tensors)

        if len(pos_docs) < n_pos:
            log.warning(
                "MongoSnapshotStore.fetch_balanced | ground_truth=1 has only "
                "%d documents (wanted %d)",
                len(pos_docs), n_pos,
            )
        if len(neg_docs) < n_neg:
            log.warning(
                "MongoSnapshotStore.fetch_balanced | ground_truth=0 has only "
                "%d documents (wanted %d)",
                len(neg_docs), n_neg,
            )

        combined = pos_docs + neg_docs
        if shuffle:
            random.shuffle(combined)

        log.info(
            "MongoSnapshotStore.fetch_balanced | Returning %d documents "
            "(%d pos, %d neg)",
            len(combined), len(pos_docs), len(neg_docs),
        )
        return combined

    def class_counts(self, extra_query: dict | None = None) -> dict[str, int]:
        """Return document counts for each ground_truth class.

        Args:
            extra_query: Optional additional filter applied to both counts.

        Returns:
            Dict with keys ``"0"`` and ``"1"`` mapping to document counts.
        """
        base = extra_query or {}
        col = self._get_col()

        count_1 = col.count_documents({"ground_truth": 1, **base})
        count_0 = col.count_documents({"ground_truth": 0, **base})

        log.info(
            "MongoSnapshotStore.class_counts | ground_truth=0: %d, "
            "ground_truth=1: %d",
            count_0, count_1,
        )
        return {"0": count_0, "1": count_1}

    def query(
        self,
        query: dict,
        limit: int = 0,
        include_tensors: bool = True,
    ) -> list[dict]:
        """Run an arbitrary MongoDB query and return matching documents.

        Args:
            query:           pymongo filter dict.
            limit:           Maximum documents to return (0 = no limit).
            include_tensors: Include ``inputs``/``outputs`` GridFS refs (default True).

        Returns:
            List of plain-Python dicts.
        """
        proj = self._projection(include_tensors)
        cursor = self._get_col().find(query, proj)
        if limit > 0:
            cursor = cursor.limit(limit)

        docs = [_serialize_doc(d) for d in cursor]
        log.info(
            "MongoSnapshotStore.query | filter=%s, limit=%d → %d docs returned",
            query, limit, len(docs),
        )
        return docs

    def ping(self) -> bool:
        """Return True if the MongoDB server is reachable."""
        try:
            self._get_col()
            log.info("MongoSnapshotStore.ping | Server reachable — %r", self._url)
            return True
        except Exception as exc:
            log.warning("MongoSnapshotStore.ping | Server unreachable: %s", exc)
            return False
