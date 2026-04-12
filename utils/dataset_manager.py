"""Persistent dataset registry backed by MongoDB + GridFS.

Identity: (feature_set_name, config_hash) where config_hash is SHA-256[:16]
over experiment_config minus volatile keys {experiment_id, script_path}.

Datasets are stored as parquet bytes in GridFS for fast pandas deserialization.
Registry documents live in the 'datasets' collection.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import pymongo
import gridfs


_VOLATILE_KEYS = {"experiment_id", "script_path"}


class DatasetManager:
    def __init__(self, mongo_url: str, db_name: str = "neuralsignal_datasets") -> None:
        self._client = pymongo.MongoClient(mongo_url)
        self._db = self._client[db_name]
        self._registry = self._db["datasets"]
        self._fs = gridfs.GridFS(self._db)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------
    def config_hash(self, experiment_config: dict) -> str:
        stable = {k: v for k, v in experiment_config.items()
                  if k not in _VOLATILE_KEYS}
        canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def find_cached(
        self, feature_set_name: str, experiment_config: dict
    ) -> dict | None:
        h = self.config_hash(experiment_config)
        doc = self._registry.find_one(
            {"feature_set_name": feature_set_name, "config_hash": h}
        )
        if doc is None:
            return None
        # Verify the GridFS file still exists
        if not self._fs.exists(doc["gridfs_id"]):
            return None
        doc["_id"] = str(doc["_id"])
        doc["gridfs_id"] = str(doc["gridfs_id"])
        return doc

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_dataset(
        self,
        experiment_id: str,
        feature_set_name: str,
        experiment_config: dict,
        column_names: list[str],
        column_values: list,          # list[float] (single row) or list[list[float]]
        scan_ids: list[str] | None = None,
    ) -> dict:
        # Normalise to list-of-rows
        if column_values and not isinstance(column_values[0], (list, tuple)):
            rows = [column_values]
        else:
            rows = list(column_values)

        if scan_ids is None:
            scan_ids = [str(uuid4()) for _ in rows]

        df = pd.DataFrame(rows, columns=column_names, index=scan_ids)
        df.index.name = "scan_id"

        # Serialise to parquet bytes
        buf = io.BytesIO()
        df.to_parquet(buf)
        buf.seek(0)

        dataset_id = str(uuid4())
        gridfs_id = self._fs.put(
            buf.read(),
            filename=f"{feature_set_name}/{dataset_id}.parquet",
            content_type="application/octet-stream",
        )

        doc = {
            "dataset_id": dataset_id,
            "feature_set_name": feature_set_name,
            "experiment_id": experiment_id,
            "creation_time": datetime.now(timezone.utc).isoformat(),
            "config_hash": self.config_hash(experiment_config),
            "column_names": column_names,
            "num_rows": len(rows),
            "gridfs_id": gridfs_id,
        }
        self._registry.insert_one(doc)

        entry = {**doc, "_id": str(doc["_id"]), "gridfs_id": str(gridfs_id)}
        return entry

    def load_dataset(self, entry: dict) -> pd.DataFrame:
        import bson
        gid = entry["gridfs_id"]
        if isinstance(gid, str):
            gid = bson.ObjectId(gid)
        data = self._fs.get(gid).read()
        return pd.read_parquet(io.BytesIO(data))

    # ------------------------------------------------------------------
    # On-demand meld (utility — not called from the runner)
    # ------------------------------------------------------------------
    def meld(self, entries: list[dict]) -> pd.DataFrame:
        dfs = [
            self.load_dataset(e).add_prefix(f"{e['feature_set_name']}__")
            for e in entries
        ]
        return pd.concat(dfs, axis=1)

    def meld_by_config_hash(self, config_hash: str) -> pd.DataFrame:
        docs = list(self._registry.find({"config_hash": config_hash}))
        entries = [
            {**d, "_id": str(d["_id"]), "gridfs_id": str(d["gridfs_id"])}
            for d in docs
            if self._fs.exists(d["gridfs_id"])
        ]
        return self.meld(entries)

    # ------------------------------------------------------------------
    # Build synthetic result for cache hits
    # ------------------------------------------------------------------
    def build_cached_result(self, entry: dict, script: dict) -> dict:
        return {
            "experiment_id": entry["experiment_id"],
            "proposal_name": script.get("proposal_name", ""),
            "stdout": "",
            "stderr": "",
            "success": True,
            "raw_results": {
                "feature_set_name": entry["feature_set_name"],
                "column_names": entry["column_names"],
                "status": "ok",
                "from_cache": True,
            },
            "dataset_id": entry["dataset_id"],
            "from_cache": True,
        }
