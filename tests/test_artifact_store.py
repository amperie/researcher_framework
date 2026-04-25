"""Tests for artifact storage backends and Mongo metadata registry."""
from __future__ import annotations

import json
from pathlib import Path

import mongomock

from core.artifacts.store import (
    ArtifactStore,
    FilesystemArtifactBackend,
    MongoArtifactMetadataStore,
    S3ArtifactBackend,
)


def test_filesystem_artifact_store_copies_file_and_registers_metadata(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    metadata_store = MongoArtifactMetadataStore(
        mongo_url="mongodb://localhost:27017",
        db_name="artifacts",
        client=mongomock.MongoClient(),
    )
    backend = FilesystemArtifactBackend(tmp_path / "artifacts")
    store = ArtifactStore(metadata_store=metadata_store, backend=backend)

    record = store.store_file(
        source,
        artifact_type="dataset",
        profile_name="neuralsignal",
        proposal_name="activation_sparsity",
        experiment_id="exp-1",
        metadata={"rows": 1, "columns": 2},
        tags=["dataset"],
    )

    stored_path = Path(record["uri"])
    assert stored_path.exists()
    assert stored_path.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    fetched = store.get(record["artifact_id"])
    assert fetched is not None
    assert fetched["artifact_type"] == "dataset"
    assert fetched["metadata"]["rows"] == 1


def test_artifact_store_writes_json_artifact(tmp_path):
    metadata_store = MongoArtifactMetadataStore(
        mongo_url="mongodb://localhost:27017",
        db_name="artifacts",
        client=mongomock.MongoClient(),
    )
    backend = FilesystemArtifactBackend(tmp_path / "artifacts")
    store = ArtifactStore(metadata_store=metadata_store, backend=backend)

    record = store.store_json(
        {"metrics": {"test_auc": 0.71}},
        artifact_type="model",
        profile_name="neuralsignal",
        proposal_name="activation_sparsity",
        experiment_id="exp-2",
        artifact_name="model.json",
    )

    stored_path = Path(record["uri"])
    assert stored_path.exists()
    payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["test_auc"] == 0.71
    assert record["format"] == "json"


def test_mongo_metadata_store_find_filters_records():
    client = mongomock.MongoClient()
    store = MongoArtifactMetadataStore(
        mongo_url="mongodb://localhost:27017",
        db_name="artifacts",
        client=client,
    )
    store.put({"artifact_id": "a1", "artifact_type": "dataset", "proposal_name": "p1"})
    store.put({"artifact_id": "a2", "artifact_type": "model", "proposal_name": "p1"})
    store.put({"artifact_id": "a3", "artifact_type": "dataset", "proposal_name": "p2"})

    rows = store.find({"proposal_name": "p1"})

    assert {row["artifact_id"] for row in rows} == {"a1", "a2"}


def test_s3_backend_uploads_file_and_returns_uri(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")

    class FakeS3Client:
        def __init__(self):
            self.calls = []

        def upload_fileobj(self, fh, bucket, key, ExtraArgs=None):
            self.calls.append({
                "body": fh.read(),
                "bucket": bucket,
                "key": key,
                "extra": ExtraArgs,
            })

        def put_object(self, **kwargs):
            self.calls.append(kwargs)

    client = FakeS3Client()
    backend = S3ArtifactBackend(
        bucket="artifacts",
        prefix="research",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="secret",
        client=client,
    )

    uri = backend.put_file(source, "neuralsignal/dataset/file.csv")

    assert uri == "http://minio:9000/artifacts/research/neuralsignal/dataset/file.csv"
    assert client.calls[0]["bucket"] == "artifacts"
    assert client.calls[0]["key"] == "research/neuralsignal/dataset/file.csv"
    assert client.calls[0]["body"] == source.read_bytes()
