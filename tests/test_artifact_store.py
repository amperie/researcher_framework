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
    get_artifact_store,
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
    assert fetched["storage_key"].endswith("/source.csv")
    assert fetched["storage_bucket"] == ""


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
    assert record["storage_key"].endswith("/model.json")


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


def test_artifact_store_records_s3_location_fields(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")

    class FakeS3Client:
        def upload_fileobj(self, fh, bucket, key, ExtraArgs=None):
            _ = fh.read()

        def put_object(self, **kwargs):
            return None

    metadata_store = MongoArtifactMetadataStore(
        mongo_url="mongodb://localhost:27017",
        db_name="artifacts",
        client=mongomock.MongoClient(),
    )
    backend = S3ArtifactBackend(
        bucket="artifacts",
        prefix="research",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="secret",
        client=FakeS3Client(),
    )
    store = ArtifactStore(metadata_store=metadata_store, backend=backend)

    record = store.store_file(
        source,
        artifact_type="dataset",
        profile_name="neuralsignal",
        proposal_name="activation_sparsity",
    )

    assert record["storage_bucket"] == "artifacts"
    assert record["storage_endpoint_url"] == "http://minio:9000"
    assert record["storage_key"].startswith("research/neuralsignal/dataset/")


def test_get_artifact_store_uses_profile_specific_mongo_namespace():
    from types import SimpleNamespace
    from unittest.mock import patch

    cfg = SimpleNamespace(
        mongo_url="mongodb://localhost:27017",
        artifacts_db_name="researcher_artifacts",
        artifacts_collection="artifacts",
        artifact_store_backend="filesystem",
        artifact_store_root="dev/artifacts",
    )
    profile = {
        "storage": {
            "mongodb_results_db": "researcher",
            "artifacts_mongodb_db": "researcher",
            "artifacts_collection": "neuralsignal_artifacts",
        }
    }

    with patch("core.artifacts.store.get_config", return_value=cfg):
        store = get_artifact_store(profile)

    assert isinstance(store.metadata_store, MongoArtifactMetadataStore)
    assert store.metadata_store.db_name == "researcher"
    assert store.metadata_store.collection_name == "neuralsignal_artifacts"
