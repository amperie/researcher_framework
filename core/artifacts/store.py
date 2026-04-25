"""Artifact storage with Mongo metadata plus pluggable file backends."""
from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import pymongo

from configs.config import get_config


class ArtifactBackend(Protocol):
    """Backend that stores artifact bytes and returns a storage URI."""

    backend_name: str

    def put_file(self, src_path: str | Path, key: str) -> str: ...

    def put_bytes(self, data: bytes, key: str, content_type: str | None = None) -> str: ...


class ArtifactMetadataStore(Protocol):
    """Metadata registry for stored artifacts."""

    def put(self, record: dict[str, Any]) -> dict[str, Any]: ...

    def get(self, artifact_id: str) -> dict[str, Any] | None: ...

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]: ...


@dataclass
class FilesystemArtifactBackend:
    """Store artifacts on the local filesystem under a managed root."""

    root: Path
    backend_name: str = "filesystem"

    def put_file(self, src_path: str | Path, key: str) -> str:
        src = Path(src_path)
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return str(dest.resolve())

    def put_bytes(self, data: bytes, key: str, content_type: str | None = None) -> str:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return str(dest.resolve())


@dataclass
class S3ArtifactBackend:
    """Store artifacts in S3-compatible object storage such as MinIO."""

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    region_name: str | None = None
    secure: bool = True
    client: Any | None = None
    backend_name: str = "s3"

    def __post_init__(self) -> None:
        if self.client is None:
            import boto3

            self.client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )

    def put_file(self, src_path: str | Path, key: str) -> str:
        object_key = self._object_key(key)
        extra_args = _s3_extra_args(Path(src_path))
        with Path(src_path).open("rb") as fh:
            self.client.upload_fileobj(fh, self.bucket, object_key, ExtraArgs=extra_args or None)
        return self._uri(object_key)

    def put_bytes(self, data: bytes, key: str, content_type: str | None = None) -> str:
        object_key = self._object_key(key)
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": object_key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)
        return self._uri(object_key)

    def _object_key(self, key: str) -> str:
        cleaned = key.lstrip("/").replace("\\", "/")
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{cleaned}"
        return cleaned

    def _uri(self, object_key: str) -> str:
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{object_key}"
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.bucket}.s3.amazonaws.com/{object_key}"


@dataclass
class MongoArtifactMetadataStore:
    """Mongo-backed registry for artifact metadata and references."""

    mongo_url: str
    db_name: str
    collection_name: str = "artifacts"
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = pymongo.MongoClient(self.mongo_url)

    @property
    def collection(self) -> Any:
        return self.client[self.db_name][self.collection_name]

    def put(self, record: dict[str, Any]) -> dict[str, Any]:
        doc = dict(record)
        self.collection.replace_one({"artifact_id": doc["artifact_id"]}, doc, upsert=True)
        return doc

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        record = self.collection.find_one({"artifact_id": artifact_id})
        if not record:
            return None
        record.pop("_id", None)
        return record

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
        docs = list(self.collection.find(filters).limit(limit))
        for doc in docs:
            doc.pop("_id", None)
        return docs


@dataclass
class ArtifactStore:
    """Coordinates byte storage and Mongo metadata registration."""

    metadata_store: ArtifactMetadataStore
    backend: ArtifactBackend

    def store_file(
        self,
        src_path: str | Path,
        *,
        artifact_type: str,
        profile_name: str = "",
        proposal_name: str = "",
        experiment_id: str = "",
        artifact_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        src = Path(src_path)
        artifact_id = str(uuid4())
        filename = artifact_name or src.name
        key = _artifact_key(profile_name, artifact_type, artifact_id, filename)
        uri = self.backend.put_file(src, key)

        record = _artifact_record(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            storage_backend=self.backend.backend_name,
            uri=uri,
            file_name=filename,
            profile_name=profile_name,
            proposal_name=proposal_name,
            experiment_id=experiment_id,
            sha256=_sha256_file(src),
            size_bytes=src.stat().st_size,
            mime_type=mimetypes.guess_type(filename)[0],
            metadata=metadata or {},
            tags=tags or [],
            extra=extra or {},
        )
        return self.metadata_store.put(record)

    def store_json(
        self,
        payload: dict[str, Any],
        *,
        artifact_type: str,
        profile_name: str = "",
        proposal_name: str = "",
        experiment_id: str = "",
        artifact_name: str = "artifact.json",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_id = str(uuid4())
        data = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        key = _artifact_key(profile_name, artifact_type, artifact_id, artifact_name)
        uri = self.backend.put_bytes(data, key, content_type="application/json")

        record = _artifact_record(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            storage_backend=self.backend.backend_name,
            uri=uri,
            file_name=artifact_name,
            profile_name=profile_name,
            proposal_name=proposal_name,
            experiment_id=experiment_id,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            mime_type="application/json",
            metadata=metadata or {},
            tags=tags or [],
            extra=extra or {},
        )
        return self.metadata_store.put(record)

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        return self.metadata_store.get(artifact_id)

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
        return self.metadata_store.find(filters, limit=limit)


def get_artifact_store() -> ArtifactStore:
    """Build the configured artifact store."""
    cfg = get_config()
    metadata_store = MongoArtifactMetadataStore(
        mongo_url=cfg.mongo_url,
        db_name=getattr(cfg, "artifacts_db_name", "researcher_artifacts"),
        collection_name=getattr(cfg, "artifacts_collection", "artifacts"),
    )

    backend_name = getattr(cfg, "artifact_store_backend", "filesystem")
    if backend_name == "s3":
        bucket = getattr(cfg, "s3_bucket", None)
        if not bucket:
            raise ValueError("artifact_store_backend='s3' requires s3_bucket to be configured")
        backend = S3ArtifactBackend(
            bucket=bucket,
            prefix=getattr(cfg, "s3_prefix", None) or "",
            endpoint_url=getattr(cfg, "s3_endpoint_url", None),
            aws_access_key_id=getattr(cfg, "s3_access_key_id", None),
            aws_secret_access_key=getattr(cfg, "s3_secret_access_key", None),
            region_name=getattr(cfg, "s3_region_name", None),
            secure=bool(getattr(cfg, "s3_secure", True)),
        )
    else:
        backend = FilesystemArtifactBackend(Path(getattr(cfg, "artifact_store_root", "dev/artifacts")).resolve())
    return ArtifactStore(metadata_store=metadata_store, backend=backend)


def _artifact_key(profile_name: str, artifact_type: str, artifact_id: str, file_name: str) -> str:
    profile = _slug(profile_name or "default")
    kind = _slug(artifact_type or "artifact")
    return f"{profile}/{kind}/{artifact_id}/{file_name}"


def _artifact_record(
    *,
    artifact_id: str,
    artifact_type: str,
    storage_backend: str,
    uri: str,
    file_name: str,
    profile_name: str,
    proposal_name: str,
    experiment_id: str,
    sha256: str,
    size_bytes: int,
    mime_type: str | None,
    metadata: dict[str, Any],
    tags: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "storage_backend": storage_backend,
        "uri": uri,
        "file_name": file_name,
        "format": Path(file_name).suffix.lstrip(".").lower(),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "mime_type": mime_type,
        "profile_name": profile_name,
        "proposal_name": proposal_name,
        "experiment_id": experiment_id,
        "metadata": metadata,
        "tags": tags,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def _s3_extra_args(path: Path) -> dict[str, Any]:
    content_type = mimetypes.guess_type(path.name)[0]
    return {"ContentType": content_type} if content_type else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value.strip())
    return "_".join(part for part in normalized.split("_") if part) or "artifact"
