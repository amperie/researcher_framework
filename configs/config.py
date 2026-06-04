"""Global runtime configuration — loaded from configs/config.yaml.

Secrets are read from environment variables using ${VAR} or ${VAR:default} syntax.
Optionally reads configs/.env before interpolation (for local dev convenience).
Environment variables always take precedence over .env values.

Profile-specific settings (prompts, datasets, base classes, metrics) belong in
configs/profiles/<name>.yaml — not here.
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import yaml

_ENV_PAT = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")
_CONFIG_PATH = Path("configs/config.yaml")
_ENV_PATH = Path("configs/.env")


def _strip_inline_comment(value: str) -> str:
    """Strip unquoted inline comments from a dotenv value."""
    quote: str | None = None
    escaped = False
    for idx, ch in enumerate(value):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in ("'", '"'):
            if quote == ch:
                quote = None
            elif quote is None:
                quote = ch
            continue
        if ch == "#" and quote is None and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value


def _load_dotenv() -> None:
    """Load key=value pairs from configs/.env into os.environ.

    No-op if the file does not exist. Environment variables already set
    take precedence over .env values (twelve-factor app convention).
    """
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = _strip_inline_comment(v.strip()).strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _interpolate(val: object) -> object:
    """Substitute ${VAR} / ${VAR:default} patterns with environment variable values."""
    if not isinstance(val, str):
        return val

    def _sub(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")

    return _ENV_PAT.sub(_sub, val)


def _walk(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    return _interpolate(obj)


def _normalise_dev_root(data: dict) -> None:
    dev_root = str(data.get("dev_root") or "dev").strip() or "dev"
    data["dev_root"] = dev_root
    if data.get("artifact_store_root") in (None, "", "dev/artifacts"):
        data["artifact_store_root"] = str(Path(dev_root) / "artifacts")
    if data.get("experiments_dir") in (None, "", "dev/experiments"):
        data["experiments_dir"] = str(Path(dev_root) / "experiments")


@lru_cache(maxsize=1)
def get_config() -> SimpleNamespace:
    """Return the singleton runtime config, loaded from configs/config.yaml."""
    _load_dotenv()
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    data: dict = _walk(raw)
    _normalise_dev_root(data)

    # The logging section is consumed only by utils/logger.py — strip it here.
    data.pop("logging", None)

    # Type coercions: env var substitution always produces strings, so cast explicitly.
    for int_key in (
        "chroma_port",
        "experiment_timeout_seconds",
        "validate_timeout_seconds",
        "max_arxiv_papers",
    ):
        if int_key in data and data[int_key] is not None:
            data[int_key] = int(data[int_key])

    if "chroma_ssl" in data and isinstance(data["chroma_ssl"], str):
        data["chroma_ssl"] = data["chroma_ssl"].lower() in ("true", "1", "yes")
    if "s3_secure" in data and isinstance(data["s3_secure"], str):
        data["s3_secure"] = data["s3_secure"].lower() in ("true", "1", "yes")

    # Normalise empty strings to None for optional fields.
    for opt_key in (
        "llm_model",
        "anthropic_api_key",
        "openai_api_key",
        "chroma_auth_token",
        "memory_neo4j_uri",
        "memory_neo4j_username",
        "memory_neo4j_password",
        "memory_neo4j_database",
        "s3_endpoint_url",
        "s3_access_key_id",
        "s3_secret_access_key",
        "s3_bucket",
        "s3_prefix",
        "ray_namespace",
    ):
        if not data.get(opt_key):
            data[opt_key] = None

    ray_mode = str(data.get("ray_mode") or "local").strip().lower()
    if ray_mode not in {"local", "remote"}:
        raise ValueError(f"Invalid ray_mode {ray_mode!r}; expected 'local' or 'remote'")
    data["ray_mode"] = ray_mode
    if not data.get("ray_address"):
        data["ray_address"] = "auto"

    return SimpleNamespace(**data)


def dev_path(*parts: str) -> Path:
    """Return a path rooted under the configured development root."""
    return Path(get_config().dev_root).joinpath(*parts)


def resolve_dev_path(path: str | Path) -> Path:
    """Resolve legacy ``dev/...`` paths against the configured development root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if not candidate.parts:
        return Path(get_config().dev_root)
    if candidate.parts[0] == "dev":
        return dev_path(*candidate.parts[1:])
    return candidate


def platform_key() -> str:
    """Return the config key for the current operating system."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def cache_path(name: str) -> Path:
    """Return the configured cache path for *name* on the current platform."""
    locations = getattr(get_config(), "cache_locations", {}) or {}
    entry = locations.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"No cache location configured for {name!r}")
    value = entry.get(platform_key()) or entry.get("default")
    if not value:
        raise KeyError(f"No {platform_key()!r} or default cache location configured for {name!r}")
    return resolve_dev_path(Path(str(value)).expanduser())
