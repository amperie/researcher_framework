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


@lru_cache(maxsize=1)
def get_config() -> SimpleNamespace:
    """Return the singleton runtime config, loaded from configs/config.yaml."""
    _load_dotenv()
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    data: dict = _walk(raw)

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

    # Normalise empty strings to None for optional fields.
    for opt_key in ("llm_model", "anthropic_api_key", "openai_api_key", "chroma_auth_token"):
        if not data.get(opt_key):
            data[opt_key] = None

    return SimpleNamespace(**data)
