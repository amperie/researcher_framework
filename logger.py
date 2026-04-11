"""Centralised logging setup for NeuralSignalResearcher.

Usage
-----
In main.py (once, at startup)::

    from logger import setup_logging
    setup_logging()                        # reads config.yaml
    setup_logging("path/to/config.yaml")   # explicit path

In every other module::

    from logger import get_logger
    log = get_logger(__name__)

    log.debug("Detailed trace: %s", value)
    log.info("Human-readable milestone")
    log.warning("Non-fatal anomaly: %s", detail)
    log.error("Recoverable error: %s", exc, exc_info=True)
    log.critical("Unrecoverable failure", exc_info=True)

Config file
-----------
Reads ``config.yaml`` at the project root (or the path passed to setup_logging).
If the file is absent or malformed the module falls back to a safe default:
INFO on console, no file handler — so the app always produces some output.

Level precedence
----------------
The root logger is set to ``min(console.level, file.level)`` so that Python's
logger hierarchy never silently drops records before they reach a handler.
Each handler then applies its own level filter independently.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_DEFAULT_CONSOLE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DEFAULT_FILE_FORMAT = (
    "%(asctime)s [%(levelname)-8s] %(name)-40s (%(filename)s:%(lineno)d): %(message)s"
)
_FALLBACK_LEVEL = logging.INFO


def _level(name: str) -> int:
    """Convert a level name string to a logging int, defaulting to INFO."""
    level = logging.getLevelName(name.upper()) if isinstance(name, str) else name
    return level if isinstance(level, int) else _FALLBACK_LEVEL


def setup_logging(config_path: str = "config.yaml") -> None:
    """Configure the root logger from *config_path*.

    Safe to call multiple times — subsequent calls reconfigure handlers
    in-place rather than stacking duplicates.

    Args:
        config_path: Path to the YAML config file. Relative paths are
                     resolved from the current working directory.
    """
    cfg = _load_yaml_section(config_path)

    # Remove any handlers already attached to the root logger so that
    # repeated calls don't produce duplicate output.
    root = logging.getLogger()
    root.handlers.clear()

    console_cfg: dict[str, Any] = cfg.get("console", {})
    file_cfg: dict[str, Any] = cfg.get("file", {})

    console_level = _level(console_cfg.get("level", "INFO"))
    file_level = _level(file_cfg.get("level", "DEBUG"))

    # Root logger must be at least as permissive as the least restrictive
    # handler, otherwise records are dropped before reaching any handler.
    global_level = _level(cfg.get("level", "DEBUG"))
    effective_root = min(global_level, console_level, file_level)
    root.setLevel(effective_root)

    handlers_added: list[str] = []

    # --- Console handler ---
    if console_cfg.get("enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter(
                fmt=console_cfg.get("format", _DEFAULT_CONSOLE_FORMAT),
                datefmt=console_cfg.get("datefmt", "%H:%M:%S"),
            )
        )
        root.addHandler(console_handler)
        handlers_added.append(f"console(level={logging.getLevelName(console_level)})")

    # --- Rotating file handler ---
    if file_cfg.get("enabled", False):
        log_path = Path(file_cfg.get("path", "./logs/neuralsignalresearcher.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=file_cfg.get("max_bytes", 10 * 1024 * 1024),
            backupCount=file_cfg.get("backup_count", 5),
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt=file_cfg.get("format", _DEFAULT_FILE_FORMAT),
                datefmt=file_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
            )
        )
        root.addHandler(file_handler)
        handlers_added.append(
            f"file(level={logging.getLevelName(file_level)}, path={log_path})"
        )

    # Quieten noisy third-party loggers that tend to flood output.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = get_logger(__name__)
    log.debug(
        "Logging initialised — handlers: %s",
        ", ".join(handlers_added) if handlers_added else "none (fallback)",
    )


def _load_yaml_section(config_path: str) -> dict[str, Any]:
    """Load the ``logging`` section from *config_path*.

    Returns an empty dict on any failure so callers always get a usable
    (possibly empty) config dict and the app degrades gracefully.
    """
    try:
        import yaml  # pyyaml ships as a transitive dep of mlflow

        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("logging", {})
    except FileNotFoundError:
        # config.yaml missing — emit a one-time warning to stderr and continue.
        print(
            f"[logger] config.yaml not found at {config_path!r}; "
            "using default logging (INFO console, no file).",
            file=sys.stderr,
        )
        return {}
    except Exception as exc:  # noqa: BLE001
        print(f"[logger] Failed to parse {config_path!r}: {exc}; using defaults.", file=sys.stderr)
        return {}


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Always use ``get_logger(__name__)`` at module level so that log records
    carry the full dotted module path (e.g. ``graph.nodes.research``).

    Args:
        name: Logger name — conventionally ``__name__``.
    """
    return logging.getLogger(name)
