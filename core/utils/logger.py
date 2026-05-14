"""Centralised logging setup for the research pipeline.

Usage
-----
In main.py (once, at startup)::

    from core.utils.logger import setup_logging
    setup_logging()                           # reads configs/config.yaml
    setup_logging("path/to/config.yaml")      # explicit path

In every other module::

    from core.utils.logger import get_logger
    log = get_logger(__name__)
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import logging.handlers
import sys
import ctypes
from pathlib import Path
from typing import Any

_DEFAULT_CONSOLE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DEFAULT_FILE_FORMAT = (
    "%(asctime)s [%(levelname)-8s] %(name)-40s (%(filename)s:%(lineno)d): %(message)s"
)
_FALLBACK_LEVEL = logging.INFO
_RESET = "\x1b[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[36m",
    logging.INFO: "\x1b[32m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[35m",
}


def _level(name: str) -> int:
    level = logging.getLevelName(name.upper()) if isinstance(name, str) else name
    return level if isinstance(level, int) else _FALLBACK_LEVEL


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{_RESET}"


class _IgnoreLoggerLevelsFilter(logging.Filter):
    """Suppress records for selected logger prefixes up to a configured level."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        super().__init__()
        self.rules: list[tuple[str, int]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            if not name:
                continue
            max_level = _level(rule.get("max_level", "INFO"))
            self.rules.append((name, max_level))

    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, max_level in self.rules:
            if record.name == prefix or record.name.startswith(f"{prefix}."):
                return record.levelno > max_level
        return True


def _enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle in (0, -1):
            return
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        return


def setup_logging(config_path: str = "configs/config.yaml") -> None:
    """Configure the root logger from *config_path*.

    Safe to call multiple times — subsequent calls reconfigure handlers
    in-place rather than stacking duplicates.
    """
    cfg = _load_yaml_section(config_path)

    root = logging.getLogger()
    root.handlers.clear()

    console_cfg: dict[str, Any] = cfg.get("console", {})
    file_cfg: dict[str, Any] = cfg.get("file", {})
    ignore_loggers_cfg = list(cfg.get("ignore_loggers") or [])
    ignore_filter = _IgnoreLoggerLevelsFilter(ignore_loggers_cfg) if ignore_loggers_cfg else None

    console_level = _level(console_cfg.get("level", "INFO"))
    file_level = _level(file_cfg.get("level", "DEBUG"))
    global_level = _level(cfg.get("level", "DEBUG"))
    effective_root = min(global_level, console_level, file_level)
    root.setLevel(effective_root)

    handlers_added: list[str] = []

    if console_cfg.get("enabled", True):
        _enable_windows_ansi()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            _ColorFormatter(
                fmt=console_cfg.get("format", _DEFAULT_CONSOLE_FORMAT),
                datefmt=console_cfg.get("datefmt", "%H:%M:%S"),
            )
        )
        if ignore_filter is not None:
            console_handler.addFilter(ignore_filter)
        root.addHandler(console_handler)
        handlers_added.append(f"console(level={logging.getLevelName(console_level)})")

    if file_cfg.get("enabled", False):
        log_path = Path(file_cfg.get("path", "logs/research.log"))
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
        if ignore_filter is not None:
            file_handler.addFilter(ignore_filter)
        root.addHandler(file_handler)
        handlers_added.append(
            f"file(level={logging.getLevelName(file_level)}, path={log_path})"
        )

    for noisy in ("httpx", "httpcore", "urllib3", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = get_logger(__name__)
    log.debug(
        "Logging initialised — handlers: %s",
        ", ".join(handlers_added) if handlers_added else "none (fallback)",
    )


def _load_yaml_section(config_path: str) -> dict[str, Any]:
    try:
        import yaml
        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("logging", {})
    except FileNotFoundError:
        print(
            f"[logger] config not found at {config_path!r}; "
            "using default logging (INFO console, no file).",
            file=sys.stderr,
        )
        return {}
    except Exception as exc:
        print(f"[logger] Failed to parse {config_path!r}: {exc}; using defaults.", file=sys.stderr)
        return {}


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Always call as ``get_logger(__name__)``."""
    return logging.getLogger(name)


@contextmanager
def temporarily_raise_console_log_level(level: int | str) -> Any:
    """Temporarily raise console handler levels, then restore them."""
    target_level = _level(level)
    root = logging.getLogger()
    console_handlers: list[tuple[logging.Handler, int]] = []
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream in (sys.stdout, sys.stderr):
                console_handlers.append((handler, handler.level))

    try:
        for handler, original_level in console_handlers:
            if original_level < target_level:
                handler.setLevel(target_level)
        yield
    finally:
        for handler, original_level in console_handlers:
            handler.setLevel(original_level)


class _LoggerPrefixFilter(logging.Filter):
    """Include or exclude records by logger-name prefix."""

    def __init__(self, prefixes: list[str], *, include: bool) -> None:
        super().__init__()
        self.prefixes = tuple(prefix for prefix in prefixes if prefix)
        self.include = include

    def filter(self, record: logging.LogRecord) -> bool:
        matched = any(
            record.name == prefix or record.name.startswith(f"{prefix}.")
            for prefix in self.prefixes
        )
        return matched if self.include else not matched


def setup_plugin_file_logging(
    plugin_name: str,
    *,
    logger_prefixes: list[str],
    config_path: str = "configs/config.yaml",
) -> Path | None:
    """Route selected logger prefixes to a plugin-specific file.

    Matching records are excluded from the main rotating file handler and
    instead written to a separate file whose name includes ``plugin_name``.
    Console logging remains unchanged.
    """
    cfg = _load_yaml_section(config_path)
    file_cfg: dict[str, Any] = cfg.get("file", {})
    if not file_cfg.get("enabled", False):
        return None

    plugin_name = str(plugin_name).strip()
    if not plugin_name:
        return None

    prefixes = [prefix for prefix in logger_prefixes if prefix]
    if not prefixes:
        return None

    root = logging.getLogger()
    main_log_path = Path(file_cfg.get("path", "logs/research.log"))
    plugin_log_path = _plugin_log_path(main_log_path, plugin_name)
    plugin_log_path.parent.mkdir(parents=True, exist_ok=True)

    exclude_filter = _LoggerPrefixFilter(prefixes, include=False)
    include_filter = _LoggerPrefixFilter(prefixes, include=True)

    for handler in root.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            base_filename = getattr(handler, "baseFilename", "")
            if Path(base_filename) == main_log_path.resolve():
                if not any(
                    isinstance(existing, _LoggerPrefixFilter)
                    and existing.include is False
                    and existing.prefixes == exclude_filter.prefixes
                    for existing in handler.filters
                ):
                    handler.addFilter(exclude_filter)

    handler_name = f"plugin-file:{plugin_name}"
    for handler in root.handlers:
        if getattr(handler, "name", "") == handler_name:
            return plugin_log_path

    plugin_handler = logging.handlers.RotatingFileHandler(
        filename=plugin_log_path,
        maxBytes=file_cfg.get("max_bytes", 10 * 1024 * 1024),
        backupCount=file_cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    plugin_handler.set_name(handler_name)
    plugin_handler.setLevel(_level(file_cfg.get("level", "DEBUG")))
    plugin_handler.setFormatter(
        logging.Formatter(
            fmt=file_cfg.get("format", _DEFAULT_FILE_FORMAT),
            datefmt=file_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        )
    )
    plugin_handler.addFilter(include_filter)
    root.addHandler(plugin_handler)
    return plugin_log_path


def _plugin_log_path(main_log_path: Path, plugin_name: str) -> Path:
    safe_plugin = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_"
        for ch in plugin_name.strip().lower()
    ) or "plugin"
    if main_log_path.suffix:
        return main_log_path.with_name(f"{main_log_path.stem}.{safe_plugin}{main_log_path.suffix}")
    return main_log_path.with_name(f"{main_log_path.name}.{safe_plugin}")
