from __future__ import annotations

import logging
import sys

from core.utils.logger import _IgnoreLoggerLevelsFilter, temporarily_raise_console_log_level


def _record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1, msg="msg", args=(), exc_info=None)


def test_ignore_logger_levels_filter_suppresses_info_but_keeps_warning_and_error():
    flt = _IgnoreLoggerLevelsFilter([
        {"name": "pymongo", "max_level": "INFO"},
        {"name": "neo4j", "max_level": "INFO"},
        {"name": "chromadb", "max_level": "INFO"},
    ])

    assert flt.filter(_record("pymongo.command", logging.DEBUG)) is False
    assert flt.filter(_record("neo4j.io", logging.INFO)) is False
    assert flt.filter(_record("chromadb.telemetry", logging.WARNING)) is True
    assert flt.filter(_record("pymongo.network", logging.ERROR)) is True
    assert flt.filter(_record("core.graph.nodes.execute_experiment", logging.INFO)) is True


def test_temporarily_raise_console_log_level_restores_original_levels():
    root = logging.getLogger("test_logger_restore")
    old_propagate = root.propagate
    old_handlers = list(root.handlers)
    root.handlers.clear()
    root.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    root.addHandler(stdout_handler)

    other_handler = logging.StreamHandler()
    other_handler.setLevel(logging.ERROR)
    root.addHandler(other_handler)

    original_root = logging.getLogger
    try:
        logging.getLogger = lambda name=None: root if name is None else original_root(name)
        with temporarily_raise_console_log_level("WARNING"):
            assert stdout_handler.level == logging.WARNING
            assert other_handler.level == logging.ERROR
        assert stdout_handler.level == logging.INFO
        assert other_handler.level == logging.ERROR
    finally:
        logging.getLogger = original_root
        root.handlers.clear()
        root.handlers.extend(old_handlers)
        root.propagate = old_propagate
