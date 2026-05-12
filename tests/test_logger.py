from __future__ import annotations

import logging

from core.utils.logger import _IgnoreLoggerLevelsFilter


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
