"""Tests for graph/nodes/implement.py."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.graph.nodes.implement import _proposal_execution_context, implement_node


def _profile():
    return {
        "name": "neuralsignal",
        "datasets": [],
        "base_classes": [],
        "prompts": {"implement": {"system": "Write code."}},
    }


def test_implement_registers_artifact_and_persists_memory(tmp_path):
    cfg = SimpleNamespace(experiments_dir=str(tmp_path))
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="class MyFeature:\n    pass\n")
    store = MagicMock()
    store.store_file.return_value = {
        "artifact_id": "impl-1",
        "uri": "s3://bucket/impl.py",
        "storage_key": "neuralsignal/implementation/impl-1/impl.py",
        "storage_bucket": "researcher-artifacts",
        "storage_endpoint_url": "http://hp.lan:9000",
    }
    state = {
        "profile_name": "neuralsignal",
        "implementation_plans": [{"proposal_name": "idea_a", "class_name": "MyFeature"}],
    }

    with patch("core.graph.nodes.implement.get_config", return_value=cfg):
        with patch("core.graph.nodes.implement.get_llm", return_value=llm):
            with patch("core.graph.nodes.artifact_refs.get_artifact_store", return_value=store):
                with patch("core.graph.nodes.implement.persist_memory_records_for_state") as persist_memory:
                    result = implement_node(state, _profile())

    implementation = result["implementations"][0]
    assert implementation["stored_artifact_id"] == "impl-1"
    assert implementation["stored_artifact_uri"] == "s3://bucket/impl.py"
    assert implementation["stored_artifact_bucket"] == "researcher-artifacts"
    assert implementation["stored_artifact_key"] == "neuralsignal/implementation/impl-1/impl.py"
    persist_memory.assert_called_once()


def test_implement_artifact_failure_is_non_fatal(tmp_path):
    cfg = SimpleNamespace(experiments_dir=str(tmp_path))
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="class MyFeature:\n    pass\n")
    store = MagicMock()
    store.store_file.side_effect = RuntimeError("s3 down")
    state = {
        "profile_name": "neuralsignal",
        "implementation_plans": [{"proposal_name": "idea_a", "class_name": "MyFeature"}],
    }

    with patch("core.graph.nodes.implement.get_config", return_value=cfg):
        with patch("core.graph.nodes.implement.get_llm", return_value=llm):
            with patch("core.graph.nodes.artifact_refs.get_artifact_store", return_value=store):
                result = implement_node(state, _profile())

    implementation = result["implementations"][0]
    assert "stored_artifact_id" not in implementation
    assert any("artifact_store: implementation MyFeature failed" in error for error in result["errors"])


def test_proposal_execution_context_prefers_declared_symbols():
    state = {
        "proposals": [
            {
                "name": "idea_a",
                "symbol": "SPY",
                "symbols": ["SPY", "UPRO"],
                "timeframe": "5min",
                "data_source": "alpaca",
                "mode": "backtest",
            }
        ]
    }

    context = json.loads(_proposal_execution_context(state, "idea_a"))

    assert context["proposal_name"] == "idea_a"
    assert context["expected_symbols"] == ["SPY", "UPRO"]
    assert any("cfg/proposal-facing fields" in line for line in context["guidance"])


def test_proposal_execution_context_handles_missing_proposal():
    state = {"proposals": []}

    context = _proposal_execution_context(state, "missing")

    assert "No matching proposal found in state" in context
