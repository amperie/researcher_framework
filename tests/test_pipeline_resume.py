from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from core import pipeline_resume


@dataclass
class _InMemoryDocumentStore:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, record_id: str) -> dict[str, Any] | None:
        return self.records.get(record_id)

    def upsert(self, record: dict[str, Any]) -> None:
        self.records[record["record_id"]] = dict(record)


class _FakeService:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.document_store = _InMemoryDocumentStore()
        for record in records:
            self.document_store.upsert(record)

    def find_records(self, filters: dict[str, Any], *, limit: int = 50) -> list[dict[str, Any]]:
        matches = [record for record in self.document_store.records.values() if _matches(record, filters)]
        return matches[:limit]


def test_build_resume_state_rehydrates_implementation_inputs(minimal_profile, monkeypatch):
    records = [
        {
            "record_id": "proposal:1",
            "object_type": "proposal",
            "created_at": "2026-05-04T10:00:00+00:00",
            "content": {"name": "proposal-a", "dataset": "dataset-a", "detector": "detector-a"},
            "metadata": {
                "proposal_name": "proposal-a",
                "research_direction": "direction-a",
                "root_run_family_id": "family-1",
                "root_research_direction": "direction-a",
            },
        },
        {
            "record_id": "plan:1",
            "object_type": "implementation_plan",
            "created_at": "2026-05-04T10:05:00+00:00",
            "content": {"proposal_name": "proposal-a", "class_name": "DetectorA"},
            "metadata": {
                "proposal_name": "proposal-a",
                "research_direction": "direction-a",
            },
        },
        {
            "record_id": "impl:1",
            "object_type": "implementation",
            "object_role": "artifact",
            "created_at": "2026-05-04T10:10:00+00:00",
            "content": {"proposal_name": "proposal-a", "class_name": "DetectorA", "script_path": "impl.py"},
            "metadata": {
                "proposal_name": "proposal-a",
                "research_direction": "direction-a",
            },
        },
    ]
    service = _FakeService(records)
    monkeypatch.setattr(pipeline_resume.MemoryService, "for_profile", lambda _profile: service)

    state = pipeline_resume.build_resume_state(minimal_profile, "impl:1")

    assert state["research_direction"] == "direction-a"
    assert state["proposals"][0]["name"] == "proposal-a"
    assert state["implementation_plans"][0]["class_name"] == "DetectorA"
    assert state["implementations"][0]["script_path"] == "impl.py"


def test_build_resume_state_rehydrates_result_stage(minimal_profile, monkeypatch):
    run_record = {
        "record_id": "exp-1",
        "object_type": "experiment_result",
        "created_at": "2026-05-04T11:00:00+00:00",
        "content": {
            "experiment_id": "exp-1",
            "proposal_name": "proposal-a",
            "proposal": {"name": "proposal-a"},
            "metrics": {"test_auc": 0.81},
            "model": {"experiment_id": "exp-1", "mlflow_run_id": "run-1"},
            "evaluation_summary": {"best_proposal": "proposal-a"},
            "research_direction": "direction-a",
            "root_run_family_id": "family-1",
            "root_research_direction": "direction-a",
        },
        "metadata": {
            "experiment_id": "exp-1",
            "proposal_name": "proposal-a",
            "research_direction": "direction-a",
            "root_run_family_id": "family-1",
            "root_research_direction": "direction-a",
            "mlflow_run_id": "run-1",
        },
    }
    evaluation = {
        "record_id": "evaluation:1",
        "object_type": "evaluation_summary",
        "created_at": "2026-05-04T11:02:00+00:00",
        "content": {"best_proposal": "proposal-a", "n_experiments": 1},
        "metadata": {
            "proposal_name": "proposal-a",
            "research_direction": "direction-a",
        },
    }
    service = _FakeService([run_record, evaluation])
    monkeypatch.setattr(pipeline_resume.MemoryService, "for_profile", lambda _profile: service)

    state = pipeline_resume.build_resume_state(minimal_profile, "exp-1")

    assert state["experiment_results"][0]["experiment_id"] == "exp-1"
    assert state["models"][0]["mlflow_run_id"] == "run-1"
    assert state["evaluation_summary"]["best_proposal"] == "proposal-a"


def test_ensure_resume_state_for_node_requires_expected_keys():
    with pytest.raises(ValueError, match="missing required state keys: implementation_plans"):
        pipeline_resume.ensure_resume_state_for_node("implement", {"research_direction": "direction-a"})


def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if _get(record, key) != expected:
            return False
    return True


def _get(record: dict[str, Any], dotted: str) -> Any:
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
