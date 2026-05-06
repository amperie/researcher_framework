from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import handoffs


@dataclass
class _InMemoryDocumentStore:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def upsert(self, record: dict[str, Any]) -> None:
        self.records[record["record_id"]] = dict(record)

    def get(self, record_id: str) -> dict[str, Any] | None:
        return self.records.get(record_id)

    def find(self, filters: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
        return [record for record in self.records.values() if _matches(record, filters)][:limit]


class _FakeService:
    def __init__(self) -> None:
        self.document_store = _InMemoryDocumentStore()

    def persist_record(self, record: dict[str, Any]) -> None:
        self.document_store.upsert(record)

    def find_records(self, filters: dict[str, Any], *, limit: int = 50) -> list[dict[str, Any]]:
        return self.document_store.find(filters, limit=limit)


def test_save_run_handoff_persists_linked_record(minimal_profile, monkeypatch):
    service = _FakeService()
    monkeypatch.setattr(handoffs.MemoryService, "for_profile", lambda _profile: service)
    source_record = {
        "record_id": "exp-1",
        "title": "proposal-a",
        "metadata": {
            "experiment_id": "exp-1",
            "proposal_name": "proposal-a",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "assessment": "promising",
        },
    }

    record = handoffs.save_run_handoff(
        minimal_profile,
        source_record,
        {
            "title": "Follow-up: proposal-a",
            "launch_direction": "explore a stronger detector",
            "suggested_direction": "explore a stronger detector",
            "rationale": "The prior run was promising but underpowered.",
            "snippets": [{"title": "Direction", "body": "initial direction", "selected": True}],
        },
    )

    assert record["object_type"] == "run_handoff"
    assert record["metadata"]["source_experiment_record_id"] == "exp-1"
    assert service.document_store.get(record["record_id"])["content"]["launch_direction"] == "explore a stronger detector"


def test_resolve_run_handoff_seed_uses_latest_saved_handoff(minimal_profile, monkeypatch):
    service = _FakeService()
    source_record = {
        "record_id": "exp-1",
        "title": "proposal-a",
        "object_type": "experiment_result",
        "metadata": {
            "experiment_id": "exp-1",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    }
    older = {
        "record_id": "run_handoff:older",
        "title": "Older",
        "object_type": "run_handoff",
        "content": {"launch_direction": "older direction"},
        "metadata": {
            "source_experiment_record_id": "exp-1",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "updated_at": "2026-05-02T09:00:00+00:00",
        },
    }
    newer = {
        "record_id": "run_handoff:newer",
        "title": "Newer",
        "object_type": "run_handoff",
        "content": {"launch_direction": "new direction"},
        "metadata": {
            "source_experiment_record_id": "exp-1",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "updated_at": "2026-05-02T10:00:00+00:00",
        },
    }
    for record in (source_record, older, newer):
        service.document_store.upsert(record)
    monkeypatch.setattr(handoffs.MemoryService, "for_profile", lambda _profile: service)

    seed = handoffs.resolve_run_handoff_seed(minimal_profile, source_experiment_record_id="exp-1")

    assert seed["research_direction"] == "new direction"
    assert seed["source_next_step_record_id"] == "run_handoff:newer"
    assert seed["root_run_family_id"] == "family-1"


def test_resolve_next_step_seed_uses_persisted_next_step(minimal_profile, monkeypatch):
    service = _FakeService()
    next_step = {
        "record_id": "next_step:1",
        "title": "Investigate stronger baseline",
        "object_type": "next_step",
        "content": {
            "title": "Investigate stronger baseline",
            "suggested_direction": "Compare against a stronger baseline detector",
            "rationale": "Current gains may collapse against a stronger control.",
        },
        "metadata": {
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    }
    service.document_store.upsert(next_step)
    monkeypatch.setattr(handoffs.MemoryService, "for_profile", lambda _profile: service)

    seed = handoffs.resolve_next_step_seed(minimal_profile, next_step_record_id="next_step:1")

    assert seed["research_direction"] == "Compare against a stronger baseline detector"
    assert seed["source_next_step_record_id"] == "next_step:1"
    assert seed["source_next_step_title"] == "Investigate stronger baseline"
    assert seed["root_run_family_id"] == "family-1"


def test_save_proposal_seed_persists_linked_record(minimal_profile, monkeypatch):
    service = _FakeService()
    monkeypatch.setattr(handoffs.MemoryService, "for_profile", lambda _profile: service)
    source_record = {
        "record_id": "exp-1",
        "title": "proposal-a",
        "metadata": {
            "experiment_id": "exp-1",
            "proposal_name": "proposal-a",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    }

    record = handoffs.save_proposal_seed(
        minimal_profile,
        source_record,
        {
            "title": "Proposal Seed: proposal-a",
            "research_direction": "initial direction",
            "proposal_template": {"name": "proposal-a", "dataset": "dataset-a"},
            "planning_notes": "Keep the first implementation minimal.",
            "snippets": [{"title": "Proposal", "body": "proposal context", "selected": True}],
            "campaign_id": "campaign-1",
            "campaign_title": "Residual entropy sweep",
            "campaign_variant_id": "variant-a",
            "campaign_variant_title": "Variant A",
            "campaign_variant_index": 1,
            "campaign_size": 25,
        },
    )

    assert record["object_type"] == "proposal_seed"
    assert record["metadata"]["source_experiment_record_id"] == "exp-1"
    assert service.document_store.get(record["record_id"])["content"]["proposal_template"]["name"] == "proposal-a"
    assert record["metadata"]["campaign_id"] == "campaign-1"
    assert record["content"]["campaign_variant_title"] == "Variant A"


def test_resolve_proposal_seed_uses_latest_saved_seed(minimal_profile, monkeypatch):
    service = _FakeService()
    source_record = {
        "record_id": "exp-1",
        "title": "proposal-a",
        "object_type": "experiment_result",
        "metadata": {
            "experiment_id": "exp-1",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    }
    older = {
        "record_id": "proposal_seed:older",
        "title": "Older",
        "object_type": "proposal_seed",
        "content": {"proposal_template": {"name": "older-proposal"}},
        "metadata": {
            "source_experiment_record_id": "exp-1",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "updated_at": "2026-05-02T09:00:00+00:00",
        },
    }
    newer = {
        "record_id": "proposal_seed:newer",
        "title": "Newer",
        "object_type": "proposal_seed",
        "content": {
            "research_direction": "initial direction",
            "proposal_template": {"name": "new-proposal"},
            "planning_notes": "Use the operator-authored plan.",
        },
        "metadata": {
            "source_experiment_record_id": "exp-1",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "updated_at": "2026-05-02T10:00:00+00:00",
        },
    }
    for record in (source_record, older, newer):
        service.document_store.upsert(record)
    monkeypatch.setattr(handoffs.MemoryService, "for_profile", lambda _profile: service)

    seed = handoffs.resolve_proposal_seed(minimal_profile, source_experiment_record_id="exp-1")

    assert seed["proposals"][0]["name"] == "new-proposal"
    assert seed["source_proposal_seed_record_id"] == "proposal_seed:newer"
    assert seed["proposal_seed_planning_notes"] == "Use the operator-authored plan."


def test_resolve_proposal_seed_carries_campaign_metadata(minimal_profile, monkeypatch):
    service = _FakeService()
    source_record = {
        "record_id": "exp-1",
        "title": "proposal-a",
        "object_type": "experiment_result",
        "metadata": {
            "experiment_id": "exp-1",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
        },
    }
    seeded = {
        "record_id": "proposal_seed:campaign",
        "title": "Campaign seed",
        "object_type": "proposal_seed",
        "content": {
            "research_direction": "initial direction",
            "proposal_template": {"name": "campaign-proposal"},
            "campaign_id": "campaign-1",
            "campaign_title": "Residual entropy sweep",
            "campaign_variant_id": "variant-07",
            "campaign_variant_title": "Variant 07",
            "campaign_variant_index": 7,
            "campaign_size": 25,
        },
        "metadata": {
            "source_experiment_record_id": "exp-1",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "campaign_id": "campaign-1",
            "campaign_title": "Residual entropy sweep",
            "campaign_variant_id": "variant-07",
            "campaign_variant_title": "Variant 07",
            "campaign_variant_index": 7,
            "campaign_size": 25,
            "updated_at": "2026-05-02T10:00:00+00:00",
        },
    }
    for record in (source_record, seeded):
        service.document_store.upsert(record)
    monkeypatch.setattr(handoffs.MemoryService, "for_profile", lambda _profile: service)

    seed = handoffs.resolve_proposal_seed(
        minimal_profile,
        source_experiment_record_id="exp-1",
        proposal_seed_record_id="proposal_seed:campaign",
    )

    assert seed["campaign_id"] == "campaign-1"
    assert seed["campaign_title"] == "Residual entropy sweep"
    assert seed["campaign_variant_id"] == "variant-07"
    assert seed["campaign_variant_index"] == 7
    assert seed["campaign_size"] == 25


def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = _get(record, key)
        if actual != expected:
            return False
    return True


def _get(record: dict[str, Any], dotted: str) -> Any:
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
