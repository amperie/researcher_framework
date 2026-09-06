from __future__ import annotations

from types import SimpleNamespace

from web import service


class _FakeMemoryService:
    def __init__(self, records=None) -> None:
        self._records = list(records or [])
        self.document_store = SimpleNamespace(get=lambda record_id: None)

    def find_records(self, filters, limit=50):
        object_type = filters.get("object_type")
        if object_type == "experiment_result":
            return [record for record in self._records if str(record.get("object_type") or "") == "experiment_result"][:limit]
        if object_type == "pipeline_run":
            return [record for record in self._records if str(record.get("object_type") or "") == "pipeline_run"][:limit]
        if object_type == "run_handoff":
            source_id = filters.get("metadata.source_experiment_record_id")
            return [
                record for record in self._records
                if str(record.get("object_type") or "") == "run_handoff"
                and ((record.get("metadata") or {}).get("source_experiment_record_id") == source_id)
            ][:limit]
        if object_type == "proposal_seed":
            source_id = filters.get("metadata.source_experiment_record_id")
            return [
                record for record in self._records
                if str(record.get("object_type") or "") == "proposal_seed"
                and ((record.get("metadata") or {}).get("source_experiment_record_id") == source_id)
            ][:limit]
        proposal_name = filters.get("metadata.proposal_name")
        research_direction = filters.get("metadata.research_direction")
        if proposal_name or research_direction:
            return [
                record for record in self._records
                if (
                    (proposal_name and ((record.get("metadata") or {}).get("proposal_name") == proposal_name))
                    or (research_direction and ((record.get("metadata") or {}).get("research_direction") == research_direction))
                )
            ][:limit]
        family_id = filters.get("metadata.root_run_family_id")
        if family_id:
            return [
                record for record in self._records
                if ((record.get("metadata") or {}).get("root_run_family_id") == family_id)
            ][:limit]
        return []


class _FakeCollection:
    def __init__(self, docs) -> None:
        self._docs = list(docs)
        self._current = list(docs)

    def find(self, query=None, *_args, **_kwargs):
        if query and query.get("root_run_family_id"):
            family_id = query.get("root_run_family_id")
            self._current = [doc for doc in self._docs if doc.get("root_run_family_id") == family_id]
        else:
            self._current = list(self._docs)
        return self

    def sort(self, field, direction):
        reverse = direction == -1
        self._current = sorted(self._current, key=lambda item: item.get(field, ""), reverse=reverse)
        return self

    def limit(self, count):
        self._current = self._current[:count]
        return self

    def find_one(self, query, *_args, **_kwargs):
        for option in query.get("$or", []):
            for key, value in option.items():
                for doc in self._docs:
                    if doc.get(key) == value:
                        return dict(doc)
        return None

    def __iter__(self):
        return iter(self._current)


class _FakeMongoClient:
    def __init__(self, docs_by_db_collection) -> None:
        self._docs = docs_by_db_collection

    def __getitem__(self, db_name):
        return _FakeMongoDatabase(self._docs.get(db_name, {}))

    def close(self):
        return None


class _FakeMongoDatabase:
    def __init__(self, collections) -> None:
        self._collections = collections

    def __getitem__(self, collection_name):
        return _FakeCollection(self._collections.get(collection_name, []))


def test_list_run_summaries_falls_back_to_raw_results_collection(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {
            "mongodb_results_db": "researcher",
            "mongodb_results_collection": "trading_researcher_experiments",
        },
        "evaluation": {"primary_metric": "test_auc"},
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[]),
    )
    docs = {
        "researcher": {
            "trading_researcher_experiments": [{
                "experiment_id": "exp-1",
                "proposal_name": "proposal-a",
                "profile": "trading_researcher",
                "research_direction": "direction",
                "metrics": {"test_auc": 0.91},
                "evaluation_summary": {
                    "per_proposal_analysis": {
                        "proposal-a": {"assessment": "promising"},
                    }
                },
                "inserted_at": "2026-05-01T20:00:00+00:00",
                "mlflow_run_id": "mlf-1",
            }],
        }
    }

    monkeypatch.setattr(service, "load_profile_contexts", lambda: [context])
    monkeypatch.setattr(service.pymongo, "MongoClient", lambda *_args, **_kwargs: _FakeMongoClient(docs))
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": "http://mlflow/run"})

    runs = service.list_run_summaries(profile_name="trading_researcher", limit=10)

    assert len(runs) == 1
    assert runs[0]["record_id"] == "exp-1"
    assert runs[0]["source"] == "results_mongo"
    assert runs[0]["assessment"] == "promising"
    assert runs[0]["primary_metric_value"] == 0.91


def test_get_run_bundle_falls_back_to_raw_results_collection(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {
            "mongodb_results_db": "researcher",
            "mongodb_results_collection": "trading_researcher_experiments",
            "artifacts_mongodb_db": "researcher",
            "artifacts_collection": "trading_researcher_artifacts",
        },
        "evaluation": {"primary_metric": "test_auc"},
    }
    docs = {
        "researcher": {
            "trading_researcher_experiments": [{
                "experiment_id": "exp-2",
                "proposal_name": "proposal-b",
                "profile": "trading_researcher",
                "research_direction": "direction",
                "metrics": {"test_auc": 0.88},
                "evaluation_summary": {
                    "per_proposal_analysis": {
                        "proposal-b": {
                            "assessment": "useful",
                            "hypothesis_supported": True,
                            "lessons": ["lesson-1"],
                        },
                    }
                },
                "inserted_at": "2026-05-01T21:00:00+00:00",
                "mlflow_run_id": "mlf-2",
            }],
            "trading_researcher_artifacts": [],
        }
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[]),
    )

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service.pymongo, "MongoClient", lambda *_args, **_kwargs: _FakeMongoClient(docs))
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": "http://mlflow/run"})
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})

    bundle = service.get_run_bundle("trading_researcher", "exp-2")

    assert bundle["run"]["record"]["record_id"] == "exp-2"
    assert bundle["run"]["summary"]["experiment_id"] == "exp-2"
    assert bundle["run"]["summary"]["assessment"] == "useful"
    assert bundle["run"]["summary"]["primary_metric_value"] == 0.88


def test_get_run_bundle_includes_run_family(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    records = [
        {
            "record_id": "exp-root",
            "title": "proposal-root",
            "object_type": "experiment_result",
            "created_at": "2026-05-01T20:00:00+00:00",
            "metadata": {
                "experiment_id": "exp-root",
                "proposal_name": "proposal-root",
                "assessment": "seed",
                "research_direction": "initial direction",
                "root_run_family_id": "family-1",
                "root_research_direction": "initial direction",
                "test_auc": 0.81,
            },
            "entities": [{"entity_type": "proposal", "key": "proposal-root", "name": "proposal-root"}],
            "relations": [],
        },
        {
            "record_id": "exp-child",
            "title": "proposal-child",
            "object_type": "experiment_result",
            "created_at": "2026-05-01T21:00:00+00:00",
            "metadata": {
                "experiment_id": "exp-child",
                "proposal_name": "proposal-child",
                "assessment": "follow-up",
                "research_direction": "child direction",
                "root_run_family_id": "family-1",
                "root_research_direction": "initial direction",
                "source_next_step_title": "Try child idea",
                "test_auc": 0.93,
            },
            "entities": [{"entity_type": "proposal", "key": "proposal-child", "name": "proposal-child"}],
            "relations": [{
                "relation_type": "inspires_proposal",
                "source_type": "next_step",
                "source_key": "Try child idea",
                "target_type": "proposal",
                "target_key": "proposal-child",
            }],
        },
    ]
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=records),
    )
    context.memory_service.document_store = SimpleNamespace(get=lambda record_id: records[0] if record_id == "exp-root" else None)

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})
    monkeypatch.setattr(service, "list_run_handoffs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [])

    bundle = service.get_run_bundle("trading_researcher", "exp-root")

    family = bundle["run"]["family"]
    assert family["family_id"] == "family-1"
    assert family["root_research_direction"] == "initial direction"
    assert [run["record_id"] for run in family["runs"]] == ["exp-root", "exp-child"]
    assert family["graph"]["nodes"] == [
        {"entity_type": "proposal", "key": "proposal-root", "name": "proposal-root"},
        {"entity_type": "proposal", "key": "proposal-child", "name": "proposal-child"},
    ]


def test_get_run_bundle_includes_handoff_draft_and_saved(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    run_record = {
        "record_id": "exp-root",
        "title": "proposal-root",
        "object_type": "experiment_result",
        "created_at": "2026-05-01T20:00:00+00:00",
        "metadata": {
            "experiment_id": "exp-root",
            "proposal_name": "proposal-root",
            "assessment": "seed",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "test_auc": 0.81,
        },
        "entities": [],
        "relations": [],
    }
    saved_handoff = {
        "record_id": "run_handoff:1",
        "title": "Follow-up: proposal-root",
        "object_type": "run_handoff",
        "content": {
            "title": "Follow-up: proposal-root",
            "launch_direction": "try the next thing",
            "suggested_direction": "try the next thing",
            "snippets": [{"title": "Direction", "body": "initial direction", "selected": True}],
            "prompt_preview": "Next direction: try the next thing",
        },
        "metadata": {
            "source_experiment_record_id": "exp-root",
            "updated_at": "2026-05-02T10:00:00+00:00",
        },
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[run_record, saved_handoff]),
    )
    context.memory_service.document_store = SimpleNamespace(get=lambda record_id: run_record if record_id == "exp-root" else None)

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})
    monkeypatch.setattr(service, "list_run_handoffs", lambda *_args, **_kwargs: [saved_handoff])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [])

    bundle = service.get_run_bundle("trading_researcher", "exp-root")

    handoff = bundle["run"]["handoff"]
    assert handoff["draft"]["source_experiment_record_id"] == "exp-root"
    assert handoff["saved"][0]["record_id"] == "run_handoff:1"
    assert "--source-experiment \"exp-root\"" in handoff["saved"][0]["copy_command"]


def test_get_run_bundle_includes_proposal_seed_draft_and_saved(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    run_record = {
        "record_id": "exp-root",
        "title": "proposal-root",
        "object_type": "experiment_result",
        "created_at": "2026-05-01T20:00:00+00:00",
        "content": {
            "proposal": {
                "name": "proposal-root",
                "description": "Try the proposal template",
                "dataset": "dataset-a",
                "detector": "detector-a",
            }
        },
        "metadata": {
            "experiment_id": "exp-root",
            "proposal_name": "proposal-root",
            "assessment": "seed",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "test_auc": 0.81,
        },
        "entities": [],
        "relations": [],
    }
    saved_seed = {
        "record_id": "proposal_seed:1",
        "title": "Proposal Seed: proposal-root",
        "object_type": "proposal_seed",
        "content": {
            "title": "Proposal Seed: proposal-root",
            "research_direction": "initial direction",
            "proposal_template": {"name": "proposal-root", "dataset": "dataset-a"},
            "planning_notes": "Keep it simple.",
            "snippets": [{"title": "Proposal", "body": "Name: proposal-root", "selected": True}],
            "prompt_preview": "Proposal template",
        },
        "metadata": {
            "source_experiment_record_id": "exp-root",
            "updated_at": "2026-05-02T10:00:00+00:00",
        },
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[run_record, saved_seed]),
    )
    context.memory_service.document_store = SimpleNamespace(get=lambda record_id: run_record if record_id == "exp-root" else None)

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})
    monkeypatch.setattr(service, "list_run_handoffs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [saved_seed])

    bundle = service.get_run_bundle("trading_researcher", "exp-root")

    proposal_seed = bundle["run"]["proposal_seed"]
    assert proposal_seed["draft"]["proposal_template"]["name"] == "proposal-root"
    assert proposal_seed["saved"][0]["record_id"] == "proposal_seed:1"
    assert "--proposal-seed \"proposal_seed:1\"" in proposal_seed["saved"][0]["copy_command"]


def test_get_run_bundle_includes_proposed_next_steps_panel(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    run_record = {
        "record_id": "exp-root",
        "title": "proposal-root",
        "object_type": "experiment_result",
        "created_at": "2026-05-01T20:00:00+00:00",
        "metadata": {
            "experiment_id": "exp-root",
            "proposal_name": "proposal-root",
            "assessment": "seed",
            "research_direction": "initial direction",
            "root_run_family_id": "family-1",
            "root_research_direction": "initial direction",
            "test_auc": 0.81,
        },
        "entities": [],
        "relations": [],
    }
    next_step_record = {
        "record_id": "next_step:1",
        "title": "Try residual signal probe",
        "object_type": "next_step",
        "kind": "next_step",
        "created_at": "2026-05-01T22:00:00+00:00",
        "content": {
            "title": "Try residual signal probe",
            "priority": "high",
            "suggested_direction": "Probe residual stream signals around hallucination onset",
            "rationale": "The first run showed promising but noisy separation.",
        },
        "metadata": {
            "research_direction": "initial direction",
        },
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[run_record, next_step_record]),
    )
    context.memory_service.document_store = SimpleNamespace(get=lambda record_id: run_record if record_id == "exp-root" else None)

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})
    monkeypatch.setattr(service, "list_run_handoffs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [])

    bundle = service.get_run_bundle("trading_researcher", "exp-root")

    panel = next(item for item in bundle["run"]["text_panels"] if item["title"] == "Proposed Next Steps")
    assert "Try residual signal probe" in panel["body"]
    assert "Priority: high" in panel["body"]
    assert "Suggested Direction: Probe residual stream signals around hallucination onset" in panel["body"]
    assert bundle["run"]["next_steps"][0]["record_id"] == "next_step:1"
    assert bundle["run"]["next_steps"][0]["copy_command"] == 'uv run python main.py --profile trading_researcher --next-step "next_step:1"'


def test_get_run_bundle_includes_all_original_ideas(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    run_record = {
        "record_id": "exp-root",
        "title": "proposal-root",
        "object_type": "experiment_result",
        "created_at": "2026-05-01T20:00:00+00:00",
        "metadata": {
            "experiment_id": "exp-root",
            "proposal_name": "proposal-root",
            "research_direction": "initial direction",
        },
        "entities": [],
        "relations": [],
    }
    idea_a = {
        "record_id": "idea:a",
        "title": "Original idea A",
        "object_type": "idea",
        "kind": "idea",
        "created_at": "2026-05-01T20:01:00+00:00",
        "content": {"description": "Probe signal A", "hypothesis": "A works"},
        "metadata": {"research_direction": "initial direction"},
    }
    idea_b = {
        "record_id": "idea:b",
        "title": "Original idea B",
        "object_type": "idea",
        "kind": "idea",
        "created_at": "2026-05-01T20:02:00+00:00",
        "content": {"description": "Probe signal B", "hypothesis": "B works"},
        "metadata": {"research_direction": "initial direction"},
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[run_record, idea_a, idea_b]),
    )
    context.memory_service.document_store = SimpleNamespace(get=lambda record_id: run_record if record_id == "exp-root" else None)

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})
    monkeypatch.setattr(service, "list_run_handoffs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [])

    bundle = service.get_run_bundle("trading_researcher", "exp-root")

    panel = next(item for item in bundle["run"]["text_panels"] if item["title"] == "Ideas")
    assert "Original idea A" in panel["body"]
    assert "Probe signal A" in panel["body"]
    assert "Original idea B" in panel["body"]
    assert "Probe signal B" in panel["body"]


def test_get_run_bundle_relates_ideas_by_run_family(monkeypatch):
    profile = {
        "name": "trading_researcher",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    run_record = {
        "record_id": "exp-root",
        "title": "proposal-root",
        "object_type": "experiment_result",
        "created_at": "2026-05-01T20:00:00+00:00",
        "metadata": {
            "experiment_id": "exp-root",
            "proposal_name": "proposal-root",
            "research_direction": "final execution direction",
            "root_run_family_id": "family-1",
        },
        "entities": [],
        "relations": [],
    }
    idea_record = {
        "record_id": "idea:a",
        "title": "Original family idea",
        "object_type": "idea",
        "kind": "idea",
        "created_at": "2026-05-01T20:01:00+00:00",
        "content": {"description": "This came from the original ideation pass"},
        "metadata": {
            "research_direction": "different seed wording",
            "root_run_family_id": "family-1",
        },
    }
    context = service.ProfileContext(
        name="trading_researcher",
        profile=profile,
        memory_service=_FakeMemoryService(records=[run_record, idea_record]),
    )
    context.memory_service.document_store = SimpleNamespace(get=lambda record_id: run_record if record_id == "exp-root" else None)

    monkeypatch.setattr(service, "_load_context", lambda _profile_name: context)
    monkeypatch.setattr(service, "_artifact_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_graph_bundle", lambda *_args, **_kwargs: {"backend_enabled": False, "nodes": [], "edges": []})
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})
    monkeypatch.setattr(service, "list_run_handoffs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "list_proposal_seeds", lambda *_args, **_kwargs: [])

    bundle = service.get_run_bundle("trading_researcher", "exp-root")

    panel = next(item for item in bundle["run"]["text_panels"] if item["title"] == "Ideas")
    assert "Original family idea" in panel["body"]


def test_list_run_summaries_includes_pipeline_run_without_experiment_result(monkeypatch):
    profile = {
        "name": "trading",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    pipeline_run = {
        "record_id": "pipeline_run:family-1",
        "title": "use topology to model market movements",
        "object_type": "pipeline_run",
        "created_at": "2026-05-27T19:29:10+00:00",
        "metadata": {
            "research_direction": "use topology to model market movements",
            "root_run_family_id": "family-1",
            "n_ideas": 5,
            "n_next_steps": 3,
        },
    }
    context = service.ProfileContext(
        name="trading",
        profile=profile,
        memory_service=_FakeMemoryService(records=[pipeline_run]),
    )

    monkeypatch.setattr(service, "load_profile_contexts", lambda: [context])
    monkeypatch.setattr(service, "_raw_result_summaries", lambda *_args, **_kwargs: [])

    runs = service.list_run_summaries(profile_name="trading", limit=10)

    assert runs[0]["record_id"] == "pipeline_run:family-1"
    assert runs[0]["source"] == "pipeline_memory"
    assert runs[0]["assessment"] == "planning"


def test_list_run_summaries_keeps_parent_pipeline_run_when_child_experiment_exists(monkeypatch):
    profile = {
        "name": "trading",
        "storage": {},
        "evaluation": {"primary_metric": "test_auc"},
    }
    child_experiment = {
        "record_id": "exp-child",
        "title": "child experiment",
        "object_type": "experiment_result",
        "created_at": "2026-05-27T22:00:00+00:00",
        "metadata": {
            "experiment_id": "exp-child",
            "proposal_name": "child experiment",
            "research_direction": "child direction",
            "root_run_family_id": "family-1",
        },
    }
    parent_pipeline = {
        "record_id": "pipeline_run:parent",
        "title": "parent topology direction",
        "object_type": "pipeline_run",
        "created_at": "2026-05-27T19:00:00+00:00",
        "metadata": {
            "research_direction": "parent topology direction",
            "root_run_family_id": "family-1",
        },
    }
    duplicate_child_pipeline = {
        "record_id": "pipeline_run:child",
        "title": "child direction",
        "object_type": "pipeline_run",
        "created_at": "2026-05-27T21:00:00+00:00",
        "metadata": {
            "research_direction": "child direction",
            "root_run_family_id": "family-1",
        },
    }
    context = service.ProfileContext(
        name="trading",
        profile=profile,
        memory_service=_FakeMemoryService(records=[child_experiment, parent_pipeline, duplicate_child_pipeline]),
    )

    monkeypatch.setattr(service, "load_profile_contexts", lambda: [context])
    monkeypatch.setattr(service, "_raw_result_summaries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_mlflow_bundle", lambda *_args, **_kwargs: {"ui_url": ""})

    runs = service.list_run_summaries(profile_name="trading", limit=10)
    record_ids = {run["record_id"] for run in runs}

    assert "exp-child" in record_ids
    assert "pipeline_run:parent" in record_ids
    assert "pipeline_run:child" not in record_ids


def test_create_brainstorm_session_accepts_source_experiment_seed(monkeypatch):
    profile = {"name": "trading_researcher"}
    captured: dict[str, object] = {}

    class _FakeEngine:
        def __init__(self, _profile, _cfg) -> None:
            return None

        def run_until_pause(self, state):
            state["status"] = "awaiting_user"
            state["last_summary"] = "seeded"
            return state

    monkeypatch.setattr(service, "load_profile", lambda _profile_name: profile)
    monkeypatch.setattr(service, "load_brainstorm_config", lambda _path=None: {"name": "cfg", "path": "cfg", "roles": []})
    monkeypatch.setattr(
        service,
        "resolve_brainstorm_seed",
        lambda *_args, **_kwargs: {"research_direction": "seeded direction", "source_experiment_record_id": "exp-1"},
    )
    monkeypatch.setattr(service, "BrainstormEngine", _FakeEngine)

    def _fake_create_state(**kwargs):
        captured.update(kwargs)
        return {
            "session_id": "brainstorm-1",
            "status": "running",
            "current_goal": kwargs["direction"],
            "last_summary": "",
        }

    monkeypatch.setattr(service, "create_brainstorm_state", _fake_create_state)
    monkeypatch.setattr(service, "persist_brainstorm_session", lambda *_args, **_kwargs: {})

    result = service.create_brainstorm_session("trading_researcher", {"source_experiment": "exp-1"})

    assert captured["direction"] == "seeded direction"
    assert captured["seed"]["source_experiment_record_id"] == "exp-1"
    assert result["status"] == "awaiting_user"
