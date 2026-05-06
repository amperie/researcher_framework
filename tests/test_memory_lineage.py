from __future__ import annotations

from core.memory.defaults import (
    build_experiment_memory_records,
    build_next_step_memory_records,
    build_proposal_memory_records,
    next_step_record_id,
)


def test_next_step_record_id_matches_built_record(minimal_profile):
    step = {
        "title": "Probe sparse residual dynamics",
        "suggested_direction": "probe sparse residual dynamics",
        "priority": "high",
    }
    state = {
        "research_direction": "current direction",
        "next_steps": [step],
    }

    records = build_next_step_memory_records(minimal_profile, state)

    assert records[0]["record_id"] == next_step_record_id("test_profile", "current direction", step)


def test_proposal_records_include_next_step_lineage(minimal_profile):
    state = {
        "research_direction": "new direction",
        "source_next_step_record_id": "next_step:abc",
        "source_next_step_title": "Try the residual path idea",
        "campaign_id": "campaign-1",
        "campaign_title": "Residual entropy sweep",
        "campaign_variant_id": "variant-a",
        "campaign_variant_title": "Variant A",
        "proposals": [{
            "name": "proposal_a",
            "dataset": "test_dataset",
            "detector": "hallucination",
            "description": "desc",
        }],
    }

    record = build_proposal_memory_records(minimal_profile, state)[0]

    assert record["metadata"]["source_next_step_record_id"] == "next_step:abc"
    assert any(
        relation["relation_type"] == "inspires_proposal"
        and relation["source_type"] == "next_step"
        and relation["source_key"] == "Try the residual path idea"
        and relation["target_type"] == "proposal"
        and relation["target_key"] == "proposal_a"
        for relation in record["relations"]
    )
    assert any(
        relation["relation_type"] == "campaign_includes"
        and relation["source_type"] == "campaign"
        and relation["target_key"] == "proposal_a"
        for relation in record["relations"]
    )


def test_experiment_records_include_proposal_and_next_step_lineage(minimal_profile):
    state = {
        "research_direction": "new direction",
        "source_next_step_record_id": "next_step:abc",
        "source_next_step_title": "Try the residual path idea",
        "campaign_id": "campaign-1",
        "campaign_title": "Residual entropy sweep",
        "campaign_variant_id": "variant-a",
        "campaign_variant_title": "Variant A",
        "experiment_results": [{
            "experiment_id": "exp-123",
            "proposal_name": "proposal_a",
            "proposal": {
                "dataset": "test_dataset",
                "detector": "hallucination",
            },
            "metrics": {"test_auc": 0.71},
        }],
        "models": [],
        "evaluation_summary": {},
    }

    record = build_experiment_memory_records(minimal_profile, state)[0]

    assert record["metadata"]["source_next_step_record_id"] == "next_step:abc"
    assert any(
        relation["relation_type"] == "executed_as"
        and relation["source_type"] == "proposal"
        and relation["source_key"] == "proposal_a"
        and relation["target_type"] == "experiment_result"
        and relation["target_key"] == "exp-123"
        for relation in record["relations"]
    )
    assert any(
        relation["relation_type"] == "inspires_proposal"
        and relation["source_type"] == "next_step"
        and relation["target_key"] == "proposal_a"
        for relation in record["relations"]
    )
    assert any(
        relation["relation_type"] == "campaign_runs"
        and relation["source_type"] == "campaign"
        and relation["target_key"] == "exp-123"
        for relation in record["relations"]
    )
    assert record["metadata"]["campaign_id"] == "campaign-1"
