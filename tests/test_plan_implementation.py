from __future__ import annotations

from core.graph.nodes.plan_implementation import _normalize_implementation_plans


def test_normalize_implementation_plans_drops_string_entries_and_backfills_names():
    raw = [
        {"base_class": "FeatureSetBase"},
        "this should have been an object",
        {"proposal_name": "proposal_c", "class_name": "ProposalC"},
    ]
    proposals = [
        {"name": "proposal_a"},
        {"name": "proposal_b"},
        {"name": "proposal_c"},
    ]

    plans, errors = _normalize_implementation_plans(raw, proposals)

    assert len(plans) == 2
    assert plans[0]["proposal_name"] == "proposal_a"
    assert plans[0]["class_name"] == "ProposalA"
    assert plans[1]["proposal_name"] == "proposal_c"
    assert any("proposal_b" in error for error in errors)
