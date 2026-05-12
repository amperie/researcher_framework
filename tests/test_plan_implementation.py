from __future__ import annotations

from core.graph.nodes.plan_implementation import _coerce_plan_string, _normalize_implementation_plans


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


def test_normalize_implementation_plans_accepts_stringified_json_objects():
    raw = [
        '{"proposal_name":"proposal_a","base_class":"Algorithm","main_method":"on_data_logic"}',
    ]
    proposals = [{"name": "proposal_a"}]

    plans, errors = _normalize_implementation_plans(raw, proposals)

    assert len(plans) == 1
    assert plans[0]["proposal_name"] == "proposal_a"
    assert plans[0]["class_name"] == "ProposalA"
    assert errors == []


def test_coerce_plan_string_accepts_fenced_json():
    parsed = _coerce_plan_string(
        '```json\n{"proposal_name":"proposal_a","class_name":"ProposalA"}\n```'
    )

    assert parsed == {"proposal_name": "proposal_a", "class_name": "ProposalA"}
