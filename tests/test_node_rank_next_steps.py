from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.graph.nodes.rank_next_steps import rank_next_steps_node


PROFILE = {
    "name": "test",
    "evaluation": {"primary_metric": "test_auc"},
    "next_steps": {
        "max_selected": 2,
        "min_parent_metric": 0.7,
        "prior_run_similarity_threshold": 0.9,
        "prior_run_search_results": 5,
    },
    "prompts": {"rank_next_steps": {"system": "Rank the candidates."}},
}


def test_rank_next_steps_uses_llm_selected_order():
    state = {
        "research_direction": "current direction",
        "evaluation_summary": {"best_metric_value": 0.81},
        "next_steps": [
            {"title": "A", "suggested_direction": "Direction A", "priority": 2},
            {"title": "B", "suggested_direction": "Direction B", "priority": 1},
            {"title": "C", "suggested_direction": "Direction C", "priority": 3},
        ],
    }
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='[{"candidate_id":"step_2"},{"candidate_id":"step_1"}]'
    )
    memory_service = MagicMock()
    memory_service.query.return_value = []

    with patch("core.graph.nodes.rank_next_steps.MemoryService.for_profile", return_value=memory_service):
        with patch("core.graph.nodes.rank_next_steps.get_llm", return_value=llm):
            with patch("core.graph.nodes.rank_next_steps.persist_memory_records_for_state"):
                result = rank_next_steps_node(state, PROFILE)

    assert [item["title"] for item in result["next_steps"]] == ["B", "A"]
    assert result["next_step_selection"]["ranking_mode"] == "llm"
    assert [item["title"] for item in result["next_step_selection"]["selected"]] == ["B", "A"]
    assert result["next_step_selection"]["dropped"][0]["title"] == "C"
    assert result["next_step_selection"]["dropped"][0]["drop_reason"] == "not_selected_by_ranker"


def test_rank_next_steps_dedupes_and_drops_current_direction_before_ranking():
    state = {
        "research_direction": "Current Direction",
        "evaluation_summary": {"best_metric_value": 0.81},
        "next_steps": [
            {"title": "Current Direction", "suggested_direction": "Current Direction", "priority": 1},
            {"title": "A1", "suggested_direction": "Direction A", "priority": 2},
            {"title": "A2", "suggested_direction": "direction a", "priority": 1},
            {"title": "", "suggested_direction": "", "priority": 3},
            {"title": "B", "suggested_direction": "Direction B", "priority": 3},
        ],
    }
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='[{"candidate_id":"step_2"},{"candidate_id":"step_5"}]')
    memory_service = MagicMock()
    memory_service.query.return_value = []

    with patch("core.graph.nodes.rank_next_steps.MemoryService.for_profile", return_value=memory_service):
        with patch("core.graph.nodes.rank_next_steps.get_llm", return_value=llm):
            with patch("core.graph.nodes.rank_next_steps.persist_memory_records_for_state"):
                result = rank_next_steps_node(state, PROFILE)

    assert [item["title"] for item in result["next_steps"]] == ["A1", "B"]
    drop_reasons = {item["title"]: item["drop_reason"] for item in result["next_step_selection"]["dropped"]}
    assert drop_reasons["Current Direction"] == "same_as_current_direction"
    assert drop_reasons["A2"] == "duplicate_direction"


def test_rank_next_steps_falls_back_to_priority_order_when_llm_fails():
    state = {
        "research_direction": "current direction",
        "evaluation_summary": {"best_metric_value": 0.81},
        "next_steps": [
            {"title": "A", "suggested_direction": "Direction A", "priority": 2},
            {"title": "B", "suggested_direction": "Direction B", "priority": 1},
            {"title": "C", "suggested_direction": "Direction C", "priority": 3},
        ],
        "errors": [],
    }

    memory_service = MagicMock()
    memory_service.query.return_value = []

    with patch("core.graph.nodes.rank_next_steps.MemoryService.for_profile", return_value=memory_service):
        with patch("core.graph.nodes.rank_next_steps.get_llm", side_effect=RuntimeError("boom")):
            with patch("core.graph.nodes.rank_next_steps.persist_memory_records_for_state"):
                result = rank_next_steps_node(state, PROFILE)

    assert [item["title"] for item in result["next_steps"]] == ["B", "A"]
    assert any("rank_next_steps fallback" in error for error in result["errors"])
    assert result["next_step_selection"]["ranking_mode"] == "fallback"
    assert result["next_step_selection"]["dropped"][0]["title"] == "C"
    assert result["next_step_selection"]["dropped"][0]["drop_reason"] == "not_selected_by_fallback"


def test_rank_next_steps_drops_all_candidates_when_parent_metric_below_threshold():
    state = {
        "research_direction": "current direction",
        "evaluation_summary": {"best_metric_value": 0.68},
        "next_steps": [
            {"title": "A", "suggested_direction": "Direction A", "priority": 1},
        ],
    }

    with patch("core.graph.nodes.rank_next_steps.persist_memory_records_for_state"):
        result = rank_next_steps_node(state, PROFILE)

    assert result["next_steps"] == []
    assert result["next_step_selection"]["ranking_mode"] == "parent_metric_gate"
    assert result["next_step_selection"]["dropped"][0]["drop_reason"] == "parent_metric_below_threshold:test_auc<0.7"


def test_rank_next_steps_drops_candidates_similar_to_prior_runs():
    state = {
        "research_direction": "current direction",
        "evaluation_summary": {"best_metric_value": 0.81},
        "root_run_family_id": "family-1",
        "next_steps": [
            {"title": "A", "suggested_direction": "Direction A", "priority": 1},
            {"title": "B", "suggested_direction": "Direction B", "priority": 2},
        ],
    }
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='[{"candidate_id":"step_2"}]')
    memory_service = MagicMock()
    memory_service.query.side_effect = [
        [
            {
                "distance": 0.05,
                "record": {
                    "record_id": "prior-1",
                    "title": "Similar prior",
                    "metadata": {"root_run_family_id": "family-0", "test_auc": 0.83},
                    "content": {"metrics": {"test_auc": 0.83}},
                },
            }
        ],
        [],
    ]

    with patch("core.graph.nodes.rank_next_steps.MemoryService.for_profile", return_value=memory_service):
        with patch("core.graph.nodes.rank_next_steps.get_llm", return_value=llm):
            with patch("core.graph.nodes.rank_next_steps.persist_memory_records_for_state"):
                result = rank_next_steps_node(state, PROFILE)

    assert [item["title"] for item in result["next_steps"]] == ["B"]
    dropped = result["next_step_selection"]["dropped"]
    assert dropped[0]["title"] == "A"
    assert dropped[0]["drop_reason"] == "similar_to_prior_run"
