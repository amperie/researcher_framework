from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.graph.nodes.propose_next_steps import propose_next_steps_node


def test_propose_next_steps_persists_generated_steps(minimal_profile):
    state = {
        "research_direction": "current direction",
        "evaluation_summary": {"best_metric_value": 0.81},
        "experiment_results": [{"proposal_name": "proposal-a", "metrics": {"test_auc": 0.81}}],
    }
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='[{"title":"A","suggested_direction":"Direction A","priority":"high"}]'
    )

    with patch("core.graph.nodes.propose_next_steps.get_llm", return_value=llm):
        with patch("core.graph.nodes.propose_next_steps.persist_memory_records_for_state") as persist_memory:
            result = propose_next_steps_node(state, minimal_profile)

    assert result["next_steps"][0]["title"] == "A"
    persist_memory.assert_called_once()
    persisted_state = persist_memory.call_args.args[1]
    assert persisted_state["next_steps"] == result["next_steps"]
