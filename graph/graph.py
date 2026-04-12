"""LangGraph graph definition — fully wired pipeline with follow-up loop.

Flow:
    research
        → feature_proposal
            → ns_experiment_runner   (dataset caching + S1 training + MLflow + ChromaDB + MongoDB)
                ──(success)──→ followup_proposal
                                    ──(continue_loop)──→ feature_proposal (loop)
                                    ──(done)──────────→ END
                ──(failure)──→ END

Legacy nodes (code_generation, experiment_runner, result_analysis, mlflow_logger,
generalization_eval, db_logger) are retained in graph/nodes/ for reference but are
not wired into the primary graph.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import (
    feature_proposal_node,
    followup_proposal_node,
    ns_experiment_runner_node,
    research_node,
)
from graph.state import ResearchState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_runner(state: ResearchState) -> str:
    """Route to followup_proposal if ≥1 experiment succeeded, otherwise END."""
    if state.get("execution_success"):
        return "followup_proposal"
    return END


def _route_after_followup(state: ResearchState) -> str:
    """Route back to feature_proposal for the next loop iteration, or END.

    When continue_loop is True the top follow-up proposal's suggested_direction
    has already been promoted to state['research_direction'] by followup_proposal_node.
    """
    if state.get("continue_loop"):
        return "feature_proposal"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> CompiledStateGraph:
    """Assemble and compile the research pipeline graph.

    Returns:
        A compiled LangGraph graph ready for invocation via graph.invoke(state).
    """
    graph: StateGraph = StateGraph(ResearchState)

    # --- Register nodes ---
    graph.add_node("research", research_node)
    graph.add_node("feature_proposal", feature_proposal_node)
    graph.add_node("ns_experiment_runner", ns_experiment_runner_node)
    graph.add_node("followup_proposal", followup_proposal_node)

    # --- Entry point ---
    graph.add_edge(START, "research")

    # --- Linear edges ---
    graph.add_edge("research", "feature_proposal")
    graph.add_edge("feature_proposal", "ns_experiment_runner")

    # ns_experiment_runner → followup_proposal  (success)
    #                      → END               (failure)
    graph.add_conditional_edges(
        "ns_experiment_runner",
        _route_after_runner,
        {
            "followup_proposal": "followup_proposal",
            END: END,
        },
    )

    # followup_proposal → feature_proposal  (continue_loop=True)
    #                   → END               (continue_loop=False / not set)
    graph.add_conditional_edges(
        "followup_proposal",
        _route_after_followup,
        {
            "feature_proposal": "feature_proposal",
            END: END,
        },
    )

    return graph.compile()
