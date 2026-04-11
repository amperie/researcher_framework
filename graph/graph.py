"""LangGraph graph definition — fully wired pipeline with follow-up loop.

Flow:
    research
        → feature_proposal
            → code_generation
                → experiment_runner
                    ──(success)──→ result_analysis → mlflow_logger
                                                          → generalization_eval
                                                              → db_logger
                                                                  → followup_proposal
                                                                      ──(continue_loop)──→ feature_proposal (loop)
                                                                      ──(done)──────────→ END
                    ──(failure)──→ END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import (
    code_generation_node,
    db_logger_node,
    experiment_runner_node,
    feature_proposal_node,
    followup_proposal_node,
    generalization_eval_node,
    mlflow_logger_node,
    research_node,
    result_analysis_node,
)
from graph.state import ResearchState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_runner(state: ResearchState) -> str:
    """Route to result_analysis if the experiment succeeded, otherwise END."""
    if state.get("execution_success"):
        return "result_analysis"
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
    graph.add_node("code_generation", code_generation_node)
    graph.add_node("experiment_runner", experiment_runner_node)
    graph.add_node("result_analysis", result_analysis_node)
    graph.add_node("mlflow_logger", mlflow_logger_node)
    graph.add_node("generalization_eval", generalization_eval_node)
    graph.add_node("db_logger", db_logger_node)
    graph.add_node("followup_proposal", followup_proposal_node)

    # --- Entry point ---
    graph.add_edge(START, "research")

    # --- Linear edges ---
    graph.add_edge("research", "feature_proposal")
    graph.add_edge("feature_proposal", "code_generation")
    graph.add_edge("code_generation", "experiment_runner")

    # experiment_runner → result_analysis  (success)
    #                   → END              (failure)
    graph.add_conditional_edges(
        "experiment_runner",
        _route_after_runner,
        {
            "result_analysis": "result_analysis",
            END: END,
        },
    )

    graph.add_edge("result_analysis", "mlflow_logger")
    graph.add_edge("mlflow_logger", "generalization_eval")
    graph.add_edge("generalization_eval", "db_logger")
    graph.add_edge("db_logger", "followup_proposal")

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
