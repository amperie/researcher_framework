"""Evaluate step — analyse experiment and model results against profile metrics.

Reads:
    state['experiment_results']
    state['models']

Writes:
    state['evaluation_summary']
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils import extract_json_object
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)


def evaluate_node(state: ResearchState, profile: dict) -> dict:
    experiment_results = state.get("experiment_results") or []
    models = state.get("models") or []
    direction = state.get("research_direction", "")

    if not experiment_results and not models:
        log.warning("evaluate_node | No results to evaluate")
        return {"evaluation_summary": {}}

    system_prompt = get_prompt(profile, "evaluate")
    llm = get_llm("evaluate", profile)

    evaluation_cfg = profile.get("evaluation") or {}
    primary_metric = evaluation_cfg.get("primary_metric", "")
    thresholds = evaluation_cfg.get("thresholds", {})

    # Build a consolidated results block for the LLM
    results_for_llm = []
    model_by_experiment = {m.get("experiment_id"): m for m in models}

    for result in experiment_results:
        entry = {
            "proposal_name": result.get("proposal_name"),
            "experiment_id": result.get("experiment_id"),
            "metrics": result.get("metrics", {}),
            "feature_importance": result.get("feature_importance", {}),
        }
        model = model_by_experiment.get(result.get("experiment_id"))
        if model:
            entry["model_metrics"] = model.get("metrics", {})
        results_for_llm.append(entry)

    # Also compute best result programmatically for the summary
    best_result = None
    best_value = -float("inf")
    for result in experiment_results:
        metrics = result.get("metrics") or {}
        value = float(metrics.get(primary_metric, 0.0))
        if value > best_value:
            best_value = value
            best_result = result

    user_content = (
        f"Research direction: {direction}\n\n"
        f"Primary metric: {primary_metric}\n"
        f"Thresholds: {json.dumps(thresholds)}\n\n"
        f"Results:\n{json.dumps(results_for_llm, indent=2, default=str)}"
    )

    log.info(
        "evaluate_node | Evaluating %d results — primary_metric=%r, best=%.4f",
        len(experiment_results), primary_metric, best_value,
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        analysis = extract_json_object(resp.content)
    except Exception as exc:
        log.error("evaluate_node | LLM analysis failed: %s", exc, exc_info=True)
        analysis = {}

    summary = {
        "best_metric_value": best_value if best_result else None,
        "best_metric_name": primary_metric,
        "best_proposal": best_result.get("proposal_name") if best_result else None,
        "n_experiments": len(experiment_results),
        "llm_analysis": analysis,
    }

    return {"evaluation_summary": summary}
