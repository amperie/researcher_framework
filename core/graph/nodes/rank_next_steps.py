"""Rank and prune proposed follow-up research directions."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.graph.nodes.memory import persist_memory_records_for_state
from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.memory import MemoryService
from core.utils import extract_json_array
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)


def rank_next_steps_node(state: ResearchState, profile: dict) -> dict:
    cfg = profile.get("next_steps") or {}
    max_selected = int(cfg.get("max_selected") or 3)
    steps = list(state.get("next_steps") or [])
    primary_metric = str((profile.get("evaluation") or {}).get("primary_metric") or "primary_metric")
    metric_value = _float_or_none((state.get("evaluation_summary") or {}).get("best_metric_value"))
    min_metric = _float_or_none(cfg.get("min_parent_metric"))

    if min_metric is not None and (metric_value is None or metric_value < min_metric):
        return _finish(state, profile, [], _selection(
            [], [_drop(s, f"parent_metric_below_threshold:{primary_metric}<{min_metric}") for s in steps],
            "parent_metric_gate",
        ))

    candidates, dropped = _dedupe_candidates(steps, str(state.get("research_direction") or ""))
    candidates, prior_dropped = _drop_prior_similar(candidates, state, profile, cfg)
    dropped.extend(prior_dropped)

    try:
        selected = _rank_with_llm(candidates, state, profile, max_selected)
        mode = "llm"
    except Exception as exc:
        log.warning("rank_next_steps_node | Falling back to priority order: %s", exc)
        selected = sorted(candidates, key=lambda item: _priority(item["step"]))[:max_selected]
        mode = "fallback"
        state = {**state, "errors": (state.get("errors") or []) + [f"rank_next_steps fallback: {exc}"]}

    selected_ids = {item["candidate_id"] for item in selected}
    reason = "not_selected_by_fallback" if mode == "fallback" else "not_selected_by_ranker"
    dropped.extend(_drop(item["step"], reason) for item in candidates if item["candidate_id"] not in selected_ids)
    next_steps = [item["step"] for item in selected]
    return _finish(state, profile, next_steps, _selection(next_steps, dropped, mode))


def _dedupe_candidates(steps: list[dict], direction: str) -> tuple[list[dict], list[dict]]:
    seen: set[str] = set()
    candidates: list[dict] = []
    dropped: list[dict] = []
    current = _norm(direction)
    for index, step in enumerate(steps, 1):
        text = str(step.get("suggested_direction") or step.get("title") or "")
        key = _norm(text)
        if not key:
            dropped.append(_drop(step, "empty_direction"))
        elif key == current:
            dropped.append(_drop(step, "same_as_current_direction"))
        elif key in seen:
            dropped.append(_drop(step, "duplicate_direction"))
        else:
            seen.add(key)
            candidates.append({"candidate_id": f"step_{index}", "step": step})
    return candidates, dropped


def _drop_prior_similar(candidates: list[dict], state: ResearchState, profile: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    threshold = _float_or_none(cfg.get("prior_run_similarity_threshold"))
    if threshold is None:
        return candidates, []
    max_distance = 1.0 - threshold
    keep: list[dict] = []
    dropped: list[dict] = []
    service = MemoryService.for_profile(profile)
    for item in candidates:
        step = item["step"]
        hits = service.query(
            query=str(step.get("suggested_direction") or step.get("title") or ""),
            object_type="pipeline_run",
            n_results=int(cfg.get("prior_run_search_results") or 5),
        )
        if any(_is_prior_hit(hit, state, max_distance) for hit in hits):
            dropped.append(_drop(step, "similar_to_prior_run"))
        else:
            keep.append(item)
    return keep, dropped


def _rank_with_llm(candidates: list[dict], state: ResearchState, profile: dict, max_selected: int) -> list[dict]:
    if not candidates:
        return []
    by_id = {item["candidate_id"]: item for item in candidates}
    payload = [
        {
            "candidate_id": item["candidate_id"],
            **{k: v for k, v in item["step"].items() if k in {"title", "rationale", "suggested_direction", "priority"}},
        }
        for item in candidates
    ]
    resp = get_llm("rank_next_steps", profile).invoke([
        SystemMessage(content=get_prompt(profile, "rank_next_steps")),
        HumanMessage(content=(
            f"Research direction: {state.get('research_direction', '')}\n\n"
            f"Evaluation summary:\n{json.dumps(state.get('evaluation_summary') or {}, indent=2, default=str)}\n\n"
            f"Candidates:\n{json.dumps(payload, indent=2, default=str)}"
        )),
    ])
    selected: list[dict] = []
    seen: set[str] = set()
    for item in extract_json_array(resp.content):
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id in by_id and candidate_id not in seen:
            selected.append(by_id[candidate_id])
            seen.add(candidate_id)
        if len(selected) >= max_selected:
            break
    return selected


def _finish(state: ResearchState | dict, profile: dict, next_steps: list[dict], selection: dict) -> dict:
    delta = {"next_steps": next_steps, "next_step_selection": selection}
    if state.get("errors"):
        delta["errors"] = state["errors"]
    try:
        persist_memory_records_for_state(profile, {**state, **delta})
    except Exception as exc:
        log.warning("rank_next_steps_node | Memory persistence failed: %s", exc)
        delta["errors"] = (state.get("errors") or []) + [f"rank_next_steps: memory persistence failed: {exc}"]
    return delta


def _selection(selected: list[dict], dropped: list[dict], mode: str) -> dict:
    return {"selected": selected, "dropped": dropped, "ranking_mode": mode}


def _drop(step: dict, reason: str) -> dict:
    return {**step, "drop_reason": reason}


def _priority(step: dict) -> tuple[float, str]:
    return (_float_or_none(step.get("priority")) or 999999.0, str(step.get("title") or ""))


def _is_prior_hit(hit: dict[str, Any], state: ResearchState, max_distance: float) -> bool:
    distance = _float_or_none(hit.get("distance"))
    if distance is None or distance > max_distance:
        return False
    family_id = str(state.get("root_run_family_id") or "")
    hit_family = str(((hit.get("record") or {}).get("metadata") or {}).get("root_run_family_id") or "")
    return not family_id or hit_family != family_id


def _norm(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
