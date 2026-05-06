"""Rank and prune proposed next steps before loop execution."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.memory import MemoryService
from core.graph.nodes.memory import persist_memory_records_for_state
from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils import extract_json_array
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)

_DEFAULT_SELECTION_LIMIT = 3
_RAW_RESPONSE_LOG_LIMIT = 12000


def rank_next_steps_node(state: ResearchState, profile: dict) -> dict:
    raw_steps = state.get("next_steps") or []
    direction = str(state.get("research_direction") or "")
    selection_cfg = _selection_cfg(profile)
    if not raw_steps:
        return {
            "next_steps": [],
            "next_step_selection": {"selected": [], "dropped": [], "ranking_mode": "empty"},
        }

    performance_gate = _performance_gate(profile, state, selection_cfg)
    if performance_gate is not None:
        return performance_gate

    candidates, dropped = _prepare_candidates(raw_steps, current_direction=direction)
    if not candidates:
        return {
            "next_steps": [],
            "next_step_selection": {"selected": [], "dropped": dropped, "ranking_mode": "filtered_empty"},
        }

    candidates, novelty_dropped, novelty_errors = _apply_prior_run_novelty_filter(
        candidates,
        profile=profile,
        state=state,
        selection_cfg=selection_cfg,
    )
    dropped.extend(novelty_dropped)
    if not candidates:
        delta: dict[str, Any] = {
            "next_steps": [],
            "next_step_selection": {"selected": [], "dropped": dropped, "ranking_mode": "novelty_filtered_empty"},
        }
        if novelty_errors:
            delta["errors"] = (state.get("errors") or []) + novelty_errors
        _persist_ranked_next_steps(profile, state, delta)
        return delta

    selection_limit = min(_selection_limit(profile), len(candidates))
    if len(candidates) <= 1:
        selected = [candidate["step"] for candidate in candidates[:selection_limit]]
        delta = {
            "next_steps": selected,
            "next_step_selection": _selection_summary(
                selected_candidates=candidates[:selection_limit],
                dropped=dropped,
                extra_dropped=[],
                ranking_mode="single_candidate",
            ),
        }
        _persist_ranked_next_steps(profile, state, delta)
        return delta

    evaluation_summary = state.get("evaluation_summary") or {}
    experiment_results = state.get("experiment_results") or []
    results_summary = [
        {
            "proposal_name": result.get("proposal_name"),
            "metrics": result.get("metrics", {}),
        }
        for result in experiment_results
    ]

    response = None
    try:
        system_prompt = get_prompt(profile, "rank_next_steps")
        llm = get_llm("rank_next_steps", profile)
        user_content = (
            f"Research direction: {direction}\n\n"
            f"Selection limit: {selection_limit}\n\n"
            f"Evaluation summary:\n{json.dumps(evaluation_summary, indent=2, default=str)}\n\n"
            f"Experiment results overview:\n{json.dumps(results_summary, indent=2, default=str)}\n\n"
            f"Candidate next steps:\n{json.dumps([candidate['prompt_view'] for candidate in candidates], indent=2)}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
        )
        ranked = extract_json_array(response.content)
        selected_candidates, extra_dropped = _select_ranked_candidates(ranked, candidates, selection_limit)
        log.info(
            "rank_next_steps_node | Ranked %d candidate(s) down to %d selected step(s)",
            len(candidates),
            len(selected_candidates),
        )
        delta = {
            "next_steps": [candidate["step"] for candidate in selected_candidates],
            "next_step_selection": _selection_summary(
                selected_candidates=selected_candidates,
                dropped=dropped,
                extra_dropped=extra_dropped,
                ranking_mode="llm",
            ),
        }
        if novelty_errors:
            delta["errors"] = (state.get("errors") or []) + novelty_errors
    except Exception as exc:
        log.warning("rank_next_steps_node | Ranking failed, using fallback ordering: %s", exc)
        if response is not None:
            log.warning(
                "rank_next_steps_node | Raw LLM response follows:\n%s",
                _truncate_text(str(getattr(response, "content", "") or ""), _RAW_RESPONSE_LOG_LIMIT),
            )
        selected_candidates, extra_dropped = _fallback_ranked_candidates(candidates, selection_limit)
        delta = {
            "next_steps": [candidate["step"] for candidate in selected_candidates],
            "next_step_selection": _selection_summary(
                selected_candidates=selected_candidates,
                dropped=dropped,
                extra_dropped=extra_dropped,
                ranking_mode="fallback",
            ),
            "errors": (state.get("errors") or []) + novelty_errors + [f"rank_next_steps fallback: {exc}"],
        }

    _persist_ranked_next_steps(profile, state, delta)
    return delta


def _prepare_candidates(
    raw_steps: list[dict[str, Any]],
    *,
    current_direction: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_keys: set[str] = set()
    candidates: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    current_key = _normalize_direction_key(current_direction)
    for idx, step in enumerate(raw_steps, start=1):
        candidate_id = f"step_{idx}"
        if not isinstance(step, dict):
            continue
        direction = str(step.get("suggested_direction") or step.get("title") or "").strip()
        title = str(step.get("title") or step.get("suggested_direction") or "").strip()
        if not direction or not title:
            dropped.append(_dropped_entry(candidate_id, step, "empty_direction_or_title"))
            continue
        direction_key = _normalize_direction_key(direction)
        if not direction_key:
            dropped.append(_dropped_entry(candidate_id, step, "empty_normalized_direction"))
            continue
        if direction_key == current_key:
            dropped.append(_dropped_entry(candidate_id, step, "same_as_current_direction"))
            continue
        if direction_key in seen_keys:
            dropped.append(_dropped_entry(candidate_id, step, "duplicate_direction"))
            continue
        seen_keys.add(direction_key)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "direction_key": direction_key,
                "step": step,
                "prompt_view": {
                    "candidate_id": candidate_id,
                    "title": title,
                    "rationale": step.get("rationale", ""),
                    "suggested_direction": direction,
                    "priority": step.get("priority"),
                },
            }
        )
    return candidates, dropped


def _select_ranked_candidates(
    ranked: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selection_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in ranked:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen_ids:
            continue
        candidate = candidate_by_id.get(candidate_id)
        if not candidate:
            continue
        selected.append(candidate["step"])
        seen_ids.add(candidate_id)
        if len(selected) >= selection_limit:
            break
    if selected:
        selected_candidates = [candidate_by_id[_candidate_id_for_step(step, candidates)] for step in selected]
        extra_dropped = [
            _dropped_entry(candidate["candidate_id"], candidate["step"], "not_selected_by_ranker")
            for candidate in candidates
            if candidate["candidate_id"] not in seen_ids
        ]
        return selected_candidates, extra_dropped
    return _fallback_ranked_candidates(candidates, selection_limit)


def _fallback_ranked_candidates(
    candidates: list[dict[str, Any]],
    selection_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            _priority_value(candidate["step"].get("priority")),
            candidate["prompt_view"]["title"].lower(),
        ),
    )
    selected = ranked[:selection_limit]
    selected_ids = {candidate["candidate_id"] for candidate in selected}
    dropped = [
        _dropped_entry(candidate["candidate_id"], candidate["step"], "not_selected_by_fallback")
        for candidate in ranked
        if candidate["candidate_id"] not in selected_ids
    ]
    return selected, dropped


def _selection_limit(profile: dict[str, Any]) -> int:
    cfg = _selection_cfg(profile)
    try:
        limit = int(cfg.get("max_selected", _DEFAULT_SELECTION_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_SELECTION_LIMIT
    return max(1, limit)


def _selection_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("next_steps") or {}
    return cfg if isinstance(cfg, dict) else {}


def _performance_gate(
    profile: dict[str, Any],
    state: ResearchState,
    selection_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    threshold = _maybe_float(selection_cfg.get("min_parent_metric"))
    if threshold is None:
        return None
    evaluation_cfg = profile.get("evaluation") or {}
    metric_name = str(evaluation_cfg.get("primary_metric") or "test_auc")
    best_value = _maybe_float((state.get("evaluation_summary") or {}).get("best_metric_value"))
    if best_value is None:
        return None
    if best_value >= threshold:
        return None
    return {
        "next_steps": [],
        "next_step_selection": {
            "selected": [],
            "dropped": [
                {
                    "candidate_id": "*",
                    "title": "",
                    "suggested_direction": "",
                    "drop_reason": f"parent_metric_below_threshold:{metric_name}<{threshold:g}",
                }
            ],
            "ranking_mode": "parent_metric_gate",
        },
    }


def _apply_prior_run_novelty_filter(
    candidates: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    state: ResearchState,
    selection_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    threshold = _maybe_float(selection_cfg.get("prior_run_similarity_threshold"))
    if threshold is None:
        return candidates, [], []
    n_results = int(selection_cfg.get("prior_run_search_results", 5) or 5)
    root_run_family_id = str(state.get("root_run_family_id") or "")
    primary_metric = str((profile.get("evaluation") or {}).get("primary_metric") or "test_auc")
    try:
        service = MemoryService.for_profile(profile)
    except Exception as exc:
        return candidates, [], [f"rank_next_steps novelty lookup unavailable: {exc}"]

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in candidates:
        direction = candidate["prompt_view"]["suggested_direction"]
        try:
            hits = service.query(
                query=direction,
                domain=str(profile.get("name") or ""),
                object_type="experiment_result",
                n_results=n_results,
            )
        except Exception as exc:
            errors.append(f"rank_next_steps novelty query failed for {candidate['candidate_id']}: {exc}")
            kept.append(candidate)
            continue
        match = _find_similar_prior_run(
            hits,
            similarity_threshold=threshold,
            root_run_family_id=root_run_family_id,
        )
        if not match:
            kept.append(candidate)
            continue
        similarity = float(match["similarity"])
        prior_metric = _extract_hit_metric(match["hit"], primary_metric)
        dropped.append(
            {
                "candidate_id": candidate["candidate_id"],
                "title": candidate["prompt_view"]["title"],
                "suggested_direction": direction,
                "drop_reason": "similar_to_prior_run",
                "matched_record_id": str((match["hit"].get("record") or {}).get("record_id") or ""),
                "matched_title": str((match["hit"].get("record") or {}).get("title") or ""),
                "similarity": similarity,
                "matched_metric_name": primary_metric,
                "matched_metric_value": prior_metric,
            }
        )
    return kept, dropped, errors


def _find_similar_prior_run(
    hits: list[dict[str, Any]],
    *,
    similarity_threshold: float,
    root_run_family_id: str,
) -> dict[str, Any] | None:
    for hit in hits:
        record = hit.get("record") or {}
        metadata = record.get("metadata") or {}
        if root_run_family_id and str(metadata.get("root_run_family_id") or "") == root_run_family_id:
            continue
        similarity = _distance_to_similarity(hit.get("distance"))
        if similarity >= similarity_threshold:
            return {"hit": hit, "similarity": similarity}
    return None


def _distance_to_similarity(distance: Any) -> float:
    value = _maybe_float(distance)
    if value is None:
        return 0.0
    return 1.0 / (1.0 + max(0.0, value))


def _extract_hit_metric(hit: dict[str, Any], metric_name: str) -> float | None:
    record = hit.get("record") or {}
    metadata = record.get("metadata") or {}
    content = record.get("content") or {}
    metrics = content.get("metrics") or {}
    return _maybe_float(metrics.get(metric_name) if isinstance(metrics, dict) else metadata.get(metric_name))


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _priority_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    return 999


def _normalize_direction_key(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return ""
    return re.sub(r"\s+", " ", lowered)


def _selection_summary(
    *,
    selected_candidates: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    extra_dropped: list[dict[str, Any]],
    ranking_mode: str,
) -> dict[str, Any]:
    return {
        "selected": [
            {
                "candidate_id": candidate["candidate_id"],
                "title": candidate["prompt_view"]["title"],
                "suggested_direction": candidate["prompt_view"]["suggested_direction"],
                "selection_reason": "selected_for_execution",
            }
            for candidate in selected_candidates
        ],
        "dropped": [*dropped, *extra_dropped],
        "ranking_mode": ranking_mode,
    }


def _dropped_entry(candidate_id: str, step: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "title": str(step.get("title") or step.get("suggested_direction") or "").strip(),
        "suggested_direction": str(step.get("suggested_direction") or step.get("title") or "").strip(),
        "drop_reason": reason,
    }


def _candidate_id_for_step(step: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    for candidate in candidates:
        if candidate["step"] is step:
            return str(candidate["candidate_id"])
    raise KeyError("candidate id not found for selected step")


def _persist_ranked_next_steps(profile: dict[str, Any], state: ResearchState, delta: dict[str, Any]) -> None:
    try:
        persist_memory_records_for_state(profile, {**state, **delta})
    except Exception as exc:
        log.warning("rank_next_steps_node | Memory persistence failed: %s", exc)
        delta["errors"] = (delta.get("errors") or state.get("errors") or []) + [
            f"rank_next_steps: memory persistence failed: {exc}"
        ]


def _truncate_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
