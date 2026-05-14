from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from core.handoffs import (
    load_run_record,
    proposal_template_from_run,
    resolve_next_step_seed,
    resolve_proposal_seed,
    resolve_run_handoff_seed,
)
from core.memory import MemoryService


def resolve_brainstorm_seed(
    profile: dict[str, Any],
    *,
    source_experiment_record_id: str = "",
    handoff_record_id: str = "",
    proposal_seed_record_id: str = "",
    next_step_record_id: str = "",
) -> dict[str, Any]:
    selectors = [
        bool(handoff_record_id),
        bool(proposal_seed_record_id),
        bool(next_step_record_id),
    ]
    if sum(1 for item in selectors if item) > 1:
        raise ValueError("handoff_record_id, proposal_seed_record_id, and next_step_record_id are mutually exclusive")

    seed: dict[str, Any]
    if next_step_record_id:
        seed = _resolve_next_step_brainstorm_seed(profile, next_step_record_id=next_step_record_id)
    elif proposal_seed_record_id:
        seed = resolve_proposal_seed(
            profile,
            source_experiment_record_id=source_experiment_record_id,
            proposal_seed_record_id=proposal_seed_record_id,
        )
    elif handoff_record_id:
        seed = resolve_run_handoff_seed(
            profile,
            source_experiment_record_id=source_experiment_record_id,
            handoff_record_id=handoff_record_id,
        )
    elif source_experiment_record_id:
        seed = _resolve_experiment_brainstorm_seed(profile, source_experiment_record_id=source_experiment_record_id)
    else:
        return {}

    source_record_id = str(seed.get("source_experiment_record_id") or source_experiment_record_id or "").strip()
    if source_record_id:
        seed = _enrich_seed_with_run_context(profile, seed, source_experiment_record_id=source_record_id)
    return seed


def apply_brainstorm_seed(state: dict[str, Any], seed: dict[str, Any]) -> dict[str, Any]:
    if not seed:
        return state

    goal = str(seed.get("research_direction") or state.get("current_goal") or "").strip()
    if goal:
        state["current_goal"] = goal
        state["plan_draft"]["research_direction"] = goal

    for key in (
        "source_experiment_record_id",
        "source_next_step_record_id",
        "source_next_step_title",
        "source_proposal_seed_record_id",
        "source_proposal_seed_title",
        "proposal_seed_planning_notes",
        "root_run_family_id",
        "root_research_direction",
        "campaign_id",
        "campaign_title",
        "campaign_variant_id",
        "campaign_variant_title",
        "campaign_variant_index",
        "campaign_size",
    ):
        value = seed.get(key)
        if value not in (None, "", 0):
            state[key] = value

    if seed.get("proposals"):
        state["plan_draft"]["proposals"] = [dict(item) for item in seed.get("proposals") or [] if isinstance(item, dict)]
    if seed.get("refined_ideas"):
        state["plan_draft"]["refined_ideas"] = [dict(item) for item in seed.get("refined_ideas") or [] if isinstance(item, dict)]
    if seed.get("implementation_plans"):
        state["plan_draft"]["implementation_plans"] = [
            dict(item) for item in seed.get("implementation_plans") or [] if isinstance(item, dict)
        ]

    notes = list(state.get("user_intent_notes") or [])
    for text in list(seed.get("seed_notes") or []):
        note = str(text or "").strip()
        if note and note not in notes:
            notes.append(note)
    planning_notes = str(seed.get("proposal_seed_planning_notes") or "").strip()
    if planning_notes and planning_notes not in notes:
        notes.append(planning_notes)
    if notes:
        state["user_intent_notes"] = notes

    artifacts = list(state.get("consensus", {}).get("evidence") or [])
    seed_artifacts: list[dict[str, Any]] = []
    for item in list(seed.get("context_artifacts") or []):
        if isinstance(item, dict):
            artifact = dict(item)
            artifacts.append(artifact)
            seed_artifacts.append(artifact)
    if artifacts:
        state["consensus"]["evidence"] = artifacts[:8]
    if seed_artifacts:
        state["seed_context_artifacts"] = seed_artifacts[:8]

    active_options = list(state.get("consensus", {}).get("active_options") or [])
    for proposal in list(state.get("plan_draft", {}).get("proposals") or [])[:3]:
        name = str(proposal.get("name") or proposal.get("title") or "").strip()
        rationale = str(proposal.get("rationale") or proposal.get("description") or "").strip()
        if not name and not rationale:
            continue
        active_options.append({
            "name": name or "Imported proposal",
            "reason": rationale,
        })
    if active_options:
        state["consensus"]["active_options"] = active_options[:5]

    turn_content = str(seed.get("import_turn_content") or "").strip()
    if turn_content:
        state["turn_log"] = list(state.get("turn_log") or []) + [{
            "turn_id": f"seed:{state.get('session_id') or 'session'}",
            "round_index": 0,
            "role_name": "seed",
            "role_type": "seed",
            "message_type": "seed_import",
            "content": turn_content,
            "structured_points": [],
            "citations": [],
            "created_at": _now_iso(),
        }]

    return state


def _resolve_experiment_brainstorm_seed(profile: dict[str, Any], *, source_experiment_record_id: str) -> dict[str, Any]:
    run_record = load_run_record(profile, source_experiment_record_id)
    if not run_record:
        raise ValueError(f"Run record not found: {source_experiment_record_id}")
    metadata = dict(run_record.get("metadata") or {})
    research_direction = str(metadata.get("research_direction") or run_record.get("title") or "").strip()
    proposal = proposal_template_from_run(profile, run_record)
    seed: dict[str, Any] = {
        "research_direction": research_direction,
        "source_experiment_record_id": str(run_record.get("record_id") or source_experiment_record_id),
        "root_run_family_id": str(metadata.get("root_run_family_id") or ""),
        "root_research_direction": str(metadata.get("root_research_direction") or research_direction),
    }
    if proposal:
        seed["proposals"] = [proposal]
    return seed


def _resolve_next_step_brainstorm_seed(profile: dict[str, Any], *, next_step_record_id: str) -> dict[str, Any]:
    seed = resolve_next_step_seed(profile, next_step_record_id=next_step_record_id)
    record = MemoryService.for_profile(profile).document_store.get(next_step_record_id)
    content = dict(record.get("content") or {}) if isinstance(record, dict) else {}
    notes: list[str] = []
    rationale = str(content.get("rationale") or "").strip()
    if rationale:
        notes.append(f"Imported next step rationale: {rationale}")
    priority = str(content.get("priority") or "").strip()
    if priority:
        notes.append(f"Imported next step priority: {priority}")
    if notes:
        seed["seed_notes"] = notes
    if rationale:
        seed["context_artifacts"] = [{
            "title": str(record.get("title") or next_step_record_id),
            "summary": rationale,
            "source_type": "next_step",
            "metadata": {"record_id": next_step_record_id},
        }]
    return seed


def _enrich_seed_with_run_context(
    profile: dict[str, Any],
    seed: dict[str, Any],
    *,
    source_experiment_record_id: str,
) -> dict[str, Any]:
    run_record = load_run_record(profile, source_experiment_record_id)
    if not run_record:
        return seed

    metadata = dict(run_record.get("metadata") or {})
    proposal = proposal_template_from_run(profile, run_record)
    if proposal and not seed.get("proposals"):
        seed["proposals"] = [proposal]

    for key in (
        "campaign_id",
        "campaign_title",
        "campaign_variant_id",
        "campaign_variant_title",
        "campaign_variant_index",
        "campaign_size",
    ):
        if seed.get(key) in (None, "", 0):
            value = metadata.get(key)
            if value not in (None, "", 0):
                seed[key] = value

    artifacts = list(seed.get("context_artifacts") or [])
    run_summary = str(run_record.get("summary") or "").strip()
    assessment = str(metadata.get("assessment") or "").strip()
    parts = [f"Direction: {metadata.get('research_direction') or seed.get('research_direction') or ''}".strip()]
    if assessment:
        parts.append(f"Assessment: {assessment}")
    if run_summary:
        parts.append(run_summary[:240])
    artifacts.append({
        "title": str(run_record.get("title") or source_experiment_record_id),
        "summary": "\n".join(part for part in parts if part).strip(),
        "source_type": "experiment_result",
        "metadata": {"record_id": source_experiment_record_id},
    })
    if proposal:
        artifacts.append({
            "title": f"Imported proposal: {proposal.get('name') or run_record.get('title') or 'proposal'}",
            "summary": _compact_proposal_summary(proposal),
            "source_type": "proposal",
            "metadata": {"record_id": source_experiment_record_id},
        })
    seed["context_artifacts"] = artifacts[:6]

    notes = list(seed.get("seed_notes") or [])
    if assessment:
        notes.append(f"Imported source run assessment: {assessment}")
    seed["seed_notes"] = notes

    import_bits: list[str] = []
    if source_experiment_record_id:
        import_bits.append(f"run={source_experiment_record_id}")
    if seed.get("source_next_step_record_id"):
        import_bits.append(f"next_step={seed['source_next_step_record_id']}")
    if seed.get("source_proposal_seed_record_id"):
        import_bits.append(f"proposal_seed={seed['source_proposal_seed_record_id']}")
    seed["import_turn_content"] = "Imported context from " + ", ".join(import_bits) if import_bits else ""
    return seed


def _compact_proposal_summary(proposal: dict[str, Any]) -> str:
    preferred = {
        key: value
        for key, value in (
            ("name", proposal.get("name")),
            ("description", proposal.get("description")),
            ("hypothesis", proposal.get("hypothesis")),
            ("rationale", proposal.get("rationale")),
            ("dataset", proposal.get("dataset")),
            ("detector", proposal.get("detector")),
        )
        if value not in (None, "", [])
    }
    if not preferred:
        return json.dumps(proposal, default=str)[:240]
    return json.dumps(preferred, default=str)[:240]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
