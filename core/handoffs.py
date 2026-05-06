from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from uuid import uuid4

import pymongo

from configs.config import get_config
from core.memory import MemoryService, build_memory_record


def build_handoff_draft(profile_name: str, run_record: dict[str, Any], text_panels: list[dict[str, str]]) -> dict[str, Any]:
    metadata = dict(run_record.get("metadata") or {})
    summary = str(metadata.get("assessment") or "").strip()
    proposal_name = str(metadata.get("proposal_name") or run_record.get("title") or run_record.get("record_id") or "").strip()
    direction = str(metadata.get("research_direction") or "").strip()

    snippets: list[dict[str, Any]] = []
    preferred_titles = {
        "Direction",
        "Research Summary",
        "Refined Idea",
        "Proposal",
        "Run Goal",
        "Evaluation Summary",
    }
    for panel in text_panels:
        title = str(panel.get("title") or "").strip()
        body = str(panel.get("body") or "").strip()
        if not title or not body:
            continue
        snippets.append({
            "title": title,
            "body": body,
            "selected": title in preferred_titles,
        })

    title = f"Follow-up: {proposal_name}" if proposal_name else f"Follow-up: {run_record.get('record_id', 'run')}"
    rationale = summary or "Use the selected run context and notes below to define the next research direction."
    launch_direction = direction
    return {
        "record_id": "",
        "profile_name": profile_name,
        "title": title,
        "suggested_direction": launch_direction,
        "launch_direction": launch_direction,
        "rationale": rationale,
        "user_notes": "",
        "snippets": snippets,
        "source_experiment_record_id": str(run_record.get("record_id") or ""),
        "source_experiment_id": str(metadata.get("experiment_id") or ""),
        "root_run_family_id": str(metadata.get("root_run_family_id") or ""),
        "root_research_direction": str(metadata.get("root_research_direction") or metadata.get("research_direction") or ""),
    }


def save_run_handoff(
    profile: dict[str, Any],
    source_run_record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(source_run_record.get("metadata") or {})
    record_id = str(payload.get("record_id") or "").strip() or f"run_handoff:{uuid4()}"
    title = str(payload.get("title") or payload.get("launch_direction") or payload.get("suggested_direction") or "run handoff").strip()
    launch_direction = str(payload.get("launch_direction") or payload.get("suggested_direction") or "").strip()
    if not launch_direction:
        raise ValueError("launch_direction is required")

    snippets = _normalize_snippets(payload.get("snippets") or [])
    suggested_direction = str(payload.get("suggested_direction") or launch_direction).strip()
    rationale = str(payload.get("rationale") or "").strip()
    user_notes = str(payload.get("user_notes") or "").strip()
    updated_at = _now_iso()
    source_record_id = str(source_run_record.get("record_id") or "")
    source_experiment_id = str(metadata.get("experiment_id") or "")
    root_run_family_id = str(metadata.get("root_run_family_id") or "")
    root_research_direction = str(metadata.get("root_research_direction") or metadata.get("research_direction") or "")

    prompt_preview = build_handoff_prompt_preview({
        "title": title,
        "launch_direction": launch_direction,
        "suggested_direction": suggested_direction,
        "rationale": rationale,
        "user_notes": user_notes,
        "snippets": snippets,
    })
    record = build_memory_record(
        profile=profile,
        object_type="run_handoff",
        payload={
            "record_id": record_id,
            "title": title,
            "suggested_direction": suggested_direction,
            "launch_direction": launch_direction,
            "rationale": rationale,
            "user_notes": user_notes,
            "snippets": snippets,
            "prompt_preview": prompt_preview,
            "source_experiment_record_id": source_record_id,
            "source_experiment_id": source_experiment_id,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
        },
        node="web_ui",
        kind="run_handoff",
        object_key=record_id,
        object_role="recommendation",
        title=title,
        summary=prompt_preview[:1200],
        metadata={
            "profile": str(profile.get("name") or ""),
            "source_experiment_record_id": source_record_id,
            "source_experiment_id": source_experiment_id,
            "source_proposal_name": str(metadata.get("proposal_name") or ""),
            "source_assessment": str(metadata.get("assessment") or ""),
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "updated_at": updated_at,
            "launch_direction": launch_direction,
            "status": "active",
        },
        tags=[str(profile.get("name") or ""), "run_handoff"],
        source_record_ids=[source_record_id] if source_record_id else [],
    )
    MemoryService.for_profile(profile).persist_record(record)
    return record


def build_proposal_seed_draft(
    profile_name: str,
    run_record: dict[str, Any],
    related_records: list[dict[str, Any]],
    text_panels: list[dict[str, str]],
) -> dict[str, Any]:
    metadata = dict(run_record.get("metadata") or {})
    proposal_template = _proposal_template_from_run(run_record, related_records)
    proposal_name = str(proposal_template.get("name") or metadata.get("proposal_name") or run_record.get("title") or run_record.get("record_id") or "").strip()
    return {
        "record_id": "",
        "profile_name": profile_name,
        "title": f"Proposal Seed: {proposal_name}" if proposal_name else f"Proposal Seed: {run_record.get('record_id', 'run')}",
        "research_direction": str(metadata.get("research_direction") or "").strip(),
        "proposal_template": proposal_template,
        "planning_notes": "",
        "snippets": _proposal_seed_snippets(text_panels),
        "source_experiment_record_id": str(run_record.get("record_id") or ""),
        "source_experiment_id": str(metadata.get("experiment_id") or ""),
        "root_run_family_id": str(metadata.get("root_run_family_id") or ""),
        "root_research_direction": str(metadata.get("root_research_direction") or metadata.get("research_direction") or ""),
    }


def save_proposal_seed(
    profile: dict[str, Any],
    source_run_record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(source_run_record.get("metadata") or {})
    record_id = str(payload.get("record_id") or "").strip() or f"proposal_seed:{uuid4()}"
    proposal_template = payload.get("proposal_template")
    if not isinstance(proposal_template, dict) or not proposal_template:
        raise ValueError("proposal_template must be a non-empty JSON object")
    title = str(
        payload.get("title")
        or proposal_template.get("name")
        or metadata.get("proposal_name")
        or "proposal seed"
    ).strip()
    research_direction = str(payload.get("research_direction") or metadata.get("research_direction") or "").strip()
    planning_notes = str(payload.get("planning_notes") or "").strip()
    snippets = _normalize_snippets(payload.get("snippets") or [])
    updated_at = _now_iso()
    source_record_id = str(source_run_record.get("record_id") or "")
    source_experiment_id = str(metadata.get("experiment_id") or "")
    root_run_family_id = str(metadata.get("root_run_family_id") or "")
    root_research_direction = str(metadata.get("root_research_direction") or metadata.get("research_direction") or research_direction or "")
    campaign_id = str(payload.get("campaign_id") or metadata.get("campaign_id") or "").strip()
    campaign_title = str(payload.get("campaign_title") or metadata.get("campaign_title") or "").strip()
    campaign_variant_id = str(payload.get("campaign_variant_id") or "").strip()
    campaign_variant_title = str(payload.get("campaign_variant_title") or "").strip()
    campaign_variant_index = int(payload.get("campaign_variant_index") or 0)
    campaign_size = int(payload.get("campaign_size") or 0)
    prompt_preview = build_proposal_seed_preview({
        "title": title,
        "research_direction": research_direction,
        "proposal_template": proposal_template,
        "planning_notes": planning_notes,
        "snippets": snippets,
    })
    record = build_memory_record(
        profile=profile,
        object_type="proposal_seed",
        payload={
            "record_id": record_id,
            "title": title,
            "research_direction": research_direction,
            "proposal_template": proposal_template,
            "planning_notes": planning_notes,
            "snippets": snippets,
            "prompt_preview": prompt_preview,
            "source_experiment_record_id": source_record_id,
            "source_experiment_id": source_experiment_id,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "campaign_id": campaign_id,
            "campaign_title": campaign_title,
            "campaign_variant_id": campaign_variant_id,
            "campaign_variant_title": campaign_variant_title,
            "campaign_variant_index": campaign_variant_index,
            "campaign_size": campaign_size,
        },
        node="web_ui",
        kind="proposal_seed",
        object_key=record_id,
        object_role="recommendation",
        title=title,
        summary=prompt_preview[:1200],
        metadata={
            "profile": str(profile.get("name") or ""),
            "source_experiment_record_id": source_record_id,
            "source_experiment_id": source_experiment_id,
            "source_proposal_name": str(metadata.get("proposal_name") or ""),
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "updated_at": updated_at,
            "research_direction": research_direction,
            "status": "active",
            "campaign_id": campaign_id,
            "campaign_title": campaign_title,
            "campaign_variant_id": campaign_variant_id,
            "campaign_variant_title": campaign_variant_title,
            "campaign_variant_index": campaign_variant_index,
            "campaign_size": campaign_size,
        },
        tags=[str(profile.get("name") or ""), "proposal_seed"],
        source_record_ids=[source_record_id] if source_record_id else [],
    )
    MemoryService.for_profile(profile).persist_record(record)
    return record


def list_run_handoffs(profile: dict[str, Any], *, source_experiment_record_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not source_experiment_record_id:
        return []
    service = MemoryService.for_profile(profile)
    records = service.find_records(
        {
            "object_type": "run_handoff",
            "metadata.source_experiment_record_id": source_experiment_record_id,
        },
        limit=limit,
    )
    records.sort(
        key=lambda item: str((item.get("metadata") or {}).get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    )
    return records


def list_proposal_seeds(profile: dict[str, Any], *, source_experiment_record_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not source_experiment_record_id:
        return []
    service = MemoryService.for_profile(profile)
    records = service.find_records(
        {
            "object_type": "proposal_seed",
            "metadata.source_experiment_record_id": source_experiment_record_id,
        },
        limit=limit,
    )
    records.sort(
        key=lambda item: str((item.get("metadata") or {}).get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    )
    return records


def resolve_run_handoff_seed(
    profile: dict[str, Any],
    *,
    source_experiment_record_id: str = "",
    handoff_record_id: str = "",
) -> dict[str, str]:
    service = MemoryService.for_profile(profile)
    handoff_record = None
    if handoff_record_id:
        handoff_record = service.document_store.get(handoff_record_id)
        if not handoff_record or str(handoff_record.get("object_type") or "") != "run_handoff":
            raise ValueError(f"Run handoff not found: {handoff_record_id}")

    if handoff_record is None:
        if not source_experiment_record_id:
            raise ValueError("source_experiment_record_id is required when handoff_record_id is not provided")
        handoffs = list_run_handoffs(profile, source_experiment_record_id=source_experiment_record_id, limit=20)
        if not handoffs:
            raise ValueError(f"No saved handoff found for source experiment: {source_experiment_record_id}")
        handoff_record = handoffs[0]

    handoff_content = dict(handoff_record.get("content") or {})
    metadata = dict(handoff_record.get("metadata") or {})
    source_record_id = str(metadata.get("source_experiment_record_id") or handoff_content.get("source_experiment_record_id") or source_experiment_record_id or "")
    source_run_record = _load_source_experiment_record(profile, source_record_id) if source_record_id else {}
    source_metadata = dict(source_run_record.get("metadata") or {})
    research_direction = str(handoff_content.get("launch_direction") or handoff_content.get("suggested_direction") or "").strip()
    if not research_direction:
        raise ValueError(f"Run handoff is missing launch_direction: {handoff_record.get('record_id')}")

    root_run_family_id = str(
        metadata.get("root_run_family_id")
        or handoff_content.get("root_run_family_id")
        or source_metadata.get("root_run_family_id")
        or ""
    )
    root_research_direction = str(
        metadata.get("root_research_direction")
        or handoff_content.get("root_research_direction")
        or source_metadata.get("root_research_direction")
        or source_metadata.get("research_direction")
        or research_direction
    )
    title = str(handoff_record.get("title") or handoff_content.get("title") or research_direction)
    return {
        "research_direction": research_direction,
        "source_next_step_record_id": str(handoff_record.get("record_id") or ""),
        "source_next_step_title": title,
        "root_run_family_id": root_run_family_id,
        "root_research_direction": root_research_direction,
        "source_experiment_record_id": source_record_id,
    }


def resolve_next_step_seed(
    profile: dict[str, Any],
    *,
    next_step_record_id: str,
) -> dict[str, str]:
    service = MemoryService.for_profile(profile)
    next_step_record = service.document_store.get(next_step_record_id)
    if not next_step_record or str(next_step_record.get("object_type") or "") != "next_step":
        raise ValueError(f"Next step not found: {next_step_record_id}")

    content = next_step_record.get("content") if isinstance(next_step_record.get("content"), dict) else {}
    metadata = dict(next_step_record.get("metadata") or {})
    research_direction = str(content.get("suggested_direction") or next_step_record.get("title") or "").strip()
    if not research_direction:
        raise ValueError(f"Next step is missing suggested_direction: {next_step_record_id}")

    title = str(next_step_record.get("title") or content.get("title") or research_direction).strip()
    root_run_family_id = str(metadata.get("root_run_family_id") or "")
    root_research_direction = str(
        metadata.get("root_research_direction")
        or metadata.get("research_direction")
        or research_direction
    )
    return {
        "research_direction": research_direction,
        "source_next_step_record_id": str(next_step_record.get("record_id") or ""),
        "source_next_step_title": title,
        "root_run_family_id": root_run_family_id,
        "root_research_direction": root_research_direction,
    }


def resolve_proposal_seed(
    profile: dict[str, Any],
    *,
    source_experiment_record_id: str = "",
    proposal_seed_record_id: str = "",
) -> dict[str, Any]:
    service = MemoryService.for_profile(profile)
    seed_record = None
    if proposal_seed_record_id:
        seed_record = service.document_store.get(proposal_seed_record_id)
        if not seed_record or str(seed_record.get("object_type") or "") != "proposal_seed":
            raise ValueError(f"Proposal seed not found: {proposal_seed_record_id}")
    if seed_record is None:
        if not source_experiment_record_id:
            raise ValueError("source_experiment_record_id is required when proposal_seed_record_id is not provided")
        seeds = list_proposal_seeds(profile, source_experiment_record_id=source_experiment_record_id, limit=20)
        if not seeds:
            raise ValueError(f"No saved proposal seed found for source experiment: {source_experiment_record_id}")
        seed_record = seeds[0]

    content = dict(seed_record.get("content") or {})
    metadata = dict(seed_record.get("metadata") or {})
    proposal_template = content.get("proposal_template")
    if not isinstance(proposal_template, dict) or not proposal_template:
        raise ValueError(f"Proposal seed is missing proposal_template: {seed_record.get('record_id')}")
    source_record_id = str(metadata.get("source_experiment_record_id") or content.get("source_experiment_record_id") or source_experiment_record_id or "")
    source_run_record = _load_source_experiment_record(profile, source_record_id) if source_record_id else {}
    source_metadata = dict(source_run_record.get("metadata") or {})
    research_direction = str(
        content.get("research_direction")
        or metadata.get("research_direction")
        or source_metadata.get("research_direction")
        or ""
    ).strip()
    root_run_family_id = str(
        metadata.get("root_run_family_id")
        or content.get("root_run_family_id")
        or source_metadata.get("root_run_family_id")
        or ""
    )
    root_research_direction = str(
        metadata.get("root_research_direction")
        or content.get("root_research_direction")
        or source_metadata.get("root_research_direction")
        or source_metadata.get("research_direction")
        or research_direction
    )
    title = str(seed_record.get("title") or content.get("title") or proposal_template.get("name") or "proposal seed")
    return {
        "research_direction": research_direction,
        "proposals": [proposal_template],
        "proposal_seed_planning_notes": str(content.get("planning_notes") or "").strip(),
        "source_proposal_seed_record_id": str(seed_record.get("record_id") or ""),
        "source_proposal_seed_title": title,
        "root_run_family_id": root_run_family_id,
        "root_research_direction": root_research_direction,
        "source_experiment_record_id": source_record_id,
        "campaign_id": str(metadata.get("campaign_id") or content.get("campaign_id") or ""),
        "campaign_title": str(metadata.get("campaign_title") or content.get("campaign_title") or ""),
        "campaign_variant_id": str(metadata.get("campaign_variant_id") or content.get("campaign_variant_id") or ""),
        "campaign_variant_title": str(metadata.get("campaign_variant_title") or content.get("campaign_variant_title") or ""),
        "campaign_variant_index": int(metadata.get("campaign_variant_index") or content.get("campaign_variant_index") or 0),
        "campaign_size": int(metadata.get("campaign_size") or content.get("campaign_size") or 0),
    }


def load_run_record(profile: dict[str, Any], record_id: str) -> dict[str, Any] | None:
    service = MemoryService.for_profile(profile)
    record = service.document_store.get(record_id)
    if record:
        return record
    return _raw_result_record(profile, record_id)


def proposal_template_from_run(profile: dict[str, Any], run_record: dict[str, Any]) -> dict[str, Any]:
    service = MemoryService.for_profile(profile)
    metadata = dict(run_record.get("metadata") or {})
    filters: dict[str, Any] = {"object_type": "proposal"}
    proposal_name = str(metadata.get("proposal_name") or "")
    family_id = str(metadata.get("root_run_family_id") or "")
    direction = str(metadata.get("research_direction") or "")
    if proposal_name:
        filters["metadata.proposal_name"] = proposal_name
    if family_id:
        filters["metadata.root_run_family_id"] = family_id
    elif direction:
        filters["metadata.research_direction"] = direction
    related_records = service.find_records(filters, limit=20)
    return _proposal_template_from_run(run_record, related_records)


def handoff_record_to_summary(profile_name: str, record: dict[str, Any]) -> dict[str, Any]:
    content = dict(record.get("content") or {})
    metadata = dict(record.get("metadata") or {})
    return {
        "record_id": str(record.get("record_id") or ""),
        "title": str(record.get("title") or content.get("title") or ""),
        "launch_direction": str(content.get("launch_direction") or content.get("suggested_direction") or ""),
        "suggested_direction": str(content.get("suggested_direction") or content.get("launch_direction") or ""),
        "rationale": str(content.get("rationale") or ""),
        "user_notes": str(content.get("user_notes") or ""),
        "snippets": _normalize_snippets(content.get("snippets") or []),
        "updated_at": str(metadata.get("updated_at") or record.get("created_at") or ""),
        "source_experiment_record_id": str(metadata.get("source_experiment_record_id") or content.get("source_experiment_record_id") or ""),
        "copy_command": build_handoff_cli_command(
            profile_name,
            source_experiment_record_id=str(metadata.get("source_experiment_record_id") or content.get("source_experiment_record_id") or ""),
            handoff_record_id=str(record.get("record_id") or ""),
        ),
        "prompt_preview": str(content.get("prompt_preview") or build_handoff_prompt_preview(content)),
    }


def proposal_seed_record_to_summary(profile_name: str, record: dict[str, Any]) -> dict[str, Any]:
    content = dict(record.get("content") or {})
    metadata = dict(record.get("metadata") or {})
    proposal_template = content.get("proposal_template") if isinstance(content.get("proposal_template"), dict) else {}
    return {
        "record_id": str(record.get("record_id") or ""),
        "title": str(record.get("title") or content.get("title") or ""),
        "research_direction": str(content.get("research_direction") or metadata.get("research_direction") or ""),
        "proposal_template": proposal_template,
        "planning_notes": str(content.get("planning_notes") or ""),
        "snippets": _normalize_snippets(content.get("snippets") or []),
        "updated_at": str(metadata.get("updated_at") or record.get("created_at") or ""),
        "source_experiment_record_id": str(metadata.get("source_experiment_record_id") or content.get("source_experiment_record_id") or ""),
        "copy_command": build_proposal_seed_cli_command(
            profile_name,
            source_experiment_record_id=str(metadata.get("source_experiment_record_id") or content.get("source_experiment_record_id") or ""),
            proposal_seed_record_id=str(record.get("record_id") or ""),
        ),
        "prompt_preview": str(content.get("prompt_preview") or build_proposal_seed_preview(content)),
    }


def build_handoff_cli_command(profile_name: str, *, source_experiment_record_id: str, handoff_record_id: str) -> str:
    return (
        f'uv run python main.py --profile {profile_name} '
        f'--source-experiment "{source_experiment_record_id}" --handoff "{handoff_record_id}"'
    )


def build_proposal_seed_cli_command(profile_name: str, *, source_experiment_record_id: str, proposal_seed_record_id: str) -> str:
    return (
        f'uv run python main.py --profile {profile_name} '
        f'--source-experiment "{source_experiment_record_id}" --proposal-seed "{proposal_seed_record_id}"'
    )


def build_next_step_cli_command(profile_name: str, *, next_step_record_id: str) -> str:
    return f'uv run python main.py --profile {profile_name} --next-step "{next_step_record_id}"'


def build_handoff_prompt_preview(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    title = str(payload.get("title") or "").strip()
    launch_direction = str(payload.get("launch_direction") or payload.get("suggested_direction") or "").strip()
    rationale = str(payload.get("rationale") or "").strip()
    user_notes = str(payload.get("user_notes") or "").strip()
    if title:
        parts.append(f"Title: {title}")
    if launch_direction:
        parts.append(f"Next direction: {launch_direction}")
    if rationale:
        parts.append(f"Why this is worth pursuing: {rationale}")
    if user_notes:
        parts.append(f"User notes: {user_notes}")
    selected = [snippet for snippet in _normalize_snippets(payload.get("snippets") or []) if snippet.get("selected")]
    if selected:
        parts.append("Source context:")
        for snippet in selected:
            parts.append(f"[{snippet['title']}]\n{snippet['body']}")
    return "\n\n".join(part for part in parts if part).strip()


def build_proposal_seed_preview(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    title = str(payload.get("title") or "").strip()
    direction = str(payload.get("research_direction") or "").strip()
    proposal_template = payload.get("proposal_template") if isinstance(payload.get("proposal_template"), dict) else {}
    planning_notes = str(payload.get("planning_notes") or "").strip()
    if title:
        parts.append(f"Title: {title}")
    if direction:
        parts.append(f"Research direction: {direction}")
    if proposal_template:
        parts.append(f"Proposal template:\n{json.dumps(proposal_template, indent=2, sort_keys=True, default=str)}")
    if planning_notes:
        parts.append(f"Planning notes: {planning_notes}")
    selected = [snippet for snippet in _normalize_snippets(payload.get("snippets") or []) if snippet.get("selected")]
    if selected:
        parts.append("Source context:")
        for snippet in selected:
            parts.append(f"[{snippet['title']}]\n{snippet['body']}")
    return "\n\n".join(part for part in parts if part).strip()


def _load_source_experiment_record(profile: dict[str, Any], record_id: str) -> dict[str, Any]:
    if not record_id:
        return {}
    service = MemoryService.for_profile(profile)
    record = service.document_store.get(record_id)
    if record:
        return record
    raw = _raw_result_record(profile, record_id)
    return raw or {}


def _raw_result_record(profile: dict[str, Any], record_id: str) -> dict[str, Any] | None:
    cfg = get_config()
    storage_cfg = profile.get("storage") or {}
    client = pymongo.MongoClient(cfg.mongo_url)
    db_name = storage_cfg.get("mongodb_results_db", "researcher_results")
    collection_name = storage_cfg.get("mongodb_results_collection", "experiments")
    try:
        doc = client[db_name][collection_name].find_one(
            {"$or": [{"experiment_id": record_id}, {"proposal_name": record_id}]},
            {"_id": 0},
        )
    except Exception:
        return None
    finally:
        client.close()
    if not doc:
        return None
    return {
        "record_id": str(doc.get("experiment_id") or doc.get("proposal_name") or ""),
        "title": str(doc.get("proposal_name") or ""),
        "object_type": "experiment_result",
        "metadata": {
            "experiment_id": str(doc.get("experiment_id") or ""),
            "proposal_name": str(doc.get("proposal_name") or ""),
            "research_direction": str(doc.get("research_direction") or ""),
            "root_run_family_id": str(doc.get("root_run_family_id") or ""),
            "root_research_direction": str(doc.get("root_research_direction") or doc.get("research_direction") or ""),
            "campaign_id": str(doc.get("campaign_id") or ""),
            "campaign_title": str(doc.get("campaign_title") or ""),
            "campaign_variant_id": str(doc.get("campaign_variant_id") or ""),
            "campaign_variant_title": str(doc.get("campaign_variant_title") or ""),
            "campaign_variant_index": int(doc.get("campaign_variant_index") or 0),
            "campaign_size": int(doc.get("campaign_size") or 0),
            "assessment": str((((doc.get("evaluation_summary") or {}).get("per_proposal_analysis") or {}).get(str(doc.get("proposal_name") or ""), {}) or {}).get("assessment") or ""),
        },
    }


def _normalize_snippets(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        if not title or not body:
            continue
        snippets.append({
            "title": title,
            "body": body,
            "selected": bool(item.get("selected", True)),
        })
    return snippets


def _proposal_seed_snippets(text_panels: list[dict[str, str]]) -> list[dict[str, Any]]:
    preferred_titles = {
        "Direction",
        "Proposal",
        "Implementation Plan",
        "Evaluation Summary",
        "Proposed Next Steps",
    }
    snippets: list[dict[str, Any]] = []
    for panel in text_panels:
        title = str(panel.get("title") or "").strip()
        body = str(panel.get("body") or "").strip()
        if not title or not body:
            continue
        snippets.append({
            "title": title,
            "body": body,
            "selected": title in preferred_titles,
        })
    return snippets


def _proposal_template_from_run(run_record: dict[str, Any], related_records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = dict(run_record.get("metadata") or {})
    content = run_record.get("content") if isinstance(run_record.get("content"), dict) else {}
    proposal = content.get("proposal") if isinstance(content.get("proposal"), dict) else {}
    if proposal:
        return dict(proposal)
    for record in related_records:
        if str(record.get("kind") or record.get("object_type") or "") == "proposal":
            proposal_content = record.get("content")
            if isinstance(proposal_content, dict) and proposal_content:
                return dict(proposal_content)
    template: dict[str, Any] = {}
    if metadata.get("proposal_name"):
        template["name"] = metadata.get("proposal_name")
    if metadata.get("proposal_description"):
        template["description"] = metadata.get("proposal_description")
    elif run_record.get("summary"):
        template["description"] = run_record.get("summary")
    if metadata.get("dataset"):
        template["dataset"] = metadata.get("dataset")
    if metadata.get("detector"):
        template["detector"] = metadata.get("detector")
    return template


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
