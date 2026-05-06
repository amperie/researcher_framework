"""Plan implementation step - produce a structured implementation plan per proposal.

Does NOT generate code. Produces a JSON plan that the implement step will use.

Reads:
    state['proposals']

Writes:
    state['implementation_plans']
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from core.graph.nodes.memory import persist_memory_records_for_state
from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils import extract_json_array
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

log = get_logger(__name__)


def plan_implementation_node(state: ResearchState, profile: dict) -> dict:
    proposals = state.get("proposals") or []
    proposal_seed_notes = str(state.get("proposal_seed_planning_notes") or "").strip()

    if not proposals:
        log.warning("plan_implementation_node | No proposals in state")
        return {"implementation_plans": []}

    system_prompt = get_prompt(profile, "plan_implementation")
    llm = get_llm("plan_implementation", profile)

    base_class_docs = _base_class_context(profile)
    scan_constraints = _scan_constraints_context(profile)

    user_content = (
        f"Available base classes:\n{base_class_docs}\n\n"
        f"Scan field constraints:\n{scan_constraints}\n"
        f"{f'Operator proposal-seed notes:\n{proposal_seed_notes}\n\n' if proposal_seed_notes else ''}"
        f"Proposals to plan:\n{json.dumps(proposals, indent=2)}"
    )

    log.info("plan_implementation_node | Planning %d proposals", len(proposals))
    resp = None
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        raw_plans = extract_json_array(resp.content)
        plans, normalization_errors = _normalize_implementation_plans(raw_plans, proposals)
        log.info(
            "plan_implementation_node | Generated %d implementation plans",
            len(plans),
        )
        if normalization_errors:
            log.warning(
                "plan_implementation_node | Malformed implementation-plan entries detected. Raw LLM response follows:\n%s",
                _truncate_text(str(getattr(resp, "content", "") or ""), 12000),
            )
    except Exception as exc:
        log.error("plan_implementation_node | Failed: %s", exc, exc_info=True)
        if resp is not None:
            log.error(
                "plan_implementation_node | Raw LLM response follows:\n%s",
                _truncate_text(str(getattr(resp, "content", "") or ""), 12000),
            )
        return {
            "implementation_plans": [],
            "errors": (state.get("errors") or []) + [f"plan_implementation failed: {exc}"],
        }

    delta = {"implementation_plans": plans}
    if normalization_errors:
        delta["errors"] = (state.get("errors") or []) + normalization_errors
    try:
        persist_memory_records_for_state(profile, {**state, **delta})
    except Exception as exc:
        log.warning("plan_implementation_node | Memory persistence failed: %s", exc)
        delta["errors"] = list(delta.get("errors") or (state.get("errors") or [])) + [f"plan_implementation: memory persistence failed: {exc}"]
    return delta


def _base_class_context(profile: dict) -> str:
    base_classes = profile.get("base_classes") or []
    return "\n\n".join(
        (
            f"Base class: {bc['name']}\n"
            f"Module: {bc.get('module', 'n/a')}\n"
            f"Description: {bc.get('description', '')}\n"
            f"Interface excerpt:\n{_truncate_text(bc.get('key_interface', ''), 2200)}"
        )
        for bc in base_classes
    )


def _scan_constraints_context(profile: dict) -> str:
    datasets = profile.get("datasets") or []
    return "\n".join(
        (
            f"Dataset '{ds['name']}':\n"
            f"  Guaranteed: {(ds.get('available_scan_fields') or {}).get('guaranteed', [])}\n"
            f"  NOT available: {(ds.get('available_scan_fields') or {}).get('not_available', [])}"
        )
        for ds in datasets
    )


def _truncate_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalize_implementation_plans(
    raw_plans: list,
    proposals: list[dict],
) -> tuple[list[dict], list[str]]:
    proposal_names = [str(item.get("name") or "") for item in proposals if isinstance(item, dict)]
    normalized: list[dict] = []
    errors: list[str] = []

    for idx, item in enumerate(raw_plans):
        if isinstance(item, dict):
            plan = dict(item)
            if not plan.get("proposal_name") and idx < len(proposal_names) and proposal_names[idx]:
                plan["proposal_name"] = proposal_names[idx]
            if not plan.get("class_name"):
                seed_name = str(plan.get("proposal_name") or (proposal_names[idx] if idx < len(proposal_names) else "") or "GeneratedPlan")
                plan["class_name"] = _default_class_name(seed_name)
            normalized.append(plan)
            continue

        if isinstance(item, str) and item.strip():
            proposal_name = proposal_names[idx] if idx < len(proposal_names) and proposal_names[idx] else f"proposal_{idx + 1}"
            errors.append(
                f"plan_implementation: dropped malformed plan entry for {proposal_name}: expected object, got string"
            )
            continue

        errors.append(
            f"plan_implementation: dropped malformed plan entry at index {idx}: expected object, got {type(item).__name__}"
        )

    if not normalized and raw_plans:
        errors.append("plan_implementation: no valid implementation-plan objects were returned by the LLM")
    return normalized, errors


def _default_class_name(proposal_name: str) -> str:
    text = "".join(ch if ch.isalnum() else " " for ch in str(proposal_name))
    parts = [part for part in text.split() if part]
    if not parts:
        return "GeneratedPlan"
    return "".join(part[:1].upper() + part[1:] for part in parts)
