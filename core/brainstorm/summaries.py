from __future__ import annotations

from typing import Any


def render_consensus_summary(state: dict[str, Any]) -> str:
    consensus = dict(state.get("consensus") or {})
    summary_cfg = dict(state.get("summary_config") or {})
    max_items = int(summary_cfg.get("max_items_per_section", 3) or 3)
    evidence_chars = int(summary_cfg.get("evidence_summary_chars", 80) or 80)
    sections: list[str] = []
    sections.append(f"Goal: {state.get('current_goal', '')}".strip())
    sections.append(_render_string_list("Agreed", consensus.get("agreed_points") or [], max_items=max_items))
    sections.append(_render_option_list("Ideas", consensus.get("active_options") or [], max_items=max_items))
    sections.append(_render_objections(consensus.get("objections") or [], max_items=max_items))
    sections.append(_render_evidence(consensus.get("evidence") or [], max_items=max_items, summary_chars=evidence_chars))
    sections.append(_render_string_list("Questions", consensus.get("open_questions") or [], max_items=max_items))
    recommendation = str(consensus.get("next_recommendation") or "").strip()
    if recommendation:
        sections.append(f"Next: {recommendation}")
    return "\n".join(section for section in sections if section.strip())


def _render_string_list(title: str, items: list[Any], *, max_items: int) -> str:
    if not items:
        return ""
    lines = [f"{title}:"]
    for item in items[:max_items]:
        lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)


def _render_option_list(title: str, items: list[Any], *, max_items: int) -> str:
    if not items:
        return ""
    lines = [f"{title}:"]
    for item in items[:max_items]:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("title") or item.get("description") or "").strip()
            reason = str(item.get("reason") or item.get("rationale") or "").strip()
            label = name or str(item)
            if reason:
                lines.append(f"  - {label}: {reason}")
            else:
                lines.append(f"  - {label}")
        else:
            lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)


def _render_objections(items: list[Any], *, max_items: int) -> str:
    if not items:
        return ""
    lines = ["Risks:"]
    for item in items[:max_items]:
        if isinstance(item, dict):
            objection = str(item.get("objection") or item.get("text") or item.get("reason") or "").strip()
            owner = str(item.get("role_name") or item.get("role") or "").strip()
            label = f"{owner}: {objection}" if owner and objection else objection
            if label:
                lines.append(f"  - {label}")
        else:
            lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)


def _render_evidence(items: list[Any], *, max_items: int, summary_chars: int) -> str:
    if not items:
        return ""
    lines = ["Evidence:"]
    for item in items[:max_items]:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("artifact_id") or "").strip()
            summary = str(item.get("summary") or "").strip()
            lines.append(f"  - {title}: {summary[:summary_chars]}")
        else:
            lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)
