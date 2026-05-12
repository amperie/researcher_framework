from __future__ import annotations

from typing import Any


def render_consensus_summary(state: dict[str, Any]) -> str:
    consensus = dict(state.get("consensus") or {})
    sections: list[str] = []
    sections.append(f"Current goal: {state.get('current_goal', '')}".strip())
    sections.append(_render_string_list("Agreed points", consensus.get("agreed_points") or []))
    sections.append(_render_option_list("Leading ideas", consensus.get("active_options") or []))
    sections.append(_render_objections(consensus.get("objections") or []))
    sections.append(_render_evidence(consensus.get("evidence") or []))
    sections.append(_render_string_list("Open questions", consensus.get("open_questions") or []))
    recommendation = str(consensus.get("next_recommendation") or "").strip()
    if recommendation:
        sections.append(f"Likely next step: {recommendation}")
    return "\n".join(section for section in sections if section.strip())


def _render_string_list(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    lines = [f"{title}:"]
    for item in items[:5]:
        lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)


def _render_option_list(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    lines = [f"{title}:"]
    for item in items[:5]:
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


def _render_objections(items: list[Any]) -> str:
    if not items:
        return ""
    lines = ["Main objections:"]
    for item in items[:5]:
        if isinstance(item, dict):
            objection = str(item.get("objection") or item.get("text") or item.get("reason") or "").strip()
            owner = str(item.get("role_name") or item.get("role") or "").strip()
            label = f"{owner}: {objection}" if owner and objection else objection
            if label:
                lines.append(f"  - {label}")
        else:
            lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)


def _render_evidence(items: list[Any]) -> str:
    if not items:
        return ""
    lines = ["Evidence gathered:"]
    for item in items[:5]:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("artifact_id") or "").strip()
            summary = str(item.get("summary") or "").strip()
            lines.append(f"  - {title}: {summary[:120]}")
        else:
            lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)
