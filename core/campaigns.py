from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from core.handoffs import (
    load_run_record,
    proposal_template_from_run,
    resolve_proposal_seed,
    save_proposal_seed,
)
from core.memory import MemoryService, build_memory_record
from core.utils.logger import get_logger
from core.utils.profile_loader import load_profile
from main import _add_plugin_to_path, build_initial_state, run_pipeline_graph

log = get_logger(__name__)


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Campaign config must be a YAML object")
    data["_config_path"] = str(config_path)
    return data


def materialize_variants(base_proposal: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants_cfg = cfg.get("variants") or []
    if not isinstance(variants_cfg, list) or not variants_cfg:
        raise ValueError("Campaign config must define a non-empty variants list")

    default_patch = dict((cfg.get("variant_defaults") or {}).get("proposal_patch") or {})
    default_notes = str((cfg.get("variant_defaults") or {}).get("planning_notes") or "").strip()
    rendered: list[dict[str, Any]] = []
    base_name = str(base_proposal.get("name") or "proposal").strip() or "proposal"

    for index, item in enumerate(variants_cfg, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Variant {index} must be a YAML object")
        proposal = _deep_merge_dicts(deepcopy(base_proposal), default_patch)
        proposal = _deep_merge_dicts(proposal, dict(item.get("proposal_patch") or {}))

        name_suffix = str(item.get("name_suffix") or "").strip()
        if name_suffix:
            proposal["name"] = f"{base_name}_{_slug(name_suffix)}"
        elif not proposal.get("name"):
            proposal["name"] = f"{base_name}_{index:02d}"

        title = str(item.get("title") or proposal.get("name") or f"variant_{index:02d}").strip()
        planning_notes = "\n\n".join(part for part in (default_notes, str(item.get("planning_notes") or "").strip()) if part).strip()
        key = str(item.get("key") or _slug(title) or f"variant_{index:02d}").strip()
        rendered.append({
            "key": key,
            "title": title,
            "research_direction": str(item.get("research_direction") or cfg.get("research_direction") or "").strip(),
            "planning_notes": planning_notes,
            "proposal_template": proposal,
            "start_node": str(item.get("start_node") or cfg.get("start_node") or "").strip(),
        })
    return rendered


def run_campaign(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    cfg = load_campaign_config(config_path)
    profile_name = str(cfg.get("profile") or "").strip()
    if not profile_name:
        raise ValueError("Campaign config must include profile")

    profile = load_profile(profile_name)
    _add_plugin_to_path(profile)

    source_run_record, base_seed = _resolve_campaign_base(profile, cfg)
    base_direction = str(
        cfg.get("research_direction")
        or base_seed.get("research_direction")
        or (source_run_record.get("metadata") or {}).get("research_direction")
        or ""
    ).strip()
    if not base_direction:
        raise ValueError("Campaign requires a research_direction, source experiment, or proposal seed")

    base_proposal = dict((base_seed.get("proposals") or [{}])[0] or {})
    if not base_proposal:
        raise ValueError("Unable to resolve a base proposal template for the campaign")

    campaign_id = str(cfg.get("campaign_id") or f"campaign:{uuid4()}").strip()
    campaign_title = str(cfg.get("campaign_title") or campaign_id).strip()
    root_run_family_id = str(cfg.get("root_run_family_id") or campaign_id).strip()
    root_research_direction = str(cfg.get("root_research_direction") or base_direction).strip()
    variants = materialize_variants(base_proposal, cfg)
    source_experiment_record_id = str(
        cfg.get("source_experiment_record_id")
        or base_seed.get("source_experiment_record_id")
        or source_run_record.get("record_id")
        or ""
    ).strip()

    campaign_summary = {
        "campaign_id": campaign_id,
        "campaign_title": campaign_title,
        "profile": profile_name,
        "root_run_family_id": root_run_family_id,
        "root_research_direction": root_research_direction,
        "source_experiment_record_id": source_experiment_record_id,
        "variant_count": len(variants),
        "variants": [
            {
                "key": item["key"],
                "title": item["title"],
                "proposal_name": str(item["proposal_template"].get("name") or ""),
                "research_direction": item["research_direction"] or base_direction,
            }
            for item in variants
        ],
    }
    if dry_run:
        return campaign_summary

    start_node_default = str(cfg.get("start_node") or "").strip() or "plan_implementation"
    campaign_runs: list[dict[str, Any]] = []
    _persist_campaign_record(
        profile,
        campaign_id=campaign_id,
        campaign_title=campaign_title,
        root_run_family_id=root_run_family_id,
        root_research_direction=root_research_direction,
        source_experiment_record_id=source_experiment_record_id,
        status="running",
        variants=campaign_summary["variants"],
        runs=campaign_runs,
    )

    for index, variant in enumerate(variants, 1):
        run_direction = str(variant.get("research_direction") or base_direction)
        proposal_seed_record = save_proposal_seed(
            profile,
            source_run_record,
            {
                "title": f"{campaign_title} :: {variant['title']}",
                "research_direction": run_direction,
                "proposal_template": dict(variant["proposal_template"]),
                "planning_notes": str(variant.get("planning_notes") or ""),
                "snippets": [],
                "campaign_id": campaign_id,
                "campaign_title": campaign_title,
                "campaign_variant_id": variant["key"],
                "campaign_variant_title": variant["title"],
                "campaign_variant_index": index,
                "campaign_size": len(variants),
            },
        )

        seed = {
            "research_direction": run_direction,
            "proposals": [dict(variant["proposal_template"])],
            "proposal_seed_planning_notes": str(variant.get("planning_notes") or ""),
            "source_proposal_seed_record_id": str(proposal_seed_record.get("record_id") or ""),
            "source_proposal_seed_title": str(proposal_seed_record.get("title") or ""),
            "source_experiment_record_id": source_experiment_record_id,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "campaign_id": campaign_id,
            "campaign_title": campaign_title,
            "campaign_variant_id": variant["key"],
            "campaign_variant_title": variant["title"],
            "campaign_variant_index": index,
            "campaign_size": len(variants),
        }
        start_node = str(variant.get("start_node") or start_node_default or "").strip() or "plan_implementation"
        initial_state = build_initial_state(
            profile_name,
            run_direction,
            seed,
            continue_loop=False,
        )
        log.info(
            "campaign | running variant %d/%d key=%r proposal=%r start_node=%r",
            index,
            len(variants),
            variant["key"],
            variant["proposal_template"].get("name"),
            start_node,
        )
        final_state = run_pipeline_graph(
            profile_name,
            profile,
            initial_state=initial_state,
            start_node=start_node,
            print_results=True,
        )
        run_summary = {
            "variant_id": variant["key"],
            "variant_title": variant["title"],
            "proposal_seed_record_id": str(proposal_seed_record.get("record_id") or ""),
            "stored_result_ids": list(final_state.get("stored_result_ids") or []),
            "best_proposal": str((final_state.get("evaluation_summary") or {}).get("best_proposal") or ""),
            "best_metric_name": str((final_state.get("evaluation_summary") or {}).get("best_metric_name") or ""),
            "best_metric_value": (final_state.get("evaluation_summary") or {}).get("best_metric_value"),
            "errors": list(final_state.get("errors") or []),
        }
        campaign_runs.append(run_summary)
        _persist_campaign_record(
            profile,
            campaign_id=campaign_id,
            campaign_title=campaign_title,
            root_run_family_id=root_run_family_id,
            root_research_direction=root_research_direction,
            source_experiment_record_id=source_experiment_record_id,
            status="running",
            variants=campaign_summary["variants"],
            runs=campaign_runs,
        )

    _persist_campaign_record(
        profile,
        campaign_id=campaign_id,
        campaign_title=campaign_title,
        root_run_family_id=root_run_family_id,
        root_research_direction=root_research_direction,
        source_experiment_record_id=source_experiment_record_id,
        status="completed",
        variants=campaign_summary["variants"],
        runs=campaign_runs,
    )
    return {**campaign_summary, "runs": campaign_runs}


def _resolve_campaign_base(profile: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal_seed_record_id = str(cfg.get("proposal_seed_record_id") or "").strip()
    source_experiment_record_id = str(cfg.get("source_experiment_record_id") or "").strip()
    if proposal_seed_record_id:
        seed = resolve_proposal_seed(
            profile,
            source_experiment_record_id=source_experiment_record_id,
            proposal_seed_record_id=proposal_seed_record_id,
        )
        source_record_id = str(seed.get("source_experiment_record_id") or source_experiment_record_id or "").strip()
        source_run_record = load_run_record(profile, source_record_id) if source_record_id else {}
        if not source_run_record:
            raise ValueError(f"Campaign base source run not found: {source_record_id}")
        return source_run_record, seed

    if not source_experiment_record_id:
        raise ValueError("Campaign config must include source_experiment_record_id or proposal_seed_record_id")
    source_run_record = load_run_record(profile, source_experiment_record_id)
    if not source_run_record:
        raise ValueError(f"Campaign base run not found: {source_experiment_record_id}")
    proposal = proposal_template_from_run(profile, source_run_record)
    metadata = dict(source_run_record.get("metadata") or {})
    return source_run_record, {
        "research_direction": str(cfg.get("research_direction") or metadata.get("research_direction") or "").strip(),
        "proposals": [proposal],
        "source_experiment_record_id": source_experiment_record_id,
        "root_run_family_id": str(metadata.get("root_run_family_id") or ""),
        "root_research_direction": str(metadata.get("root_research_direction") or metadata.get("research_direction") or ""),
    }


def _persist_campaign_record(
    profile: dict[str, Any],
    *,
    campaign_id: str,
    campaign_title: str,
    root_run_family_id: str,
    root_research_direction: str,
    source_experiment_record_id: str,
    status: str,
    variants: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> None:
    service = MemoryService.for_profile(profile)
    record = build_memory_record(
        profile=profile,
        object_type="campaign",
        payload={
            "campaign_id": campaign_id,
            "campaign_title": campaign_title,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "source_experiment_record_id": source_experiment_record_id,
            "status": status,
            "variants": variants,
            "runs": runs,
        },
        node="run_campaign",
        kind="campaign",
        object_key=campaign_id,
        object_role="orchestration",
        title=campaign_title,
        summary=f"Campaign {campaign_title} status={status} variants={len(variants)} runs={len(runs)}",
        metadata={
            "profile": str(profile.get("name") or ""),
            "campaign_id": campaign_id,
            "campaign_title": campaign_title,
            "root_run_family_id": root_run_family_id,
            "root_research_direction": root_research_direction,
            "source_experiment_record_id": source_experiment_record_id,
            "status": status,
            "variant_count": len(variants),
            "run_count": len(runs),
        },
        tags=[str(profile.get("name") or ""), "campaign"],
        source_record_ids=[source_experiment_record_id] if source_experiment_record_id else [],
        entities=[
            {
                "entity_type": "campaign",
                "key": campaign_id,
                "name": campaign_title,
                "metadata": {
                    "domain": str(profile.get("name") or ""),
                    "status": status,
                },
            }
        ],
        relations=[
            {
                "relation_type": "campaign_branches_from",
                "source_type": "campaign",
                "source_key": campaign_title or campaign_id,
                "target_type": "experiment_result",
                "target_key": source_experiment_record_id,
                "metadata": {"domain": str(profile.get("name") or "")},
            }
        ] if source_experiment_record_id else [],
    )
    service.persist_record(record)


def _deep_merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge_dicts(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _slug(value: str) -> str:
    chars = []
    last_sep = False
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
            last_sep = False
            continue
        if not last_sep:
            chars.append("_")
            last_sep = True
    return "".join(chars).strip("_")
