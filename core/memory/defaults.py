"""Default memory record builders and retrieval mappers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.memory.fingerprints import fingerprint_json
from core.memory.models import MemoryEntity, MemoryObjectSpec, MemoryRecord, MemoryRelation
from core.utils.logger import get_logger

log = get_logger(__name__)


def build_core_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic canonical memory records from the current graph state."""
    records: list[MemoryRecord] = []
    records.extend(build_research_memory_records(profile, state))
    records.extend(build_idea_memory_records(profile, state))
    records.extend(build_refined_idea_memory_records(profile, state))
    records.extend(build_proposal_memory_records(profile, state))
    records.extend(build_implementation_plan_memory_records(profile, state))
    records.extend(build_implementation_memory_records(profile, state))
    records.extend(build_validation_memory_records(profile, state))
    records.extend(build_experiment_artifact_memory_records(profile, state))
    records.extend(build_experiment_memory_records(profile, state))
    records.extend(build_evaluation_memory_records(profile, state))
    records.extend(build_next_step_memory_records(profile, state))
    deduped = dedupe_memory_records(records)
    log.debug(
        "memory.defaults | Built %d generic record(s), %d after dedupe profile=%r",
        len(records),
        len(deduped),
        profile.get("name"),
    )
    return deduped


def ensure_memory_record_defaults(record: MemoryRecord, *, node: str = "") -> MemoryRecord:
    """Fill newer canonical fields on older/ad-hoc record builders."""
    hydrated = dict(record)
    metadata = dict(hydrated.get("metadata") or {})
    domain = str(hydrated.get("domain") or metadata.get("profile") or "")
    object_type = str(hydrated.get("object_type") or hydrated.get("kind") or "memory")
    hydrated.setdefault("domain", domain)
    hydrated.setdefault("object_type", object_type)
    hydrated.setdefault("object_key", str(hydrated.get("record_id") or ""))
    hydrated.setdefault("object_role", "artifact")
    hydrated.setdefault("schema_version", "1")
    hydrated.setdefault("content", {})
    hydrated.setdefault("metadata", metadata)
    hydrated.setdefault("tags", [domain, object_type] if domain else [object_type])
    hydrated.setdefault("created_at", _now_iso())
    hydrated.setdefault("lineage", {
        "node": node,
        "source_state_keys": [],
        "source_record_ids": [],
    })
    hydrated.setdefault("validity", {
        "status": str(metadata.get("status") or metadata.get(f"{object_type}_status") or "ready"),
        "checked_at": hydrated.get("created_at"),
        "checks": {},
    })
    hydrated.setdefault("blob_refs", [])
    hydrated.setdefault("entities", [])
    hydrated.setdefault("relations", [])
    if hydrated.keys() != record.keys():
        log.debug(
            "memory.defaults | Normalized record id=%r with canonical defaults",
            hydrated.get("record_id"),
        )
    return hydrated


def memory_object_specs(profile: dict[str, Any]) -> dict[str, MemoryObjectSpec]:
    """Return profile-declared memory object specs keyed by object_type."""
    specs: dict[str, MemoryObjectSpec] = {}
    for raw in ((profile.get("memory") or {}).get("objects") or []):
        if not isinstance(raw, dict) or not raw.get("object_type"):
            log.warning("memory.defaults | Ignoring invalid memory object spec in profile=%r: %r", profile.get("name"), raw)
            continue
        spec: MemoryObjectSpec = dict(raw)
        spec.setdefault("schema_version", "1")
        spec.setdefault("fingerprint_metadata_key", "fingerprint")
        spec.setdefault("status_metadata_key", "status")
        spec.setdefault("ready_statuses", ["ready"])
        specs[str(spec["object_type"])] = spec
    log.debug("memory.defaults | Loaded %d memory object spec(s) for profile=%r", len(specs), profile.get("name"))
    return specs


def memory_object_spec(profile: dict[str, Any], object_type: str) -> MemoryObjectSpec | None:
    """Return one profile-declared memory object spec, if present."""
    return memory_object_specs(profile).get(object_type)


def fingerprint_for_spec(spec: MemoryObjectSpec, payload: dict[str, Any]) -> str:
    """Compute a deterministic fingerprint from a spec's dotted payload fields."""
    fields = list(spec.get("fingerprint_fields") or [])
    if not fields:
        log.debug("memory.defaults | Fingerprinting full payload for object_type=%r", spec.get("object_type"))
        return fingerprint_json(payload)
    selected = {field: _get_dotted(payload, field) for field in fields}
    log.debug(
        "memory.defaults | Fingerprinting spec object_type=%r fields=%s",
        spec.get("object_type"),
        fields,
    )
    return fingerprint_json(selected)


def build_memory_record(
    *,
    profile: dict[str, Any],
    object_type: str,
    payload: dict[str, Any],
    node: str = "",
    kind: str = "",
    object_key: str = "",
    object_role: str = "artifact",
    title: str = "",
    summary: str = "",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    spec: MemoryObjectSpec | None = None,
    source_state_keys: list[str] | None = None,
    source_record_ids: list[str] | None = None,
    blob_refs: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> MemoryRecord:
    """Build a canonical memory record from a typed node emission."""
    domain = str(profile.get("name") or "")
    object_spec = spec or memory_object_spec(profile, object_type) or {}
    record_kind = kind or str(object_spec.get("kind") or object_type)
    record_metadata = dict(metadata or {})
    if object_spec.get("fingerprint_fields"):
        fingerprint_key = str(object_spec.get("fingerprint_metadata_key") or "fingerprint")
        record_metadata.setdefault(fingerprint_key, fingerprint_for_spec(object_spec, payload))
    fingerprint = str(record_metadata.get(str(object_spec.get("fingerprint_metadata_key") or "fingerprint")) or "")
    key = object_key or fingerprint or str(payload.get("id") or payload.get("name") or title or object_type)
    record_id = str(payload.get("record_id") or f"{object_type}:{key}")
    created_at = _now_iso()
    log.debug(
        "memory.defaults | Building memory record id=%r domain=%r kind=%r object_type=%r",
        record_id,
        domain,
        record_kind,
        object_type,
    )

    return {
        "record_id": record_id,
        "domain": domain,
        "kind": record_kind,
        "object_type": object_type,
        "object_key": key,
        "object_role": object_role,
        "schema_version": str(object_spec.get("schema_version") or "1"),
        "title": title or str(payload.get("title") or payload.get("name") or key),
        "summary": summary or _compact_payload_summary(payload),
        "content": dict(payload),
        "metadata": {
            "profile": domain,
            "memory_kind": record_kind,
            **record_metadata,
        },
        "tags": list(tags or [domain, object_type]),
        "created_at": created_at,
        "lineage": {
            "node": node,
            "source_state_keys": list(source_state_keys or []),
            "source_record_ids": list(source_record_ids or []),
        },
        "validity": {
            "status": str(record_metadata.get(str(object_spec.get("status_metadata_key") or "status")) or "ready"),
            "reusable": bool(object_spec.get("reusable", False)),
            "checked_at": created_at,
            "checks": {},
        },
        "blob_refs": list(blob_refs or []),
        "entities": list(entities or []),
        "relations": list(relations or []),
    }


def build_experiment_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build canonical memory records from generic experiment results."""
    records: list[MemoryRecord] = []
    experiment_results = state.get("experiment_results") or []
    models = state.get("models") or []
    evaluation_summary = state.get("evaluation_summary") or {}
    direction = state.get("research_direction", "")
    root_run_family_id = str(state.get("root_run_family_id") or "")
    root_research_direction = str(state.get("root_research_direction") or direction or "")
    source_next_step_record_id = str(state.get("source_next_step_record_id") or "")
    source_next_step_title = str(state.get("source_next_step_title") or "")
    source_proposal_seed_record_id = str(state.get("source_proposal_seed_record_id") or "")
    source_proposal_seed_title = str(state.get("source_proposal_seed_title") or "")
    campaign_id = str(state.get("campaign_id") or "")
    campaign_title = str(state.get("campaign_title") or "")
    campaign_variant_id = str(state.get("campaign_variant_id") or "")
    campaign_variant_title = str(state.get("campaign_variant_title") or "")
    campaign_variant_index = int(state.get("campaign_variant_index") or 0)
    campaign_size = int(state.get("campaign_size") or 0)
    model_by_exp = {m.get("experiment_id"): m for m in models}

    for result in experiment_results:
        experiment_id = str(result.get("experiment_id") or "")
        proposal_name = str(result.get("proposal_name") or "unknown")
        proposal = result.get("proposal") or {}
        metrics = result.get("metrics") or {}
        model = model_by_exp.get(experiment_id) or result.get("model") or {}
        inserted_at = datetime.now(timezone.utc).isoformat()
        assessment = _proposal_assessment(proposal_name, evaluation_summary)
        lessons = _proposal_lessons(proposal_name, evaluation_summary)
        hypothesis_supported = _proposal_hypothesis_supported(proposal_name, evaluation_summary)
        summary = _build_memory_summary(
            profile=profile,
            direction=direction,
            proposal_name=proposal_name,
            proposal=proposal,
            metrics=metrics,
            assessment=assessment,
            lessons=lessons,
            hypothesis_supported=hypothesis_supported,
        )

        record: MemoryRecord = {
            "record_id": experiment_id or proposal_name,
            "domain": str(profile.get("name") or ""),
            "kind": "prior_experiment",
            "object_type": "experiment_result",
            "object_key": experiment_id or proposal_name,
            "object_role": "result",
            "schema_version": "1",
            "title": proposal_name,
            "summary": summary,
            "content": {
                "experiment_id": experiment_id,
                "proposal_name": proposal_name,
                "proposal": proposal,
                "metrics": metrics,
                "model": model,
                "evaluation_summary": evaluation_summary,
                "research_direction": direction,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "campaign_id": campaign_id,
                "campaign_title": campaign_title,
                "campaign_variant_id": campaign_variant_id,
                "campaign_variant_title": campaign_variant_title,
                "campaign_variant_index": campaign_variant_index,
                "campaign_size": campaign_size,
                "artifacts": result.get("artifacts") or {},
            },
            "metadata": {
                "experiment_id": experiment_id,
                "proposal_name": proposal_name,
                "profile": profile.get("name", ""),
                "research_direction": direction,
                "proposal_description": proposal.get("description", ""),
                "dataset": proposal.get("dataset", ""),
                "detector": proposal.get("detector", ""),
                "assessment": assessment,
                "hypothesis_supported": hypothesis_supported,
                "lessons": lessons,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "inserted_at": inserted_at,
                "mlflow_run_id": result.get("mlflow_run_id") or model.get("mlflow_run_id") or "",
                "source_next_step_record_id": source_next_step_record_id,
                "source_next_step_title": source_next_step_title,
                "source_proposal_seed_record_id": source_proposal_seed_record_id,
                "source_proposal_seed_title": source_proposal_seed_title,
                "campaign_id": campaign_id,
                "campaign_title": campaign_title,
                "campaign_variant_id": campaign_variant_id,
                "campaign_variant_title": campaign_variant_title,
                "campaign_variant_index": campaign_variant_index,
                "campaign_size": campaign_size,
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
            },
            "tags": [str(profile.get("name") or ""), "prior_experiment"],
            "created_at": inserted_at,
            "source_run_id": str(result.get("mlflow_run_id") or model.get("mlflow_run_id") or "") or None,
            "entities": _experiment_entities(profile, proposal_name, proposal, experiment_id or proposal_name),
            "relations": _experiment_relations(
                profile,
                proposal_name,
                proposal,
                experiment_id or proposal_name,
                source_next_step_record_id=source_next_step_record_id,
                source_next_step_title=source_next_step_title,
                campaign_id=campaign_id,
                campaign_title=campaign_title,
                campaign_variant_id=campaign_variant_id,
                campaign_variant_title=campaign_variant_title,
            ),
        }
        records.append(record)
    return records


def build_research_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for research summaries and retrieved artifacts."""
    domain = str(profile.get("name") or "")
    direction = str(state.get("research_direction") or "")
    summary = str(state.get("research_summary") or "")
    artifacts = state.get("research_artifacts") or []
    papers = state.get("research_papers") or []
    digests = state.get("paper_digests") or []
    records: list[MemoryRecord] = []

    if summary or artifacts or papers or digests:
        summary_fingerprint = fingerprint_json({
            "domain": domain,
            "direction": direction,
            "summary": summary,
            "artifact_ids": [item.get("artifact_id") for item in artifacts if isinstance(item, dict)],
            "paper_ids": [item.get("arxiv_id") or item.get("title") for item in papers if isinstance(item, dict)],
            "digest_titles": [item.get("title") for item in digests if isinstance(item, dict)],
        })
        records.append({
            "record_id": f"research_summary:{summary_fingerprint}",
            "domain": domain,
            "kind": "research_summary",
            "object_type": "research_summary",
            "object_key": summary_fingerprint,
            "object_role": "context",
            "schema_version": "1",
            "title": direction or "research_summary",
            "summary": summary or f"Research context for: {direction}",
            "content": {
                "research_direction": direction,
                "research_summary": summary,
                "research_artifacts": artifacts,
                "research_papers": papers,
                "paper_digests": digests,
            },
            "metadata": {
                "profile": domain,
                "research_direction": direction,
                "n_artifacts": len(artifacts),
                "n_papers": len(papers),
                "n_digests": len(digests),
            },
            "tags": [domain, "research_summary"],
            "created_at": _now_iso(),
            "entities": _research_summary_entities(domain, direction, artifacts, papers, digests),
            "relations": _research_summary_relations(domain, direction, artifacts, papers, digests),
        })

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or fingerprint_json(artifact))
        records.append({
            "record_id": f"research_artifact:{artifact_id}",
            "domain": domain,
            "kind": "research_artifact",
            "object_type": "research_artifact",
            "object_key": artifact_id,
            "object_role": "context",
            "schema_version": "1",
            "title": str(artifact.get("title") or artifact_id),
            "summary": str(artifact.get("summary") or ""),
            "content": dict(artifact),
            "metadata": {
                "profile": domain,
                "research_direction": direction,
                "source": artifact.get("source", ""),
                "source_type": artifact.get("source_type", ""),
                "relevance_score": artifact.get("relevance_score"),
                "url": artifact.get("url", ""),
            },
            "tags": [domain, "research_artifact", str(artifact.get("source_type") or "unknown")],
            "created_at": _now_iso(),
            "entities": _research_artifact_entities(domain, direction, artifact_id, artifact),
            "relations": _research_artifact_relations(domain, direction, artifact_id, artifact),
        })
    return records


def build_idea_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for brainstormed ideas."""
    ideas = state.get("ideas") or []
    direction = state.get("research_direction", "")
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        idea_name = str(idea.get("name") or idea.get("title") or "idea")
        idea_fingerprint = fingerprint_json({
            "domain": domain,
            "direction": direction,
            "idea": idea,
        })
        records.append({
            "record_id": f"idea:{idea_fingerprint}",
            "domain": domain,
            "kind": "idea",
            "object_type": "idea",
            "object_key": idea_name,
            "object_role": "plan",
            "schema_version": "1",
            "title": idea_name,
            "summary": (
                f"Direction: {direction}\n"
                f"Description: {idea.get('description', '')}\n"
                f"Hypothesis: {idea.get('hypothesis', '')}\n"
                f"Rationale: {idea.get('rationale', '')}"
            ).strip(),
            "content": dict(idea),
            "metadata": {
                "profile": domain,
                "research_direction": direction,
                "idea_name": idea_name,
            },
            "tags": [domain, "idea"],
            "created_at": _now_iso(),
            "entities": _idea_entities(domain, direction, idea_name),
            "relations": _idea_relations(domain, direction, idea_name),
        })
    return records


def build_refined_idea_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for refined ideas."""
    ideas = state.get("refined_ideas") or []
    direction = state.get("research_direction", "")
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        idea_name = str(idea.get("name") or idea.get("title") or "idea")
        idea_fingerprint = fingerprint_json({
            "domain": domain,
            "direction": direction,
            "idea": idea,
        })
        records.append({
            "record_id": f"refined_idea:{idea_fingerprint}",
            "domain": domain,
            "kind": "refined_idea",
            "object_type": "refined_idea",
            "object_key": idea_name,
            "object_role": "plan",
            "schema_version": "1",
            "title": idea_name,
            "summary": (
                f"Direction: {direction}\n"
                f"Description: {idea.get('description', '')}\n"
                f"Hypothesis: {idea.get('hypothesis', '')}\n"
                f"Rationale: {idea.get('rationale', '')}"
            ).strip(),
            "content": dict(idea),
            "metadata": {
                "profile": domain,
                "research_direction": direction,
                "idea_name": idea_name,
            },
            "tags": [domain, "refined_idea"],
            "created_at": _now_iso(),
            "entities": _refined_idea_entities(domain, direction, idea_name),
            "relations": _refined_idea_relations(domain, direction, idea_name),
        })
    return records


def build_proposal_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for experiment proposals."""
    proposals = state.get("proposals") or []
    direction = state.get("research_direction", "")
    root_run_family_id = str(state.get("root_run_family_id") or "")
    root_research_direction = str(state.get("root_research_direction") or direction or "")
    domain = str(profile.get("name") or "")
    source_next_step_record_id = str(state.get("source_next_step_record_id") or "")
    source_next_step_title = str(state.get("source_next_step_title") or "")
    source_proposal_seed_record_id = str(state.get("source_proposal_seed_record_id") or "")
    source_proposal_seed_title = str(state.get("source_proposal_seed_title") or "")
    campaign_id = str(state.get("campaign_id") or "")
    campaign_title = str(state.get("campaign_title") or "")
    campaign_variant_id = str(state.get("campaign_variant_id") or "")
    campaign_variant_title = str(state.get("campaign_variant_title") or "")
    campaign_variant_index = int(state.get("campaign_variant_index") or 0)
    campaign_size = int(state.get("campaign_size") or 0)
    records: list[MemoryRecord] = []

    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        proposal_name = str(proposal.get("name") or "proposal")
        proposal_fingerprint = fingerprint_json({
            "domain": domain,
            "direction": direction,
            "proposal": proposal,
        })
        records.append({
            "record_id": f"proposal:{proposal_fingerprint}",
            "domain": domain,
            "kind": "proposal",
            "object_type": "proposal",
            "object_key": proposal_name,
            "object_role": "plan",
            "schema_version": "1",
            "title": proposal_name,
            "summary": (
                f"Direction: {direction}\n"
                f"Dataset: {proposal.get('dataset', '')}\n"
                f"Detector: {proposal.get('detector', '')}\n"
                f"Description: {proposal.get('description', '')}"
            ).strip(),
            "content": dict(proposal),
            "metadata": {
                "profile": domain,
                "research_direction": direction,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "proposal_name": proposal_name,
                "dataset": proposal.get("dataset", ""),
                "detector": proposal.get("detector", ""),
                "source_next_step_record_id": source_next_step_record_id,
                "source_next_step_title": source_next_step_title,
                "source_proposal_seed_record_id": source_proposal_seed_record_id,
                "source_proposal_seed_title": source_proposal_seed_title,
                "campaign_id": campaign_id,
                "campaign_title": campaign_title,
                "campaign_variant_id": campaign_variant_id,
                "campaign_variant_title": campaign_variant_title,
                "campaign_variant_index": campaign_variant_index,
                "campaign_size": campaign_size,
            },
            "tags": [domain, "proposal"],
            "created_at": _now_iso(),
            "entities": _proposal_entities(domain, direction, proposal_name, proposal),
            "relations": _proposal_relations(
                domain,
                direction,
                proposal_name,
                proposal,
                source_next_step_record_id=source_next_step_record_id,
                source_next_step_title=source_next_step_title,
                source_proposal_seed_record_id=source_proposal_seed_record_id,
                source_proposal_seed_title=source_proposal_seed_title,
                campaign_id=campaign_id,
                campaign_title=campaign_title,
                campaign_variant_id=campaign_variant_id,
                campaign_variant_title=campaign_variant_title,
            ),
        })
    return records


def build_implementation_plan_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for implementation plans."""
    plans = state.get("implementation_plans") or []
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for plan in plans:
        if not isinstance(plan, dict):
            continue
        proposal_name = str(plan.get("proposal_name") or plan.get("class_name") or "plan")
        class_name = str(plan.get("class_name") or proposal_name)
        plan_fingerprint = fingerprint_json({
            "domain": domain,
            "plan": plan,
        })
        records.append({
            "record_id": f"implementation_plan:{plan_fingerprint}",
            "domain": domain,
            "kind": "implementation_plan",
            "object_type": "implementation_plan",
            "object_key": proposal_name,
            "object_role": "plan",
            "schema_version": "1",
            "title": class_name,
            "summary": (
                f"Proposal: {proposal_name}\n"
                f"Class: {class_name}\n"
                f"Base class: {plan.get('base_class', '')}\n"
                f"Main method: {plan.get('main_method', '')}"
            ).strip(),
            "content": dict(plan),
            "metadata": {
                "profile": domain,
                "proposal_name": proposal_name,
                "class_name": class_name,
                "base_class": plan.get("base_class", ""),
            },
            "tags": [domain, "implementation_plan"],
            "created_at": _now_iso(),
            "entities": _implementation_plan_entities(domain, proposal_name, class_name, plan),
            "relations": _implementation_plan_relations(domain, proposal_name, class_name, plan),
        })
    return records


def build_validation_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for validation outcomes."""
    validations = state.get("validation_results") or []
    implementations = state.get("implementations") or []
    impl_by_class = {
        impl.get("class_name"): impl
        for impl in implementations
        if isinstance(impl, dict) and impl.get("class_name")
    }
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for validation in validations:
        if not isinstance(validation, dict):
            continue
        class_name = str(validation.get("class_name") or "unknown")
        implementation = impl_by_class.get(class_name) or {}
        proposal_name = implementation.get("proposal_name") or validation.get("proposal_name") or class_name
        records.append({
            "record_id": f"validation:{class_name}",
            "domain": domain,
            "kind": "validation_result",
            "object_type": "validation_result",
            "object_key": class_name,
            "object_role": "validation",
            "schema_version": "1",
            "title": class_name,
            "summary": (
                f"Proposal: {proposal_name}\n"
                f"Passed: {validation.get('passed')}\n"
                f"Attempts: {validation.get('attempts', 0)}\n"
                f"Test source: {validation.get('test_source', '')}"
            ).strip(),
            "content": dict(validation),
            "metadata": {
                "profile": domain,
                "proposal_name": proposal_name,
                "class_name": class_name,
                "passed": validation.get("passed"),
                "attempts": validation.get("attempts", 0),
                "test_source": validation.get("test_source", ""),
                "script_path": implementation.get("script_path", validation.get("script_path", "")),
            },
            "tags": [domain, "validation_result"],
            "created_at": _now_iso(),
            "blob_refs": _blob_refs_from_metadata_targets(
                validation,
                (
                    ("validation_result", "stored_artifact_uri", "stored_artifact_id", "application/json"),
                    ("validation_test", "test_file_artifact_uri", "test_file_artifact_id", "text/x-python"),
                    ("implementation", "implementation_artifact_uri", "implementation_artifact_id", "text/x-python"),
                ),
            ),
            "entities": _validation_entities(domain, proposal_name, class_name),
            "relations": _validation_relations(domain, proposal_name, class_name, validation),
        })
    return records


def build_implementation_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for generated implementations."""
    implementations = state.get("implementations") or []
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for implementation in implementations:
        if not isinstance(implementation, dict):
            continue
        class_name = str(implementation.get("class_name") or "unknown")
        proposal_name = str(implementation.get("proposal_name") or class_name)
        script_path = str(implementation.get("script_path") or "")
        if not script_path:
            continue
        records.append({
            "record_id": f"implementation:{class_name}",
            "domain": domain,
            "kind": "implementation",
            "object_type": "implementation",
            "object_key": class_name,
            "object_role": "artifact",
            "schema_version": "1",
            "title": class_name,
            "summary": (
                f"Proposal: {proposal_name}\n"
                f"Class: {class_name}\n"
                f"Validated: {implementation.get('validated')}\n"
                f"Cached: {implementation.get('cached')}"
            ).strip(),
            "content": dict(implementation),
            "metadata": {
                "profile": domain,
                "proposal_name": proposal_name,
                "class_name": class_name,
                "script_path": script_path,
                "validated": implementation.get("validated"),
                "cached": implementation.get("cached"),
                "stored_artifact_id": implementation.get("stored_artifact_id", ""),
                "stored_artifact_uri": implementation.get("stored_artifact_uri", ""),
            },
            "tags": [domain, "implementation"],
            "created_at": _now_iso(),
            "blob_refs": _blob_refs_from_metadata_targets(
                implementation,
                (
                    ("implementation", "stored_artifact_uri", "stored_artifact_id", "text/x-python"),
                ),
            ),
            "entities": _implementation_entities(domain, proposal_name, class_name),
            "relations": _implementation_relations(domain, proposal_name, class_name),
        })
    return records


def build_experiment_artifact_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for prepared experiment artifacts."""
    artifacts = state.get("experiment_artifacts") or state.get("datasets") or []
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        proposal_name = str(artifact.get("proposal_name") or "unknown")
        artifact_id = str(artifact.get("artifact_id") or artifact.get("dataset_id") or fingerprint_json(artifact))
        artifact_type = str(artifact.get("artifact_type") or artifact.get("type") or "experiment_artifact")
        records.append({
            "record_id": f"artifact:{artifact_id}",
            "domain": domain,
            "kind": artifact_type,
            "object_type": artifact_type,
            "object_key": artifact_id,
            "object_role": "artifact",
            "schema_version": "1",
            "title": proposal_name,
            "summary": (
                f"Proposal: {proposal_name}\n"
                f"Artifact type: {artifact_type}\n"
                f"Status: {artifact.get('status', '')}"
            ).strip(),
            "content": dict(artifact),
            "metadata": {
                "profile": domain,
                "proposal_name": proposal_name,
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "status": artifact.get("status", ""),
                "dataset": artifact.get("dataset", ""),
                "detector": artifact.get("detector", ""),
            },
            "tags": [domain, artifact_type],
            "created_at": _now_iso(),
            "blob_refs": _blob_refs_from_metadata_targets(
                artifact,
                (
                    (artifact_type, "stored_artifact_uri", "stored_artifact_id", None),
                ),
            ),
            "entities": _artifact_entities(domain, proposal_name, artifact_id, artifact_type, artifact),
            "relations": _artifact_relations(domain, proposal_name, artifact_id, artifact_type, artifact),
        })
    return records


def build_evaluation_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for evaluation summaries."""
    evaluation = state.get("evaluation_summary") or {}
    if not evaluation:
        return []
    domain = str(profile.get("name") or "")
    direction = str(state.get("research_direction") or "")
    evaluation_fingerprint = fingerprint_json({
        "domain": domain,
        "direction": direction,
        "evaluation": evaluation,
    })
    return [{
        "record_id": f"evaluation:{evaluation_fingerprint}",
        "domain": domain,
        "kind": "evaluation_summary",
        "object_type": "evaluation_summary",
        "object_key": evaluation_fingerprint,
        "object_role": "summary",
        "schema_version": "1",
        "title": evaluation.get("best_proposal") or "evaluation_summary",
        "summary": (
            f"Direction: {direction}\n"
            f"Best proposal: {evaluation.get('best_proposal')}\n"
            f"Best metric: {evaluation.get('best_metric_name')}={evaluation.get('best_metric_value')}\n"
            f"Experiments: {evaluation.get('n_experiments')}"
        ).strip(),
        "content": dict(evaluation),
        "metadata": {
            "profile": domain,
            "research_direction": direction,
            "best_proposal": evaluation.get("best_proposal"),
            "best_metric_name": evaluation.get("best_metric_name"),
            "best_metric_value": evaluation.get("best_metric_value"),
            "n_experiments": evaluation.get("n_experiments"),
        },
        "tags": [domain, "evaluation_summary"],
        "created_at": _now_iso(),
        "entities": _evaluation_entities(domain, direction, evaluation),
        "relations": _evaluation_relations(domain, direction, evaluation),
    }]


def build_next_step_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for proposed next steps."""
    next_steps = state.get("next_steps") or []
    direction = state.get("research_direction", "")
    root_run_family_id = str(state.get("root_run_family_id") or "")
    root_research_direction = str(state.get("root_research_direction") or direction or "")
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for step in next_steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or step.get("suggested_direction") or "next_step")
        records.append({
            "record_id": next_step_record_id(domain, direction, step),
            "domain": domain,
            "kind": "next_step",
            "object_type": "next_step",
            "object_key": title,
            "object_role": "recommendation",
            "schema_version": "1",
            "title": title,
            "summary": (
                f"Direction: {direction}\n"
                f"Suggested direction: {step.get('suggested_direction', '')}\n"
                f"Rationale: {step.get('rationale', '')}\n"
                f"Priority: {step.get('priority', '')}"
            ).strip(),
            "content": dict(step),
            "metadata": {
                "profile": domain,
                "research_direction": direction,
                "root_run_family_id": root_run_family_id,
                "root_research_direction": root_research_direction,
                "priority": step.get("priority"),
            },
            "tags": [domain, "next_step"],
            "created_at": _now_iso(),
            "entities": _next_step_entities(domain, direction, title, step),
            "relations": _next_step_relations(domain, direction, title, step),
        })
    return records


def next_step_record_id(domain: str, direction: str, step: dict[str, Any]) -> str:
    step_fingerprint = fingerprint_json({
        "domain": domain,
        "direction": direction,
        "next_step": step,
    })
    return f"next_step:{step_fingerprint}"


def default_memory_record_to_artifact(record: MemoryRecord, *, source_name: str, distance: float | None = None) -> dict[str, Any]:
    """Convert a canonical memory record into a generic research artifact."""
    metadata = dict(record.get("metadata") or {})
    assessment = metadata.get("assessment")
    lessons = metadata.get("lessons") or []
    direction = metadata.get("research_direction")
    title_parts = [record.get("title") or record.get("record_id", "memory")]
    if assessment:
        title_parts.append(f"[{assessment}]")

    summary = str(record.get("summary") or "")
    if direction and "Direction:" not in summary:
        summary = f"Direction: {direction}\n{summary}".strip()
    if lessons:
        lesson_lines = "\n".join(f"- {lesson}" for lesson in lessons[:3])
        summary = f"{summary}\nKey lessons:\n{lesson_lines}".strip()

    return {
        "artifact_id": f"memory:{record.get('record_id', 'unknown')}",
        "source": source_name,
        "source_type": record.get("kind", "memory"),
        "title": " ".join(title_parts),
        "summary": summary,
        "metadata": {
            **metadata,
            "domain": record.get("domain", ""),
            "kind": record.get("kind", ""),
            "object_type": record.get("object_type", ""),
            "object_key": record.get("object_key", ""),
            "object_role": record.get("object_role", ""),
            "distance": distance,
        },
        "raw": record,
    }


def default_memory_projection(record: MemoryRecord) -> dict[str, Any]:
    """Build default vector and graph projections for a canonical memory record."""
    metadata = dict(record.get("metadata") or {})
    domain = str(record.get("domain") or metadata.get("profile") or "")
    return {
        "embedding_text": _embedding_text(record),
        "vector_metadata": {
            "record_id": record.get("record_id", ""),
            "domain": domain,
            "memory_kind": record.get("kind", ""),
            "object_type": record.get("object_type", ""),
            "object_key": record.get("object_key", ""),
            "object_role": record.get("object_role", ""),
            "schema_version": record.get("schema_version", "1"),
            "title": record.get("title", ""),
            "tags": "|".join(str(tag) for tag in (record.get("tags") or [])[:20]),
            **_scalar_metadata(metadata),
            "memory_summary": record.get("summary", ""),
        },
        "kg_update": {"record_id": str(record.get("record_id") or ""), "domain": domain, "nodes": [], "relations": []},
    }


def record_from_vector_hit(hit: dict[str, Any]) -> MemoryRecord:
    """Fallback record synthesis when the document store cannot hydrate a full record."""
    metadata = dict(hit.get("metadata") or {})
    return {
        "record_id": str(hit.get("id") or metadata.get("record_id") or "unknown"),
        "domain": str(metadata.get("domain") or metadata.get("profile") or ""),
        "kind": str(metadata.get("memory_kind") or "memory"),
        "object_type": str(metadata.get("object_type") or ""),
        "object_key": str(metadata.get("object_key") or ""),
        "object_role": str(metadata.get("object_role") or ""),
        "schema_version": str(metadata.get("schema_version") or "1"),
        "title": str(metadata.get("title") or metadata.get("proposal_name") or hit.get("id") or "memory"),
        "summary": str(metadata.get("memory_summary") or hit.get("document") or ""),
        "content": {},
        "metadata": metadata,
        "tags": list(metadata.get("tags") or []),
        "created_at": str(metadata.get("created_at") or metadata.get("inserted_at") or ""),
    }


def _embedding_text(record: MemoryRecord) -> str:
    title = str(record.get("title") or "")
    summary = str(record.get("summary") or "")
    content = record.get("content") or {}
    extra_parts: list[str] = []
    direction = (record.get("metadata") or {}).get("research_direction")
    if direction:
        extra_parts.append(f"Direction: {direction}")
    proposal = content.get("proposal") if isinstance(content, dict) else None
    if isinstance(proposal, dict):
        if proposal.get("description"):
            extra_parts.append(f"Description: {proposal['description']}")
        if proposal.get("dataset"):
            extra_parts.append(f"Dataset: {proposal['dataset']}")
        if proposal.get("detector"):
            extra_parts.append(f"Detector: {proposal['detector']}")
    return "\n".join(part for part in [title, summary, *extra_parts] if part).strip()


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            compact[key] = value
        elif isinstance(value, bool):
            compact[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value[:20]):
            compact[key] = " | ".join(str(item) for item in value[:20])
    return compact


def dedupe_memory_records(records: list[MemoryRecord]) -> list[MemoryRecord]:
    deduped: dict[str, MemoryRecord] = {}
    dropped = 0
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            dropped += 1
            continue
        if record_id in deduped:
            log.debug("memory.defaults | Dedupe replacing duplicate record id=%r", record_id)
        deduped[record_id] = record
    if dropped:
        log.warning("memory.defaults | Dropped %d memory record(s) without record_id", dropped)
    log.debug(
        "memory.defaults | Dedupe complete kept=%d ids=%s",
        len(deduped),
        sorted(deduped.keys()),
    )
    return list(deduped.values())


def _build_memory_summary(
    *,
    profile: dict[str, Any],
    direction: str,
    proposal_name: str,
    proposal: dict[str, Any],
    metrics: dict[str, Any],
    assessment: str,
    lessons: list[str],
    hypothesis_supported: bool | None,
) -> str:
    lines = [
        f"Profile: {profile.get('name', '')}",
        f"Direction: {direction}",
        f"Proposal: {proposal_name}",
        f"Description: {proposal.get('description', '')}",
        f"Dataset: {proposal.get('dataset', '')}",
        f"Detector: {proposal.get('detector', '')}",
        f"Metrics: {metrics}",
    ]
    if assessment:
        lines.append(f"Assessment: {assessment}")
    if hypothesis_supported is not None:
        lines.append(f"Hypothesis supported: {hypothesis_supported}")
    if lessons:
        lines.append("Lessons:")
        lines.extend(f"- {lesson}" for lesson in lessons[:5])
    return "\n".join(lines)


def _proposal_analysis(proposal_name: str, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    llm_analysis = evaluation_summary.get("llm_analysis") or {}
    per_proposal = llm_analysis.get("per_proposal") or []
    for item in per_proposal:
        if item.get("proposal_name") == proposal_name:
            return item
    return {}


def _proposal_assessment(proposal_name: str, evaluation_summary: dict[str, Any]) -> str:
    return str(_proposal_analysis(proposal_name, evaluation_summary).get("assessment") or "")


def _proposal_hypothesis_supported(proposal_name: str, evaluation_summary: dict[str, Any]) -> bool | None:
    analysis = _proposal_analysis(proposal_name, evaluation_summary)
    if "hypothesis_supported" in analysis:
        return bool(analysis.get("hypothesis_supported"))
    return None


def _proposal_lessons(proposal_name: str, evaluation_summary: dict[str, Any]) -> list[str]:
    analysis = _proposal_analysis(proposal_name, evaluation_summary)
    lessons: list[str] = []
    interpretation = analysis.get("interpretation")
    if interpretation:
        lessons.append(str(interpretation))
    for feature in analysis.get("key_features") or []:
        lessons.append(f"Important feature: {feature}")
    return lessons


def _experiment_entities(
    profile: dict[str, Any],
    proposal_name: str,
    proposal: dict[str, Any],
    experiment_key: str,
) -> list[MemoryEntity]:
    domain = str(profile.get("name") or "")
    entities: list[MemoryEntity] = [
        {
            "entity_type": "proposal",
            "key": proposal_name,
            "name": proposal_name,
            "metadata": {"domain": domain},
        }
    ]
    if experiment_key:
        entities.append({
            "entity_type": "experiment_result",
            "key": experiment_key,
            "name": experiment_key,
            "metadata": {"domain": domain},
        })
    dataset = proposal.get("dataset")
    if dataset:
        entities.append({
            "entity_type": "dataset",
            "key": str(dataset),
            "name": str(dataset),
            "metadata": {"domain": domain},
        })
    detector = proposal.get("detector")
    if detector:
        entities.append({
            "entity_type": "detector",
            "key": str(detector),
            "name": str(detector),
            "metadata": {"domain": domain},
        })
    return entities


def _experiment_relations(
    profile: dict[str, Any],
    proposal_name: str,
    proposal: dict[str, Any],
    experiment_key: str,
    *,
    source_next_step_record_id: str = "",
    source_next_step_title: str = "",
    campaign_id: str = "",
    campaign_title: str = "",
    campaign_variant_id: str = "",
    campaign_variant_title: str = "",
) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
    if campaign_id:
        relations.append({
            "relation_type": "campaign_runs",
            "source_type": "campaign",
            "source_key": campaign_title or campaign_id,
            "target_type": "experiment_result",
            "target_key": experiment_key,
            "metadata": {
                "domain": profile.get("name", ""),
                "campaign_id": campaign_id,
                "campaign_variant_id": campaign_variant_id,
                "campaign_variant_title": campaign_variant_title,
            },
        })
    if experiment_key:
        relations.append({
            "relation_type": "executed_as",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "experiment_result",
            "target_key": experiment_key,
            "metadata": {"domain": profile.get("name", "")},
        })
    if source_next_step_title:
        relations.append({
            "relation_type": "inspires_proposal",
            "source_type": "next_step",
            "source_key": source_next_step_title,
            "target_type": "proposal",
            "target_key": proposal_name,
            "metadata": {
                "domain": profile.get("name", ""),
                "source_next_step_record_id": source_next_step_record_id,
            },
        })
    dataset = proposal.get("dataset")
    if dataset:
        relations.append({
            "relation_type": "tested_on",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "dataset",
            "target_key": str(dataset),
            "metadata": {"domain": profile.get("name", "")},
        })
    detector = proposal.get("detector")
    if detector:
        relations.append({
            "relation_type": "used_detector",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "detector",
            "target_key": str(detector),
            "metadata": {"domain": profile.get("name", "")},
        })
    return relations


def _research_summary_entities(
    domain: str,
    direction: str,
    artifacts: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    digests: list[dict[str, Any]],
) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = []
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    for paper in papers[:10]:
        title = str(paper.get("title") or "")
        if title:
            entities.append({
                "entity_type": "paper",
                "key": str(paper.get("arxiv_id") or title),
                "name": title,
                "metadata": {"domain": domain},
            })
    for digest in digests[:10]:
        title = str(digest.get("title") or "")
        if title:
            entities.append({
                "entity_type": "paper_digest",
                "key": str(digest.get("arxiv_id") or title),
                "name": title,
                "metadata": {"domain": domain},
            })
    for artifact in artifacts[:10]:
        artifact_id = str(artifact.get("artifact_id") or "")
        title = str(artifact.get("title") or artifact_id)
        if artifact_id:
            entities.append({
                "entity_type": "research_artifact",
                "key": artifact_id,
                "name": title,
                "metadata": {"domain": domain},
            })
    return entities


def _research_summary_relations(
    domain: str,
    direction: str,
    artifacts: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    digests: list[dict[str, Any]],
) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
    if not direction:
        return relations
    for paper in papers[:10]:
        paper_key = str(paper.get("arxiv_id") or paper.get("title") or "")
        if paper_key:
            relations.append({
                "relation_type": "summarizes_paper",
                "source_type": "research_direction",
                "source_key": direction,
                "target_type": "paper",
                "target_key": paper_key,
                "metadata": {"domain": domain},
            })
    for digest in digests[:10]:
        digest_key = str(digest.get("arxiv_id") or digest.get("title") or "")
        if digest_key:
            relations.append({
                "relation_type": "has_digest",
                "source_type": "research_direction",
                "source_key": direction,
                "target_type": "paper_digest",
                "target_key": digest_key,
                "metadata": {"domain": domain},
            })
    for artifact in artifacts[:10]:
        artifact_id = str(artifact.get("artifact_id") or "")
        if artifact_id:
            relations.append({
                "relation_type": "uses_context",
                "source_type": "research_direction",
                "source_key": direction,
                "target_type": "research_artifact",
                "target_key": artifact_id,
                "metadata": {"domain": domain},
            })
    return relations


def _research_artifact_entities(
    domain: str,
    direction: str,
    artifact_id: str,
    artifact: dict[str, Any],
) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [{
        "entity_type": "research_artifact",
        "key": artifact_id,
        "name": str(artifact.get("title") or artifact_id),
        "metadata": {"domain": domain},
    }]
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    source = str(artifact.get("source") or "")
    if source:
        entities.append({
            "entity_type": "source",
            "key": source,
            "name": source,
            "metadata": {"domain": domain},
        })
    source_type = str(artifact.get("source_type") or "")
    if source_type:
        entities.append({
            "entity_type": "source_type",
            "key": source_type,
            "name": source_type,
            "metadata": {"domain": domain},
        })
    return entities


def _research_artifact_relations(
    domain: str,
    direction: str,
    artifact_id: str,
    artifact: dict[str, Any],
) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
    if direction:
        relations.append({
            "relation_type": "informs",
            "source_type": "research_artifact",
            "source_key": artifact_id,
            "target_type": "research_direction",
            "target_key": direction,
            "metadata": {"domain": domain},
        })
    source = str(artifact.get("source") or "")
    if source:
        relations.append({
            "relation_type": "originated_from",
            "source_type": "research_artifact",
            "source_key": artifact_id,
            "target_type": "source",
            "target_key": source,
            "metadata": {"domain": domain},
        })
    source_type = str(artifact.get("source_type") or "")
    if source_type:
        relations.append({
            "relation_type": "categorized_as",
            "source_type": "research_artifact",
            "source_key": artifact_id,
            "target_type": "source_type",
            "target_key": source_type,
            "metadata": {"domain": domain},
        })
    return relations


def _idea_entities(domain: str, direction: str, idea_name: str) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [{
        "entity_type": "idea",
        "key": idea_name,
        "name": idea_name,
        "metadata": {"domain": domain},
    }]
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    return entities


def _refined_idea_entities(domain: str, direction: str, idea_name: str) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [{
        "entity_type": "refined_idea",
        "key": idea_name,
        "name": idea_name,
        "metadata": {"domain": domain},
    }]
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    return entities


def _idea_relations(domain: str, direction: str, idea_name: str) -> list[MemoryRelation]:
    if not direction:
        return []
    return [{
        "relation_type": "generated_for",
        "source_type": "idea",
        "source_key": idea_name,
        "target_type": "research_direction",
        "target_key": direction,
        "metadata": {"domain": domain},
    }]


def _refined_idea_relations(domain: str, direction: str, idea_name: str) -> list[MemoryRelation]:
    relations = _idea_relations(domain, direction, idea_name)
    if direction:
        relations.append({
            "relation_type": "refines",
            "source_type": "refined_idea",
            "source_key": idea_name,
            "target_type": "research_direction",
            "target_key": direction,
            "metadata": {"domain": domain},
        })
    return relations


def _proposal_entities(domain: str, direction: str, proposal_name: str, proposal: dict[str, Any]) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [{
        "entity_type": "proposal",
        "key": proposal_name,
        "name": proposal_name,
        "metadata": {"domain": domain},
    }]
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    dataset = str(proposal.get("dataset") or "")
    if dataset:
        entities.append({
            "entity_type": "dataset",
            "key": dataset,
            "name": dataset,
            "metadata": {"domain": domain},
        })
    detector = str(proposal.get("detector") or "")
    if detector:
        entities.append({
            "entity_type": "detector",
            "key": detector,
            "name": detector,
            "metadata": {"domain": domain},
        })
    return entities


def _proposal_relations(
    domain: str,
    direction: str,
    proposal_name: str,
    proposal: dict[str, Any],
    *,
    source_next_step_record_id: str = "",
    source_next_step_title: str = "",
    source_proposal_seed_record_id: str = "",
    source_proposal_seed_title: str = "",
    campaign_id: str = "",
    campaign_title: str = "",
    campaign_variant_id: str = "",
    campaign_variant_title: str = "",
) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
    if campaign_id:
        relations.append({
            "relation_type": "campaign_includes",
            "source_type": "campaign",
            "source_key": campaign_title or campaign_id,
            "target_type": "proposal",
            "target_key": proposal_name,
            "metadata": {
                "domain": domain,
                "campaign_id": campaign_id,
                "campaign_variant_id": campaign_variant_id,
                "campaign_variant_title": campaign_variant_title,
            },
        })
    if source_proposal_seed_title:
        relations.append({
            "relation_type": "seeded_proposal",
            "source_type": "proposal_seed",
            "source_key": source_proposal_seed_title,
            "target_type": "proposal",
            "target_key": proposal_name,
            "metadata": {
                "domain": domain,
                "source_proposal_seed_record_id": source_proposal_seed_record_id,
            },
        })
    if source_next_step_title:
        relations.append({
            "relation_type": "inspires_proposal",
            "source_type": "next_step",
            "source_key": source_next_step_title,
            "target_type": "proposal",
            "target_key": proposal_name,
            "metadata": {
                "domain": domain,
                "source_next_step_record_id": source_next_step_record_id,
            },
        })
    if direction:
        relations.append({
            "relation_type": "proposed_for",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "research_direction",
            "target_key": direction,
            "metadata": {"domain": domain},
        })
    dataset = str(proposal.get("dataset") or "")
    if dataset:
        relations.append({
            "relation_type": "targets_dataset",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "dataset",
            "target_key": dataset,
            "metadata": {"domain": domain},
        })
    detector = str(proposal.get("detector") or "")
    if detector:
        relations.append({
            "relation_type": "uses_detector",
            "source_type": "proposal",
            "source_key": proposal_name,
            "target_type": "detector",
            "target_key": detector,
            "metadata": {"domain": domain},
        })
    return relations


def _implementation_plan_entities(domain: str, proposal_name: str, class_name: str, plan: dict[str, Any]) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [
        {
            "entity_type": "proposal",
            "key": proposal_name,
            "name": proposal_name,
            "metadata": {"domain": domain},
        },
        {
            "entity_type": "implementation_plan",
            "key": class_name,
            "name": class_name,
            "metadata": {"domain": domain},
        },
    ]
    base_class = str(plan.get("base_class") or "")
    if base_class:
        entities.append({
            "entity_type": "base_class",
            "key": base_class,
            "name": base_class,
            "metadata": {"domain": domain},
        })
    return entities


def _implementation_plan_relations(domain: str, proposal_name: str, class_name: str, plan: dict[str, Any]) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = [{
        "relation_type": "plans_implementation",
        "source_type": "proposal",
        "source_key": proposal_name,
        "target_type": "implementation_plan",
        "target_key": class_name,
        "metadata": {"domain": domain},
    }]
    base_class = str(plan.get("base_class") or "")
    if base_class:
        relations.append({
            "relation_type": "extends_base_class",
            "source_type": "implementation_plan",
            "source_key": class_name,
            "target_type": "base_class",
            "target_key": base_class,
            "metadata": {"domain": domain},
        })
    return relations


def _validation_entities(domain: str, proposal_name: str, class_name: str) -> list[MemoryEntity]:
    return [
        {
            "entity_type": "proposal",
            "key": proposal_name,
            "name": proposal_name,
            "metadata": {"domain": domain},
        },
        {
            "entity_type": "implementation",
            "key": class_name,
            "name": class_name,
            "metadata": {"domain": domain},
        },
        {
            "entity_type": "validation",
            "key": class_name,
            "name": class_name,
            "metadata": {"domain": domain},
        },
    ]


def _validation_relations(domain: str, proposal_name: str, class_name: str, validation: dict[str, Any]) -> list[MemoryRelation]:
    return [
        {
            "relation_type": "validates",
            "source_type": "validation",
            "source_key": class_name,
            "target_type": "implementation",
            "target_key": class_name,
            "metadata": {"domain": domain, "passed": validation.get("passed")},
        },
        {
            "relation_type": "belongs_to_proposal",
            "source_type": "validation",
            "source_key": class_name,
            "target_type": "proposal",
            "target_key": proposal_name,
            "metadata": {"domain": domain},
        },
    ]


def _implementation_entities(domain: str, proposal_name: str, class_name: str) -> list[MemoryEntity]:
    return [
        {
            "entity_type": "proposal",
            "key": proposal_name,
            "name": proposal_name,
            "metadata": {"domain": domain},
        },
        {
            "entity_type": "implementation",
            "key": class_name,
            "name": class_name,
            "metadata": {"domain": domain},
        },
    ]


def _implementation_relations(domain: str, proposal_name: str, class_name: str) -> list[MemoryRelation]:
    return [{
        "relation_type": "implements",
        "source_type": "implementation",
        "source_key": class_name,
        "target_type": "proposal",
        "target_key": proposal_name,
        "metadata": {"domain": domain},
    }]


def _artifact_entities(
    domain: str,
    proposal_name: str,
    artifact_id: str,
    artifact_type: str,
    artifact: dict[str, Any],
) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [
        {
            "entity_type": "proposal",
            "key": proposal_name,
            "name": proposal_name,
            "metadata": {"domain": domain},
        },
        {
            "entity_type": artifact_type,
            "key": artifact_id,
            "name": proposal_name,
            "metadata": {"domain": domain},
        },
    ]
    dataset = str(artifact.get("dataset") or "")
    if dataset:
        entities.append({
            "entity_type": "dataset",
            "key": dataset,
            "name": dataset,
            "metadata": {"domain": domain},
        })
    detector = str(artifact.get("detector") or "")
    if detector:
        entities.append({
            "entity_type": "detector",
            "key": detector,
            "name": detector,
            "metadata": {"domain": domain},
        })
    return entities


def _artifact_relations(
    domain: str,
    proposal_name: str,
    artifact_id: str,
    artifact_type: str,
    artifact: dict[str, Any],
) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = [{
        "relation_type": "produced_artifact",
        "source_type": "proposal",
        "source_key": proposal_name,
        "target_type": artifact_type,
        "target_key": artifact_id,
        "metadata": {"domain": domain, "status": artifact.get("status")},
    }]
    dataset = str(artifact.get("dataset") or "")
    if dataset:
        relations.append({
            "relation_type": "materializes_dataset",
            "source_type": artifact_type,
            "source_key": artifact_id,
            "target_type": "dataset",
            "target_key": dataset,
            "metadata": {"domain": domain},
        })
    detector = str(artifact.get("detector") or "")
    if detector:
        relations.append({
            "relation_type": "for_detector",
            "source_type": artifact_type,
            "source_key": artifact_id,
            "target_type": "detector",
            "target_key": detector,
            "metadata": {"domain": domain},
        })
    return relations


def _evaluation_entities(domain: str, direction: str, evaluation: dict[str, Any]) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [{
        "entity_type": "evaluation_summary",
        "key": str(evaluation.get("best_proposal") or "evaluation_summary"),
        "name": str(evaluation.get("best_proposal") or "evaluation_summary"),
        "metadata": {"domain": domain},
    }]
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    best_proposal = str(evaluation.get("best_proposal") or "")
    if best_proposal:
        entities.append({
            "entity_type": "proposal",
            "key": best_proposal,
            "name": best_proposal,
            "metadata": {"domain": domain},
        })
    return entities


def _evaluation_relations(domain: str, direction: str, evaluation: dict[str, Any]) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
    eval_key = str(evaluation.get("best_proposal") or "evaluation_summary")
    if direction:
        relations.append({
            "relation_type": "evaluates_direction",
            "source_type": "evaluation_summary",
            "source_key": eval_key,
            "target_type": "research_direction",
            "target_key": direction,
            "metadata": {"domain": domain},
        })
    best_proposal = str(evaluation.get("best_proposal") or "")
    if best_proposal:
        relations.append({
            "relation_type": "selects_best_proposal",
            "source_type": "evaluation_summary",
            "source_key": eval_key,
            "target_type": "proposal",
            "target_key": best_proposal,
            "metadata": {"domain": domain, "best_metric_value": evaluation.get("best_metric_value")},
        })
    return relations


def _next_step_entities(domain: str, direction: str, title: str, step: dict[str, Any]) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = [{
        "entity_type": "next_step",
        "key": title,
        "name": title,
        "metadata": {"domain": domain},
    }]
    if direction:
        entities.append({
            "entity_type": "research_direction",
            "key": direction,
            "name": direction,
            "metadata": {"domain": domain},
        })
    suggested = str(step.get("suggested_direction") or "")
    if suggested:
        entities.append({
            "entity_type": "suggested_direction",
            "key": suggested,
            "name": suggested,
            "metadata": {"domain": domain},
        })
    return entities


def _next_step_relations(domain: str, direction: str, title: str, step: dict[str, Any]) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
    if direction:
        relations.append({
            "relation_type": "follows_from",
            "source_type": "next_step",
            "source_key": title,
            "target_type": "research_direction",
            "target_key": direction,
            "metadata": {"domain": domain, "priority": step.get("priority")},
        })
    suggested = str(step.get("suggested_direction") or "")
    if suggested:
        relations.append({
            "relation_type": "suggests_direction",
            "source_type": "next_step",
            "source_key": title,
            "target_type": "suggested_direction",
            "target_key": suggested,
            "metadata": {"domain": domain},
        })
    return relations


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_dotted(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _compact_payload_summary(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"{key}: {value}")
        if len(lines) >= 12:
            break
    return "\n".join(lines)


def _blob_refs_from_metadata_targets(
    payload: dict[str, Any],
    targets: tuple[tuple[str, str, str, str | None], ...],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for name, uri_key, artifact_id_key, content_type in targets:
        uri = payload.get(uri_key)
        artifact_id = payload.get(artifact_id_key)
        if not isinstance(uri, str) or not uri:
            continue
        ref: dict[str, Any] = {
            "blob_id": str(artifact_id or uri),
            "name": name,
            "uri": uri,
            "artifact_id": str(artifact_id or ""),
            "metadata": {},
        }
        if content_type:
            ref["content_type"] = content_type
        refs.append(ref)
    return refs


def _namespaced_graph_key(domain: str, entity_type: Any, key: Any) -> str:
    domain_text = str(domain or "").strip()
    entity_type_text = str(entity_type or "").strip()
    key_text = str(key or "").strip()
    if not key_text:
        return ""
    if domain_text and entity_type_text:
        return f"{domain_text}:{entity_type_text}:{key_text}"
    if domain_text:
        return f"{domain_text}:{key_text}"
    return key_text


def _prefixed_graph_name(domain: str, name: Any) -> str:
    name_text = str(name or "").strip()
    domain_text = str(domain or "").strip()
    if not name_text:
        return ""
    if not domain_text:
        return name_text
    if name_text.startswith(f"{domain_text}:"):
        return name_text
    return f"{domain_text}:{name_text}"
