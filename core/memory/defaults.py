"""Default memory record builders and retrieval mappers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.memory.fingerprints import fingerprint_json
from core.memory.models import MemoryEntity, MemoryRecord, MemoryRelation


def build_core_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic canonical memory records from the current graph state."""
    records: list[MemoryRecord] = []
    records.extend(build_refined_idea_memory_records(profile, state))
    records.extend(build_implementation_memory_records(profile, state))
    records.extend(build_validation_memory_records(profile, state))
    records.extend(build_experiment_artifact_memory_records(profile, state))
    records.extend(build_experiment_memory_records(profile, state))
    return dedupe_memory_records(records)


def build_experiment_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build canonical memory records from generic experiment results."""
    records: list[MemoryRecord] = []
    experiment_results = state.get("experiment_results") or []
    models = state.get("models") or []
    evaluation_summary = state.get("evaluation_summary") or {}
    direction = state.get("research_direction", "")
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
                "inserted_at": inserted_at,
                "mlflow_run_id": result.get("mlflow_run_id") or model.get("mlflow_run_id") or "",
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
            },
            "tags": [str(profile.get("name") or ""), "prior_experiment"],
            "created_at": inserted_at,
            "source_run_id": str(result.get("mlflow_run_id") or model.get("mlflow_run_id") or "") or None,
            "entities": _experiment_entities(profile, proposal_name, proposal),
            "relations": _experiment_relations(profile, proposal_name, proposal),
        }
        records.append(record)
    return records


def build_refined_idea_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for refined ideas."""
    ideas = state.get("refined_ideas") or []
    direction = state.get("research_direction", "")
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for idea in ideas:
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
        })
    return records


def build_validation_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for validation outcomes."""
    validations = state.get("validation_results") or []
    implementations = state.get("implementations") or []
    impl_by_class = {
        impl.get("class_name"): impl
        for impl in implementations
        if impl.get("class_name")
    }
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for validation in validations:
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
        })
    return records


def build_implementation_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for generated implementations."""
    implementations = state.get("implementations") or []
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for implementation in implementations:
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
        })
    return records


def build_experiment_artifact_memory_records(profile: dict[str, Any], state: dict[str, Any]) -> list[MemoryRecord]:
    """Build generic memory records for prepared experiment artifacts."""
    artifacts = state.get("experiment_artifacts") or state.get("datasets") or []
    domain = str(profile.get("name") or "")
    records: list[MemoryRecord] = []

    for artifact in artifacts:
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
        })
    return records


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
    return {
        "embedding_text": _embedding_text(record),
        "vector_metadata": {
            "record_id": record.get("record_id", ""),
            "domain": record.get("domain", ""),
            "memory_kind": record.get("kind", ""),
            "object_type": record.get("object_type", ""),
            "object_key": record.get("object_key", ""),
            "object_role": record.get("object_role", ""),
            "schema_version": record.get("schema_version", "1"),
            "title": record.get("title", ""),
            "tags": list(record.get("tags") or []),
            **_scalar_metadata(metadata),
            "memory_summary": record.get("summary", ""),
        },
        "graph_nodes": [
            {
                "node_type": entity.get("entity_type", ""),
                "node_key": entity.get("key", ""),
                "name": entity.get("name", ""),
                "metadata": entity.get("metadata") or {},
            }
            for entity in (record.get("entities") or [])
        ],
        "graph_edges": [
            {
                "edge_type": relation.get("relation_type", ""),
                "source_type": relation.get("source_type", ""),
                "source_key": relation.get("source_key", ""),
                "target_type": relation.get("target_type", ""),
                "target_key": relation.get("target_key", ""),
                "metadata": relation.get("metadata") or {},
            }
            for relation in (record.get("relations") or [])
        ],
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
            compact[key] = value[:20]
    return compact


def dedupe_memory_records(records: list[MemoryRecord]) -> list[MemoryRecord]:
    deduped: dict[str, MemoryRecord] = {}
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            continue
        deduped[record_id] = record
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


def _experiment_entities(profile: dict[str, Any], proposal_name: str, proposal: dict[str, Any]) -> list[MemoryEntity]:
    domain = str(profile.get("name") or "")
    entities: list[MemoryEntity] = [
        {
            "entity_type": "proposal",
            "key": proposal_name,
            "name": proposal_name,
            "metadata": {"domain": domain},
        }
    ]
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


def _experiment_relations(profile: dict[str, Any], proposal_name: str, proposal: dict[str, Any]) -> list[MemoryRelation]:
    relations: list[MemoryRelation] = []
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
