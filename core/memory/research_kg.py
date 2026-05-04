"""Distilled research knowledge-graph extraction and canonicalization."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm.factory import get_llm
from core.memory.fingerprints import fingerprint_json
from core.memory.models import CanonicalizationResult, MemoryRecord, ResearchKGNode, ResearchKGRelation, ResearchKGUpdate
from core.plugins.loader import adapter_has, load_adapter
from core.utils import extract_json_object
from core.utils.logger import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

_DEFAULT_RELATIONS = {
    "question_hypothesis": "HAS_HYPOTHESIS",
    "hypothesis_tested_by": "TESTED_BY",
    "hypothesis_supported_by": "SUPPORTED_BY",
    "hypothesis_contradicted_by": "CONTRADICTED_BY",
    "hypothesis_refined_by": "REFINED_BY",
    "evidence_method": "USES_METHOD",
    "evidence_dataset": "ON_DATASET",
    "evidence_metric": "MEASURED_BY",
    "evidence_finding": "PRODUCED_FINDING",
    "finding_hypothesis": "SUPPORTS",
    "finding_suggests": "SUGGESTS",
    "limitation": "HAS_LIMITATION",
    "performance_band": "IN_PERFORMANCE_BAND",
    "possible_match": "POSSIBLY_SAME_AS",
    "variant_of": "VARIANT_OF",
    "method_family": "BELONGS_TO_FAMILY",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if", "in", "into", "is", "it",
    "its", "may", "of", "on", "or", "should", "show", "that", "the", "their", "then", "this", "to", "using",
    "whether", "with", "will",
}


@dataclass
class GraphCanonicalCandidate:
    canonical_id: str
    display_name: str
    aliases: list[str]
    properties: dict[str, Any]


class GraphCanonicalLookup:
    """Lookup interface used during canonicalization."""

    def list_candidates(self, *, domain: str, node_type: str, limit: int = 25) -> list[GraphCanonicalCandidate]:
        return []


def build_research_kg_update(
    record: MemoryRecord,
    *,
    profile: dict[str, Any] | None,
    canonical_lookup: GraphCanonicalLookup | None = None,
) -> ResearchKGUpdate:
    """Build one distilled KG update from a canonical memory record."""
    profile = profile or {}
    kg_cfg = knowledge_graph_config(profile)
    if not kg_cfg.get("enabled", True):
        return {"record_id": str(record.get("record_id") or ""), "domain": str(record.get("domain") or ""), "nodes": [], "relations": []}

    nodes: list[ResearchKGNode] = []
    relations: list[ResearchKGRelation] = []
    builder = _record_builder(record)
    if builder is None:
        return {"record_id": str(record.get("record_id") or ""), "domain": str(record.get("domain") or ""), "nodes": [], "relations": []}

    relation_names = dict(_DEFAULT_RELATIONS)
    relation_names.update(kg_cfg.get("relations") or {})
    builder(record, profile, kg_cfg, relation_names, canonical_lookup or GraphCanonicalLookup(), nodes, relations)
    return {
        "record_id": str(record.get("record_id") or ""),
        "domain": str(record.get("domain") or ""),
        "nodes": _dedupe_nodes(nodes),
        "relations": _dedupe_relations(relations),
        "provenance": {
            "record_id": str(record.get("record_id") or ""),
            "object_type": str(record.get("object_type") or ""),
            "kind": str(record.get("kind") or ""),
        },
    }


def knowledge_graph_config(profile: dict[str, Any]) -> dict[str, Any]:
    """Return merged KG configuration for a profile."""
    cfg = dict(profile.get("knowledge_graph") or {})
    memory_cfg = profile.get("memory") or {}
    cfg.setdefault("enabled", True)
    cfg.setdefault("max_candidate_matches", 8)
    cfg.setdefault("llm_canonicalization_step", "knowledge_graph_canonicalize")
    cfg.setdefault("relations", {})
    cfg.setdefault("methods", {})
    cfg.setdefault("hypotheses", {})
    cfg.setdefault("limitations", {})
    cfg.setdefault("datasets", {})
    cfg.setdefault("metrics", {})
    cfg.setdefault("metric_bands", _metric_bands_from_profile(profile))
    cfg.setdefault("canonicalization", {})
    canonicalization = dict(cfg.get("canonicalization") or {})
    canonicalization.setdefault("llm_enabled", bool(profile.get("llm")))
    canonicalization.setdefault("min_llm_confidence", 0.84)
    canonicalization.setdefault("max_candidates", int(cfg.get("max_candidate_matches", 8)))
    canonicalization.setdefault("ignored_tokens", ["approach", "method", "strategy", "technique"])
    cfg["canonicalization"] = canonicalization
    cfg.setdefault("memory_object_specs", memory_cfg.get("objects") or [])
    return cfg


def load_plugin_kg_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    """Allow adapters to extend KG config without hardcoding domain logic."""
    try:
        adapter = load_adapter(profile)
    except Exception:
        return {}
    if adapter is not None and adapter_has(adapter, "knowledge_graph_config"):
        try:
            return dict(adapter.knowledge_graph_config(profile) or {})
        except Exception as exc:
            log.warning("research_kg | Adapter knowledge_graph_config failed: %s", exc)
    return {}


def canonicalize_concept(
    *,
    node_type: str,
    raw_name: str,
    properties: dict[str, Any],
    domain: str,
    profile: dict[str, Any],
    canonical_lookup: GraphCanonicalLookup,
) -> CanonicalizationResult:
    """Resolve one concept to a canonical graph identity."""
    kg_cfg = knowledge_graph_config(profile)
    overrides = load_plugin_kg_overrides(profile)
    if overrides:
        merged = dict(kg_cfg)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        kg_cfg = merged
    ignored_tokens = {str(item).strip().lower() for item in (kg_cfg.get("canonicalization", {}).get("ignored_tokens") or [])}
    aliases = _aliases_for(raw_name, properties, node_type, kg_cfg)
    normalized = {
        "domain": domain,
        "node_type": node_type,
        "name": _normalize_text(raw_name, ignored_tokens=ignored_tokens),
        "scope": _normalized_scope(node_type, properties, kg_cfg),
    }
    deterministic_id = _canonical_id(node_type, domain, normalized)
    display_properties = {"_node_type": node_type, **properties}
    result: CanonicalizationResult = {
        "canonical_id": deterministic_id,
        "display_name": _display_name(raw_name, display_properties),
        "aliases": aliases,
        "normalized_fields": normalized,
        "strategy": "deterministic",
        "confidence": 0.92,
        "matched_existing_id": None,
        "rationale": "Deterministic normalized identity.",
    }

    candidates = _match_candidates(
        node_type=node_type,
        aliases=aliases,
        normalized=normalized,
        domain=domain,
        canonical_lookup=canonical_lookup,
        max_candidates=int(kg_cfg.get("canonicalization", {}).get("max_candidates", 8)),
    )
    exact = next((candidate for candidate in candidates if candidate.canonical_id == deterministic_id), None)
    if exact is not None:
        result["matched_existing_id"] = exact.canonical_id
        result["rationale"] = "Matched deterministic canonical id in existing graph."
        return result

    llm_enabled = bool(kg_cfg.get("canonicalization", {}).get("llm_enabled"))
    if not llm_enabled or not candidates or not _should_try_llm(node_type=node_type, candidates=candidates, aliases=aliases):
        return result

    llm_result = _llm_canonicalize(
        node_type=node_type,
        raw_name=raw_name,
        properties=properties,
        normalized=normalized,
        aliases=aliases,
        candidates=candidates,
        profile=profile,
        step_name=str(kg_cfg.get("llm_canonicalization_step") or "knowledge_graph_canonicalize"),
    )
    if not llm_result:
        return result

    if llm_result.get("action") == "match":
        match_id = str(llm_result.get("match_id") or "")
        if match_id and any(candidate.canonical_id == match_id for candidate in candidates):
            confidence = float(llm_result.get("confidence") or 0.0)
            min_conf = float(kg_cfg.get("canonicalization", {}).get("min_llm_confidence", 0.84))
            if confidence >= min_conf:
                matched = next(candidate for candidate in candidates if candidate.canonical_id == match_id)
                return {
                    "canonical_id": matched.canonical_id,
                    "display_name": matched.display_name,
                    "aliases": sorted(set(matched.aliases + aliases)),
                    "normalized_fields": normalized,
                    "strategy": "llm_match",
                    "confidence": confidence,
                    "matched_existing_id": matched.canonical_id,
                    "rationale": str(llm_result.get("rationale") or "LLM matched an existing canonical concept."),
                }
            result["rationale"] = f"LLM proposed match below threshold ({confidence:.2f})."
    elif llm_result.get("action") == "new":
        proposed_name = str(llm_result.get("display_name") or "").strip()
        if proposed_name:
            result["display_name"] = proposed_name
            result["aliases"] = sorted(set(aliases + [proposed_name]))
            result["rationale"] = str(llm_result.get("rationale") or "LLM proposed a new canonical concept.")
            result["strategy"] = "llm_new"
            result["confidence"] = float(llm_result.get("confidence") or 0.75)
    return result


def _record_builder(record: MemoryRecord):
    object_type = str(record.get("object_type") or "")
    if object_type in {"experiment_result"}:
        return _build_from_experiment_result
    if object_type in {"research_summary"}:
        return _build_from_research_summary
    if object_type in {"idea", "refined_idea"}:
        return _build_from_hypothesis_idea
    return None


def _build_from_research_summary(
    record: MemoryRecord,
    profile: dict[str, Any],
    kg_cfg: dict[str, Any],
    relations_cfg: dict[str, str],
    canonical_lookup: GraphCanonicalLookup,
    nodes: list[ResearchKGNode],
    relations: list[ResearchKGRelation],
) -> None:
    direction = str((record.get("content") or {}).get("research_direction") or record.get("title") or "").strip()
    if not direction:
        return
    canonical = canonicalize_concept(
        node_type="Question",
        raw_name=direction,
        properties={"research_direction": direction},
        domain=str(record.get("domain") or ""),
        profile=profile,
        canonical_lookup=canonical_lookup,
    )
    nodes.append(_kg_node("Question", canonical, {"research_direction": direction}))


def _build_from_hypothesis_idea(
    record: MemoryRecord,
    profile: dict[str, Any],
    kg_cfg: dict[str, Any],
    relations_cfg: dict[str, str],
    canonical_lookup: GraphCanonicalLookup,
    nodes: list[ResearchKGNode],
    relations: list[ResearchKGRelation],
) -> None:
    content = dict(record.get("content") or {})
    direction = str(record.get("metadata", {}).get("research_direction") or "")
    question_node = _question_node(direction, record, profile, canonical_lookup)
    if question_node:
        nodes.append(question_node)
    hypothesis_text = str(content.get("hypothesis") or content.get("description") or record.get("summary") or "").strip()
    if not hypothesis_text:
        return
    hypothesis_props = {
        "hypothesis_text": hypothesis_text,
        "idea_name": str(content.get("name") or content.get("title") or record.get("title") or ""),
        "status": "untested",
    }
    hypothesis = canonicalize_concept(
        node_type="Hypothesis",
        raw_name=hypothesis_text,
        properties=hypothesis_props,
        domain=str(record.get("domain") or ""),
        profile=profile,
        canonical_lookup=canonical_lookup,
    )
    nodes.append(_kg_node("Hypothesis", hypothesis, hypothesis_props))
    if question_node:
        relations.append(_kg_relation(relations_cfg["question_hypothesis"], question_node["canonical_id"], hypothesis["canonical_id"]))


def _build_from_experiment_result(
    record: MemoryRecord,
    profile: dict[str, Any],
    kg_cfg: dict[str, Any],
    relations_cfg: dict[str, str],
    canonical_lookup: GraphCanonicalLookup,
    nodes: list[ResearchKGNode],
    relations: list[ResearchKGRelation],
) -> None:
    content = dict(record.get("content") or {})
    metadata = dict(record.get("metadata") or {})
    metrics = dict(content.get("metrics") or {})
    proposal = dict(content.get("proposal") or {})
    direction = str(content.get("research_direction") or metadata.get("research_direction") or "")
    domain = str(record.get("domain") or "")

    question_node = _question_node(direction, record, profile, canonical_lookup)
    if question_node:
        nodes.append(question_node)

    method_name = str(metadata.get("feature_set_class_name") or content.get("proposal_name") or record.get("title") or "").strip()
    method_node = None
    method_family_node = None
    if method_name:
        method_props = {
            "family": str(proposal.get("method_family") or proposal.get("detector") or ""),
            "proposal_name": str(content.get("proposal_name") or record.get("title") or ""),
            "implementation_class": str(metadata.get("feature_set_class_name") or ""),
        }
        method_canonical = canonicalize_concept(
            node_type="Method",
            raw_name=method_name,
            properties=method_props,
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        method_node = _kg_node("Method", method_canonical, method_props)
        nodes.append(method_node)
        family_name = _method_family_name(method_name, method_props)
        family_props = {"family_name": family_name, "source_method": method_name}
        family_canonical = canonicalize_concept(
            node_type="MethodFamily",
            raw_name=family_name,
            properties=family_props,
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        method_family_node = _kg_node("MethodFamily", family_canonical, family_props)
        nodes.append(method_family_node)

    dataset_name = str(metadata.get("dataset") or proposal.get("dataset") or "").strip()
    dataset_node = None
    if dataset_name:
        dataset_canonical = canonicalize_concept(
            node_type="Dataset",
            raw_name=dataset_name,
            properties={"dataset": dataset_name},
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        dataset_node = _kg_node("Dataset", dataset_canonical, {"dataset": dataset_name})
        nodes.append(dataset_node)

    hypothesis_text = str(metadata.get("hypothesis") or proposal.get("hypothesis") or proposal.get("description") or content.get("proposal_name") or "").strip()
    hypothesis_node = None
    if hypothesis_text:
        hypothesis_props = {
            "hypothesis_text": hypothesis_text,
            "proposal_name": str(content.get("proposal_name") or ""),
            "dataset": dataset_name,
            "metric": str(_primary_metric_name(profile, metrics, metadata)),
        }
        hypothesis_canonical = canonicalize_concept(
            node_type="Hypothesis",
            raw_name=hypothesis_text,
            properties=hypothesis_props,
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        hypothesis_node = _kg_node("Hypothesis", hypothesis_canonical, hypothesis_props | {"status": _hypothesis_status(metadata, metrics, profile)})
        nodes.append(hypothesis_node)
        if question_node:
            relations.append(_kg_relation(relations_cfg["question_hypothesis"], question_node["canonical_id"], hypothesis_node["canonical_id"]))

    evidence_summary = _evidence_summary(record, metrics)
    primary_metric_name = _primary_metric_name(profile, metrics, metadata)
    primary_metric_value = _primary_metric_value(profile, metrics, metadata)
    assessment = str(metadata.get("assessment") or "")
    evidence_props = {
        "summary": evidence_summary,
        "proposal_name": str(content.get("proposal_name") or record.get("title") or ""),
        "dataset": dataset_name,
        "method": method_name,
        "assessment": assessment,
        "metric_name": primary_metric_name,
        "metric_value": primary_metric_value,
        "metrics": metrics,
        "confidence": _evidence_confidence(metadata, metrics, profile),
        "direction": "positive" if primary_metric_value >= _primary_threshold(profile, metadata) else "mixed",
        "n_experiments": int(metadata.get("n_experiments") or 1),
        "provenance_record_ids": sorted(set([str(record.get("record_id") or "")] + [str(item) for item in ((record.get("lineage") or {}).get("source_record_ids") or []) if item])),
        "source_run_id": str(record.get("source_run_id") or metadata.get("mlflow_run_id") or ""),
    }
    evidence_canonical = canonicalize_concept(
        node_type="Evidence",
        raw_name=evidence_summary,
        properties=evidence_props,
        domain=domain,
        profile=profile,
        canonical_lookup=canonical_lookup,
    )
    evidence_node = _kg_node("Evidence", evidence_canonical, evidence_props)
    nodes.append(evidence_node)

    if hypothesis_node:
        relations.append(_kg_relation(relations_cfg["hypothesis_tested_by"], hypothesis_node["canonical_id"], evidence_node["canonical_id"]))
        if _supports_hypothesis(metadata, metrics, profile):
            relations.append(_kg_relation(relations_cfg["hypothesis_supported_by"], hypothesis_node["canonical_id"], evidence_node["canonical_id"]))
        else:
            relations.append(_kg_relation(relations_cfg["hypothesis_contradicted_by"], hypothesis_node["canonical_id"], evidence_node["canonical_id"]))
    if method_node:
        relations.append(_kg_relation(relations_cfg["evidence_method"], evidence_node["canonical_id"], method_node["canonical_id"]))
    if method_node and method_family_node:
        relations.append(_kg_relation(relations_cfg["method_family"], method_node["canonical_id"], method_family_node["canonical_id"]))
    if dataset_node:
        relations.append(_kg_relation(relations_cfg["evidence_dataset"], evidence_node["canonical_id"], dataset_node["canonical_id"]))

    metric_name = primary_metric_name
    metric_node = None
    if metric_name:
        metric_props = {"metric_name": metric_name, "direction": _metric_direction(metric_name, kg_cfg)}
        metric_canonical = canonicalize_concept(
            node_type="Metric",
            raw_name=metric_name,
            properties=metric_props,
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        metric_node = _kg_node("Metric", metric_canonical, metric_props)
        nodes.append(metric_node)
        relations.append(_kg_relation(relations_cfg["evidence_metric"], evidence_node["canonical_id"], metric_node["canonical_id"]))

    finding_text = _finding_text(record, metrics)
    if finding_text:
        finding_props = {
            "summary": finding_text,
            "proposal_name": str(content.get("proposal_name") or record.get("title") or ""),
            "dataset": dataset_name,
            "assessment": assessment,
            "metric_name": metric_name,
            "metric_value": primary_metric_value,
        }
        finding_canonical = canonicalize_concept(
            node_type="Finding",
            raw_name=finding_text,
            properties=finding_props,
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        finding_node = _kg_node("Finding", finding_canonical, finding_props)
        nodes.append(finding_node)
        relations.append(_kg_relation(relations_cfg["evidence_finding"], evidence_node["canonical_id"], finding_node["canonical_id"]))
        if hypothesis_node:
            relations.append(_kg_relation(relations_cfg["finding_hypothesis"], finding_node["canonical_id"], hypothesis_node["canonical_id"]))

    limitation_text = str(metadata.get("limitations") or metadata.get("risk") or "").strip()
    if limitation_text:
        limitation_props = {"limitation_text": limitation_text}
        limitation_canonical = canonicalize_concept(
            node_type="Limitation",
            raw_name=limitation_text,
            properties=limitation_props,
            domain=domain,
            profile=profile,
            canonical_lookup=canonical_lookup,
        )
        limitation_node = _kg_node("Limitation", limitation_canonical, limitation_props)
        nodes.append(limitation_node)
        relations.append(_kg_relation(relations_cfg["limitation"], evidence_node["canonical_id"], limitation_node["canonical_id"]))

    for band in _metric_bands_for_value(metric_name, _primary_metric_value(profile, metrics, metadata), kg_cfg):
        band_props = {
            "metric_name": metric_name,
            "operator": str(band.get("operator") or ""),
            "threshold": band.get("threshold"),
            "band_key": str(band.get("band_key") or ""),
        }
        band_id = _canonical_id("PerformanceBand", domain, {"metric_name": metric_name, "operator": band_props["operator"], "threshold": band_props["threshold"]})
        band_node: ResearchKGNode = {
            "node_type": "PerformanceBand",
            "canonical_id": band_id,
            "display_name": str(band.get("display_name") or band_props["band_key"] or band_id),
            "aliases": [str(band.get("display_name") or band_props["band_key"] or band_id)],
            "properties": band_props,
        }
        nodes.append(band_node)
        relations.append(_kg_relation(relations_cfg["performance_band"], evidence_node["canonical_id"], band_id))


def _question_node(direction: str, record: MemoryRecord, profile: dict[str, Any], canonical_lookup: GraphCanonicalLookup) -> ResearchKGNode | None:
    direction = str(direction or "").strip()
    if not direction:
        return None
    canonical = canonicalize_concept(
        node_type="Question",
        raw_name=direction,
        properties={"research_direction": direction},
        domain=str(record.get("domain") or ""),
        profile=profile,
        canonical_lookup=canonical_lookup,
    )
    return _kg_node("Question", canonical, {"research_direction": direction})


def _kg_node(node_type: str, canonical: CanonicalizationResult, properties: dict[str, Any]) -> ResearchKGNode:
    merged = dict(properties)
    merged.setdefault("normalized_fields", dict(canonical.get("normalized_fields") or {}))
    merged.setdefault("canonicalization_strategy", str(canonical.get("strategy") or "deterministic"))
    merged.setdefault("canonicalization_confidence", float(canonical.get("confidence") or 0.0))
    return {
        "node_type": node_type,
        "canonical_id": str(canonical.get("canonical_id") or ""),
        "display_name": str(canonical.get("display_name") or canonical.get("canonical_id") or ""),
        "aliases": sorted(set(str(item) for item in (canonical.get("aliases") or []) if str(item).strip())),
        "properties": merged,
    }


def _kg_relation(relation_type: str, source_id: str, target_id: str, properties: dict[str, Any] | None = None) -> ResearchKGRelation:
    return {
        "relation_type": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "properties": dict(properties or {}),
    }


def _match_candidates(
    *,
    node_type: str,
    aliases: list[str],
    normalized: dict[str, Any],
    domain: str,
    canonical_lookup: GraphCanonicalLookup,
    max_candidates: int,
) -> list[GraphCanonicalCandidate]:
    candidates = canonical_lookup.list_candidates(domain=domain, node_type=node_type, limit=max(10, max_candidates * 3))
    if not candidates:
        return []
    alias_tokens = _token_set(" ".join(aliases))
    scored: list[tuple[float, GraphCanonicalCandidate]] = []
    for candidate in candidates:
        candidate_tokens = _token_set(" ".join([candidate.display_name] + candidate.aliases))
        overlap = len(alias_tokens & candidate_tokens)
        union = len(alias_tokens | candidate_tokens) or 1
        score = overlap / union
        if candidate.properties.get("normalized_fields") == normalized:
            score = 1.0
        if score > 0:
            scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in scored[:max_candidates]]


def _llm_canonicalize(
    *,
    node_type: str,
    raw_name: str,
    properties: dict[str, Any],
    normalized: dict[str, Any],
    aliases: list[str],
    candidates: list[GraphCanonicalCandidate],
    profile: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    try:
        llm = get_llm(step_name=step_name, profile=profile)
    except Exception as exc:
        log.debug("research_kg | No LLM available for canonicalization: %s", exc)
        return {}
    system = (
        "You are canonicalizing research knowledge graph concepts. "
        "Choose an existing concept only when it is truly the same concept in research terms. "
        "When unsure, create a new concept. Respond with JSON only using keys: "
        "action ('match' or 'new'), match_id, display_name, confidence, rationale."
    )
    payload = {
        "node_type": node_type,
        "raw_name": raw_name,
        "properties": properties,
        "normalized": normalized,
        "aliases": aliases,
        "candidates": [
            {
                "canonical_id": candidate.canonical_id,
                "display_name": candidate.display_name,
                "aliases": candidate.aliases,
                "properties": candidate.properties,
            }
            for candidate in candidates
        ],
    }
    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(payload, indent=2, default=str)),
        ])
        parsed = extract_json_object(response.content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        log.warning("research_kg | LLM canonicalization failed for %s %r: %s", node_type, raw_name, exc)
        return {}


def _aliases_for(raw_name: str, properties: dict[str, Any], node_type: str, kg_cfg: dict[str, Any]) -> list[str]:
    aliases = {str(raw_name or "").strip()}
    aliases.update(str(item).strip() for item in (properties.get("aliases") or []) if str(item).strip())
    alias_map = dict((kg_cfg.get(node_type.lower() + "s") or {}).get("aliases") or {})
    for source, mapped in alias_map.items():
        if _normalize_text(str(source)) == _normalize_text(raw_name):
            aliases.add(str(mapped))
    if properties.get("summary"):
        aliases.add(str(properties.get("summary")))
    return sorted(item for item in aliases if item)


def _normalized_scope(node_type: str, properties: dict[str, Any], kg_cfg: dict[str, Any]) -> dict[str, Any]:
    if node_type == "Question":
        return {
            "topic": _normalize_text(_question_label(str(properties.get("research_direction") or ""))),
        }
    if node_type == "MethodFamily":
        return {
            "family_name": _normalize_text(str(properties.get("family_name") or "")),
        }
    if node_type == "Method":
        return {
            "family": _normalize_text(str(properties.get("family_name") or properties.get("family") or properties.get("proposal_name") or "")),
            "implementation_class": _normalize_text(str(properties.get("implementation_class") or "")),
        }
    if node_type == "Hypothesis":
        short_hypothesis = _hypothesis_label(str(properties.get("hypothesis_text") or ""))
        return {
            "claim": _normalize_text(short_hypothesis),
            "dataset": _normalize_text(str(properties.get("dataset") or "")),
            "metric": _normalize_text(str(properties.get("metric") or "")),
        }
    if node_type == "Evidence":
        proposal_name = _normalize_text(str(properties.get("proposal_name") or ""))
        return {
            "proposal_name": proposal_name,
            "dataset": _normalize_text(str(properties.get("dataset") or "")),
            "method": _normalize_text(str(properties.get("method") or "")),
        }
    if node_type == "Finding":
        return {
            "proposal_name": _normalize_text(str(properties.get("proposal_name") or "")),
            "assessment": _normalize_text(str(properties.get("assessment") or "")),
            "dataset": _normalize_text(str(properties.get("dataset") or "")),
        }
    if node_type == "Metric":
        return {
            "direction": _normalize_text(str(properties.get("direction") or "")),
        }
    if node_type == "Dataset":
        return dict((kg_cfg.get("datasets") or {}).get("properties") or {})
    return {}


def _canonical_id(node_type: str, domain: str, normalized_fields: dict[str, Any]) -> str:
    return f"{domain}:{node_type.lower()}:{fingerprint_json(normalized_fields)}"


def _display_name(raw_name: str, properties: dict[str, Any]) -> str:
    concise = _concise_display_name(str(properties.get("_node_type") or ""), raw_name, properties)
    if concise:
        return concise
    if raw_name.strip():
        return raw_name.strip()
    for key in ("summary", "proposal_name", "hypothesis_text", "metric_name", "dataset"):
        value = str(properties.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _normalize_text(value: str, *, ignored_tokens: set[str] | None = None) -> str:
    tokens = _WORD_RE.findall(str(value or "").lower())
    if ignored_tokens:
        tokens = [token for token in tokens if token not in ignored_tokens]
    return "_".join(tokens[:18])


def _token_set(value: str) -> set[str]:
    return set(_WORD_RE.findall(str(value or "").lower()))


def _humanize_identifier(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(part.capitalize() if part.islower() else part for part in text.split())


def _short_label(value: str, *, max_tokens: int = 6, title_case: bool = True) -> str:
    tokens = [token for token in _WORD_RE.findall(str(value or "").lower()) if token not in _STOPWORDS]
    if not tokens:
        tokens = _WORD_RE.findall(str(value or "").lower())
    clipped = tokens[:max_tokens]
    if title_case:
        return " ".join(token.capitalize() for token in clipped)
    return " ".join(clipped)


def _hypothesis_label(text: str) -> str:
    tokens = [token for token in _WORD_RE.findall(str(text or "").lower()) if token not in _STOPWORDS]
    if not tokens:
        return "Hypothesis"
    subject = _first_matching(
        tokens,
        [
            ("attention_entropy", {"attention", "entropy"}),
            ("cross_position_cosine", {"cross", "position", "cosine"}),
            ("residual_norm_trajectory", {"residual", "norm", "trajectory"}),
            ("residual_pca_separation", {"residual", "pca"}),
            ("projection_dynamics", {"projection"}),
            ("activation_entropy", {"activation", "entropy"}),
            ("claim_attention", {"claim", "attention"}),
            ("entropy_features", {"entropy"}),
            ("attention_features", {"attention"}),
            ("residual_dynamics", {"residual"}),
        ],
        default="model_features",
    )
    effect = _first_matching(
        tokens,
        [
            ("predicts", {"predict", "predicts", "correlate", "correlates", "signature", "indicates"}),
            ("improves", {"improve", "improves", "exceeds", "outperform"}),
            ("grounds", {"ground", "grounds", "grounded"}),
            ("decouples", {"decouple", "decouples", "disconnect", "collapses"}),
        ],
        default="predicts",
    )
    target = _first_matching(
        tokens,
        [
            ("hallucination", {"hallucination", "hallucinations"}),
            ("yes_no_decision", {"yes", "no", "decision"}),
            ("generalization", {"generalization", "domain", "cluster"}),
            ("claim_grounding", {"claim", "grounding"}),
        ],
        default="hallucination",
    )
    return f"{_humanize_identifier(subject)} {effect} {_humanize_identifier(target)}".strip()


def _question_label(text: str) -> str:
    tokens = set(_WORD_RE.findall(str(text or "").lower()))
    if not tokens:
        return "Research Question"
    templates = [
        ("Hallucination Decision Geometry", {"concept", "space", "embedding", "yes", "no", "activations", "attention"}),
        ("Early Attention Grounding", {"attention", "claim", "head", "heads", "entropy"}),
        ("Cross-Position Residual Alignment", {"cross", "position", "cosine", "claim", "prompt", "output"}),
        ("Residual Decision Layer", {"residual", "hidden", "probe", "layer", "pca"}),
        ("Domain Generalization Failure", {"domain", "cluster", "generalization", "stratified"}),
        ("Entropy Regularization Baseline", {"raw", "entropy", "regularization", "baseline"}),
        ("Joint Feature Combination", {"joint", "classifier", "combined", "spearman"}),
        ("Head-Level Feature Selection", {"feature", "selection", "head", "heads", "univariate"}),
    ]
    best_name = "Research Question"
    best_score = 0
    for name, keywords in templates:
        score = len(tokens & keywords)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score > 0:
        return best_name
    return _short_label(text, max_tokens=4, title_case=True)


def _concise_display_name(node_type: str, raw_name: str, properties: dict[str, Any]) -> str:
    if node_type == "Method":
        return _humanize_identifier(str(properties.get("implementation_class") or raw_name or properties.get("proposal_name") or ""))
    if node_type == "MethodFamily":
        return _humanize_identifier(str(properties.get("family_name") or raw_name or ""))
    if node_type == "Question":
        return _question_label(str(properties.get("research_direction") or raw_name or ""))
    if node_type == "Hypothesis":
        return _hypothesis_label(str(properties.get("hypothesis_text") or raw_name or ""))
    if node_type == "Evidence":
        proposal = _humanize_identifier(str(properties.get("proposal_name") or ""))
        dataset = str(properties.get("dataset") or "")
        metric_name = str(properties.get("metric_name") or "")
        metric_value = properties.get("metric_value")
        assessment = str(properties.get("assessment") or "")
        if proposal and metric_name and metric_value not in (None, ""):
            return f"{proposal} {metric_name}={float(metric_value):.3f}"
        if proposal and assessment:
            return f"{proposal} {assessment}"
        if proposal and dataset:
            return f"{proposal} on {dataset}"
        return _short_label(raw_name, max_tokens=6, title_case=True)
    if node_type == "Finding":
        proposal = _humanize_identifier(str(properties.get("proposal_name") or ""))
        assessment = str(properties.get("assessment") or "")
        if proposal and assessment:
            return f"{proposal} {assessment}"
        if proposal:
            return proposal
        return _short_label(raw_name, max_tokens=6, title_case=True)
    if node_type == "Limitation":
        return _short_label(raw_name, max_tokens=6, title_case=True)
    if node_type == "Dataset":
        return str(properties.get("dataset") or raw_name or "").strip()
    if node_type == "Metric":
        return str(properties.get("metric_name") or raw_name or "").strip()
    if node_type == "PerformanceBand":
        return str(properties.get("band_key") or raw_name or "").strip()
    return ""


def _method_family_name(method_name: str, method_props: dict[str, Any]) -> str:
    normalized = _normalize_text(str(method_name or method_props.get("proposal_name") or ""))
    if "norm" in normalized and "trajectory" in normalized:
        return "norm_trajectory"
    if "entropy" in normalized and "attention" in normalized:
        return "attention_entropy"
    if "entropy" in normalized:
        return "entropy_features"
    if "cosine" in normalized or "cross_position" in normalized:
        return "cross_position_similarity"
    if "pca" in normalized:
        return "residual_pca"
    if "projection" in normalized:
        return "projection_dynamics"
    if "residual" in normalized:
        return "residual_dynamics"
    return _normalize_text(_humanize_identifier(method_name or str(method_props.get("proposal_name") or "")))


def _should_try_llm(*, node_type: str, candidates: list[GraphCanonicalCandidate], aliases: list[str]) -> bool:
    if node_type not in {"MethodFamily", "Method", "Hypothesis", "Limitation"}:
        return False
    if not candidates:
        return False
    alias_tokens = _token_set(" ".join(aliases))
    best_score = 0.0
    for candidate in candidates:
        candidate_tokens = _token_set(" ".join([candidate.display_name] + candidate.aliases))
        union = len(alias_tokens | candidate_tokens) or 1
        score = len(alias_tokens & candidate_tokens) / union
        best_score = max(best_score, score)
    return 0.35 <= best_score < 0.95


def _first_matching(tokens: list[str], options: list[tuple[str, set[str]]], *, default: str = "") -> str:
    token_set = set(tokens)
    best = default
    best_score = 0
    for name, keywords in options:
        score = len(token_set & keywords)
        if score > best_score:
            best = name
            best_score = score
    return best


def _metric_bands_from_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    kg_cfg = profile.get("knowledge_graph") or {}
    declared = kg_cfg.get("metric_bands")
    if isinstance(declared, list):
        return [dict(item) for item in declared if isinstance(item, dict)]
    evaluation = profile.get("evaluation") or {}
    primary_metric = str(evaluation.get("primary_metric") or "").strip()
    thresholds = dict(evaluation.get("thresholds") or {})
    bands: list[dict[str, Any]] = []
    if primary_metric and primary_metric in thresholds:
        base = float(thresholds[primary_metric])
        for threshold in sorted({base, max(base, 0.9), max(base, 0.95)}):
            bands.append({
                "metric_name": primary_metric,
                "operator": ">=",
                "threshold": threshold,
                "display_name": f"{primary_metric} >= {threshold:.2f}",
                "band_key": f"{primary_metric}_gte_{str(threshold).replace('.', '_')}",
            })
    return bands


def _metric_bands_for_value(metric_name: str, value: float, kg_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not metric_name or math.isnan(value):
        return []
    matched: list[dict[str, Any]] = []
    for band in (kg_cfg.get("metric_bands") or []):
        if str(band.get("metric_name") or "") != metric_name:
            continue
        operator = str(band.get("operator") or ">=")
        threshold = float(band.get("threshold") or 0.0)
        if operator == ">=" and value >= threshold:
            matched.append(dict(band))
        elif operator == "<=" and value <= threshold:
            matched.append(dict(band))
        elif operator == ">" and value > threshold:
            matched.append(dict(band))
        elif operator == "<" and value < threshold:
            matched.append(dict(band))
    return matched


def _primary_metric_name(profile: dict[str, Any], metrics: dict[str, Any], metadata: dict[str, Any]) -> str:
    primary = str((profile.get("evaluation") or {}).get("primary_metric") or metadata.get("primary_metric") or "").strip()
    if primary:
        return primary
    for key in metrics:
        return str(key)
    return ""


def _primary_metric_value(profile: dict[str, Any], metrics: dict[str, Any], metadata: dict[str, Any]) -> float:
    metric_name = _primary_metric_name(profile, metrics, metadata)
    try:
        return float(metrics.get(metric_name, metadata.get(metric_name, 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _primary_threshold(profile: dict[str, Any], metadata: dict[str, Any]) -> float:
    evaluation = profile.get("evaluation") or {}
    primary = str(evaluation.get("primary_metric") or metadata.get("primary_metric") or "").strip()
    thresholds = dict(evaluation.get("thresholds") or {})
    try:
        return float(thresholds.get(primary, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _supports_hypothesis(metadata: dict[str, Any], metrics: dict[str, Any], profile: dict[str, Any]) -> bool:
    if "hypothesis_supported" in metadata:
        return bool(metadata.get("hypothesis_supported"))
    return _primary_metric_value(profile, metrics, metadata) >= _primary_threshold(profile, metadata)


def _hypothesis_status(metadata: dict[str, Any], metrics: dict[str, Any], profile: dict[str, Any]) -> str:
    return "supported" if _supports_hypothesis(metadata, metrics, profile) else "contradicted"


def _metric_direction(metric_name: str, kg_cfg: dict[str, Any]) -> str:
    metric_cfg = dict((kg_cfg.get("metrics") or {}).get(metric_name) or {})
    return str(metric_cfg.get("direction") or "higher_is_better")


def _evidence_summary(record: MemoryRecord, metrics: dict[str, Any]) -> str:
    summary = str(record.get("summary") or "").strip()
    if summary:
        return summary
    parts = [str(record.get("title") or "experiment")]
    if metrics:
        parts.append(", ".join(f"{k}={v}" for k, v in metrics.items()))
    return " | ".join(parts)


def _finding_text(record: MemoryRecord, metrics: dict[str, Any]) -> str:
    metadata = dict(record.get("metadata") or {})
    interpretation = str(metadata.get("interpretation") or "").strip()
    assessment = str(metadata.get("assessment") or "").strip()
    if interpretation:
        return interpretation
    if assessment and metrics:
        return f"{record.get('title')}: {assessment} result with metrics {metrics}"
    return ""


def _evidence_confidence(metadata: dict[str, Any], metrics: dict[str, Any], profile: dict[str, Any]) -> str:
    value = _primary_metric_value(profile, metrics, metadata)
    if value >= 0.9:
        return "high"
    if value >= 0.75:
        return "medium"
    return "low"


def _dedupe_nodes(nodes: list[ResearchKGNode]) -> list[ResearchKGNode]:
    deduped: dict[str, ResearchKGNode] = {}
    for node in nodes:
        canonical_id = str(node.get("canonical_id") or "")
        if not canonical_id:
            continue
        if canonical_id not in deduped:
            deduped[canonical_id] = {
                **node,
                "aliases": list(node.get("aliases") or []),
                "properties": dict(node.get("properties") or {}),
            }
            continue
        existing = deduped[canonical_id]
        existing["aliases"] = sorted(set((existing.get("aliases") or []) + (node.get("aliases") or [])))
        existing_props = dict(existing.get("properties") or {})
        for key, value in dict(node.get("properties") or {}).items():
            if value not in ("", None, [], {}):
                existing_props[key] = value
        existing["properties"] = existing_props
    return list(deduped.values())


def _dedupe_relations(relations: list[ResearchKGRelation]) -> list[ResearchKGRelation]:
    deduped: dict[str, ResearchKGRelation] = {}
    for relation in relations:
        key = "|".join([
            str(relation.get("relation_type") or ""),
            str(relation.get("source_id") or ""),
            str(relation.get("target_id") or ""),
            json.dumps(dict(relation.get("properties") or {}), sort_keys=True, default=str),
        ])
        deduped[key] = relation
    return list(deduped.values())
