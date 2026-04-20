"""Feature proposal node — two-stage experiment ideation and implementation planning."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from configs.prompts import FEATURE_PROPOSAL_IDEATION, FEATURE_PROPOSAL_IMPLEMENTATION
from graph.state import ResearchState
from llm.factory import get_llm
from utils import extract_json_array
from utils.dataset_registry import get_all_datasets
from utils.logger import get_logger

log = get_logger(__name__)


def feature_proposal_node(state: ResearchState) -> dict:
    """Propose neuralsignal experiments in two stages.

    Stage 1 — Ideation:
        Uses the research summary and paper digests to generate 3–5 feature
        extraction ideas grounded in LLM internals and mechanistic interpretability.

    Stage 2 — Implementation planning:
        For each idea, produces a concrete neuralsignal implementation plan:
        which classes to use, feature sets, Collector config, probe model,
        training dataset, detector type, and step-by-step code guidance.

    Reads:
        state['research_direction']
        state['research_summary']
        state['paper_digests']    — preferred; falls back to arxiv_papers abstracts
        state['arxiv_papers']

    Writes:
        state['feature_proposals'] — list of fully-specified experiment proposals
    """
    direction = state.get("research_direction", "")
    summary = state.get("research_summary", "")
    digests = state.get("paper_digests") or []
    papers = state.get("arxiv_papers") or []
    llm = get_llm()

    log.info("feature_proposal_node | Starting — direction=%r, digests=%d, papers=%d",
             direction, len(digests), len(papers))

    # ------------------------------------------------------------------
    # Build research context block for the LLM
    # ------------------------------------------------------------------
    context_parts = [f"Research direction: {direction}", "", f"Research summary:\n{summary}"]

    if digests:
        context_parts.append("\nPaper digests (full-text extractions):")
        for d in digests:
            context_parts.append(f"\n[{d['title']}]\n{d['digest']}")
    elif papers:
        context_parts.append("\nRelevant papers (abstracts):")
        for p in papers[:5]:
            context_parts.append(
                f"\n[{p['title']} — score {p.get('relevance_score', '?')}]\n{p['abstract']}"
            )

    research_context = "\n".join(context_parts)

    # ------------------------------------------------------------------
    # Stage 1: Ideation — generate raw feature ideas
    # ------------------------------------------------------------------
    log.info("feature_proposal_node | Stage 1: generating feature ideas")
    try:
        ideation_response = llm.invoke([
            SystemMessage(content=FEATURE_PROPOSAL_IDEATION),
            HumanMessage(content=research_context),
        ])
        ideas = extract_json_array(ideation_response.content)
        log.info("feature_proposal_node | Stage 1 complete — %d ideas generated", len(ideas))
        for i, idea in enumerate(ideas):
            log.debug("  Idea %d: %s", i + 1, idea.get("name"))
    except Exception as exc:
        log.error("feature_proposal_node | Stage 1 failed: %s", exc, exc_info=True)
        return {"feature_proposals": [], "errors": [f"Feature ideation failed: {exc}"]}

    if not ideas:
        log.warning("feature_proposal_node | No ideas generated — returning empty proposals")
        return {"feature_proposals": []}

    # ------------------------------------------------------------------
    # Stage 2: Implementation planning — ground ideas in neuralsignal API
    # ------------------------------------------------------------------
    log.info("feature_proposal_node | Stage 2: planning neuralsignal implementation")

    import json

    # Build a compact dataset summary for the LLM so it can choose the right
    # dataset and detector for each proposal.
    datasets = get_all_datasets()
    datasets_summary = json.dumps(
        [
            {
                "name": d["name"],
                "description": d.get("description", ""),
                "available_detectors": d.get("available_detectors", []),
            }
            for d in datasets
        ],
        indent=2,
    )

    # Gather available_scan_fields across all registered datasets
    all_guaranteed: set[str] = set()
    all_optional: set[str] = set()
    all_not_available: set[str] = set()
    for d in datasets:
        asf = d.get("available_scan_fields") or {}
        all_guaranteed.update(asf.get("guaranteed") or [])
        all_optional.update(asf.get("optional") or [])
        all_not_available.update(asf.get("not_available") or [])

    scan_constraints = (
        "Scan field constraints — feature set classes MUST only access these scan document fields:\n"
        f"  Guaranteed present: {sorted(all_guaranteed)}\n"
        f"  Optional (may be None/empty — check before use): {sorted(all_optional)}\n"
        f"  NOT available (do not access these — will cause zeros or errors): {sorted(all_not_available)}\n"
        "  outputs/inputs structure: scan['outputs'][batch_idx][layer_id] → Tensor\n"
        "  Do NOT require external files (probe weights, concept means, logit lens projections, etc.)\n"
        "  Do NOT require activation patching, causal intervention, or attention score data.\n"
        "  Layer pattern config keys: 'ffn_layer_patterns' (list[str]) and 'attn_layer_patterns' (list[str])\n"
        "    — these will be injected from the dataset registry at instantiation time; use self.cfg.get() to read them.\n"
    )

    ideas_json = json.dumps(ideas, indent=2)
    implementation_prompt = (
        f"Research direction: {direction}\n\n"
        f"Available datasets (use the exact 'name' value in your proposals' "
        f"'dataset' field; choose 'detector_name' from 'available_detectors'):\n"
        f"{datasets_summary}\n\n"
        f"{scan_constraints}\n"
        f"Feature ideas to implement:\n{ideas_json}"
    )

    try:
        impl_response = llm.invoke([
            SystemMessage(content=FEATURE_PROPOSAL_IMPLEMENTATION),
            HumanMessage(content=implementation_prompt),
        ])
        proposals = extract_json_array(impl_response.content)
        log.info("feature_proposal_node | Stage 2 complete — %d proposals with implementation plans",
                 len(proposals))
        for i, p in enumerate(proposals):
            log.debug("  Proposal %d: %s | classes=%s | feature_sets=%s",
                      i + 1, p.get("name"), p.get("neuralsignal_classes"), p.get("feature_sets"))
    except Exception as exc:
        log.warning("feature_proposal_node | Stage 2 failed (%s) — using stage 1 ideas as-is", exc)
        proposals = ideas

    return {"feature_proposals": proposals}
