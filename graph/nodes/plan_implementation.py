"""Plan implementation step — produce a structured implementation plan per proposal.

Does NOT generate code. Produces a JSON plan that the implement step will use.

Reads:
    state['proposals']

Writes:
    state['implementation_plans']
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ResearchState
from llm.factory import get_llm
from utils import extract_json_array
from utils.logger import get_logger
from utils.profile_loader import get_prompt

log = get_logger(__name__)


def plan_implementation_node(state: ResearchState, profile: dict) -> dict:
    proposals = state.get("proposals") or []

    if not proposals:
        log.warning("plan_implementation_node | No proposals in state")
        return {"implementation_plans": []}

    system_prompt = get_prompt(profile, "plan_implementation")
    llm = get_llm("plan_implementation", profile)

    # Inject base class APIs into context so the LLM plans against the right interface
    base_classes = profile.get("base_classes") or []
    base_class_docs = "\n\n".join(
        f"Base class: {bc['name']}\nModule: {bc.get('module', 'n/a')}\n"
        f"Description: {bc.get('description', '')}\n"
        f"Interface:\n{bc.get('key_interface', '')}"
        for bc in base_classes
    )

    # Inject scan field constraints from each dataset
    datasets = profile.get("datasets") or []
    scan_constraints = ""
    for ds in datasets:
        asf = ds.get("available_scan_fields") or {}
        scan_constraints += (
            f"\nDataset '{ds['name']}' scan fields:\n"
            f"  Guaranteed: {asf.get('guaranteed', [])}\n"
            f"  Optional: {asf.get('optional', [])}\n"
            f"  NOT available: {asf.get('not_available', [])}\n"
        )

    user_content = (
        f"Available base classes:\n{base_class_docs}\n\n"
        f"Scan field constraints:{scan_constraints}\n"
        f"Proposals to plan:\n{json.dumps(proposals, indent=2)}"
    )

    log.info("plan_implementation_node | Planning %d proposals", len(proposals))
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        plans = extract_json_array(resp.content)
        log.info("plan_implementation_node | Generated %d implementation plans", len(plans))
    except Exception as exc:
        log.error("plan_implementation_node | Failed: %s", exc, exc_info=True)
        return {
            "implementation_plans": [],
            "errors": (state.get("errors") or []) + [f"plan_implementation failed: {exc}"],
        }

    return {"implementation_plans": plans}
