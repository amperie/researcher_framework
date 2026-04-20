"""Implement step — generate code that subclasses a profile base class.

For each implementation plan, asks the LLM to generate a Python class file.
Generated files are cached at dev/experiments/<profile>/<class_name>.py.

Reads:
    state['implementation_plans']
    state['profile_name']

Writes:
    state['implementations']  — {script_path, class_name, proposal_name, plan}
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from configs.config import get_config
from graph.state import ResearchState
from llm.factory import get_llm
from utils.logger import get_logger
from utils.profile_loader import get_prompt

log = get_logger(__name__)


def _script_cache_dir(profile_name: str) -> Path:
    cfg = get_config()
    return Path(cfg.experiments_dir) / profile_name / "implementations"


def implement_node(state: ResearchState, profile: dict) -> dict:
    plans = state.get("implementation_plans") or []
    profile_name = state.get("profile_name") or profile.get("name", "unknown")

    if not plans:
        log.warning("implement_node | No implementation plans in state")
        return {"implementations": []}

    system_prompt = get_prompt(profile, "implement")
    llm = get_llm("implement", profile)

    # Build full base class context for the code-gen prompt
    base_classes = profile.get("base_classes") or []
    base_class_docs = "\n\n".join(
        f"Base class: {bc['name']}\nModule: {bc.get('module', 'n/a')}\n"
        f"Key interface:\n{bc.get('key_interface', '')}"
        for bc in base_classes
    )

    # Build scan constraints block
    datasets = profile.get("datasets") or []
    scan_constraints_parts = []
    for ds in datasets:
        asf = ds.get("available_scan_fields") or {}
        layer_patterns = ds.get("layer_name_patterns") or {}
        scan_constraints_parts.append(
            f"Dataset '{ds['name']}':\n"
            f"  Guaranteed: {asf.get('guaranteed', [])}\n"
            f"  Optional: {asf.get('optional', [])}\n"
            f"  NOT available (do not access): {asf.get('not_available', [])}\n"
            f"  FFN layer patterns (injected via cfg): {layer_patterns.get('ffn', [])}\n"
            f"  Attn layer patterns (injected via cfg): {layer_patterns.get('attn', [])}"
        )
    scan_constraints = "\n".join(scan_constraints_parts)

    cache_dir = _script_cache_dir(profile_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    implementations: list[dict] = []

    for plan in plans:
        class_name = plan.get("class_name") or plan.get("proposal_name", "UnknownClass")
        proposal_name = plan.get("proposal_name", class_name)
        cache_path = cache_dir / f"{class_name}.py"

        if cache_path.exists():
            log.info("implement_node | Cache hit — %s", cache_path)
            implementations.append({
                "script_path": str(cache_path),
                "class_name": class_name,
                "proposal_name": proposal_name,
                "plan": plan,
                "cached": True,
            })
            continue

        user_content = (
            f"Base classes available:\n{base_class_docs}\n\n"
            f"Scan field constraints:\n{scan_constraints}\n\n"
            f"Implementation plan:\n{json.dumps(plan, indent=2)}"
        )

        log.info("implement_node | Generating code for %r", class_name)
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            code = _strip_fences(resp.content)
            if not code:
                raise ValueError("LLM returned empty code")

            cache_path.write_text(code, encoding="utf-8")
            log.info("implement_node | Saved %d lines → %s", len(code.splitlines()), cache_path)
            implementations.append({
                "script_path": str(cache_path),
                "class_name": class_name,
                "proposal_name": proposal_name,
                "plan": plan,
                "cached": False,
            })
        except Exception as exc:
            log.error("implement_node | Code generation failed for %r: %s", class_name, exc, exc_info=True)
            implementations.append({
                "script_path": "",
                "class_name": class_name,
                "proposal_name": proposal_name,
                "plan": plan,
                "error": str(exc),
            })

    return {"implementations": implementations}


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```python ... ```) from LLM output."""
    import re
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()
