"""Implement step - generate code that subclasses a profile base class.

For each implementation plan, asks the LLM to generate a Python class file.
Generated files are cached under the configured experiments directory.

Reads:
    state['implementation_plans']
    state['profile_name']

Writes:
    state['implementations']  - {script_path, class_name, proposal_name, plan}
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from configs.config import get_config
from core.graph.nodes.artifact_refs import register_implementation_artifact
from core.graph.nodes.code_safety import extract_python_source, validate_python_source
from core.graph.nodes.memory import persist_memory_records_for_state
from core.graph.state import ResearchState
from core.llm.factory import get_llm
from core.utils.logger import get_logger
from core.utils.profile_loader import get_prompt

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
    implement_cfg = profile.get("implement") or {}
    max_syntax_retries = int(implement_cfg.get("max_syntax_retries", 2) or 2)

    base_class_docs = _base_class_context(profile)
    scan_constraints = _scan_constraints_context(profile)
    implementation_examples = _load_implementation_examples(profile)

    cache_dir = _script_cache_dir(profile_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    implementations: list[dict] = []
    errors = list(state.get("errors") or [])

    for idx, plan in enumerate(plans):
        if not isinstance(plan, dict):
            errors.append(
                f"implement: skipped malformed implementation plan at index {idx}: expected object, got {type(plan).__name__}"
            )
            continue
        class_name = plan.get("class_name") or plan.get("proposal_name", "UnknownClass")
        proposal_name = plan.get("proposal_name", class_name)
        cache_path = cache_dir / f"{class_name}.py"

        if cache_path.exists():
            try:
                cached_code = cache_path.read_text(encoding="utf-8")
                validate_python_source(cached_code, expected_class_name=class_name)
                log.info("implement_node | Cache hit - %s", cache_path)
                implementation = {
                    "script_path": str(cache_path),
                    "class_name": class_name,
                    "proposal_name": proposal_name,
                    "plan": plan,
                    "cached": True,
                }
                register_implementation_artifact(profile, implementation, errors)
                implementations.append(implementation)
                continue
            except Exception as exc:
                log.warning(
                    "implement_node | Ignoring invalid cached implementation %s: %s",
                    cache_path,
                    exc,
                )

        user_content = (
            f"Base classes available:\n{base_class_docs}\n\n"
            f"Scan field constraints:\n{scan_constraints}\n\n"
            f"Reference implementation examples:\n{implementation_examples}\n\n"
            f"Proposal execution context:\n{_proposal_execution_context(state, proposal_name)}\n\n"
            f"Implementation plan:\n{json.dumps(plan, indent=2)}"
        )

        log.info("implement_node | Generating code for %r", class_name)
        resp = None
        try:
            code = ""
            last_error: Exception | None = None
            for attempt in range(max_syntax_retries + 1):
                if attempt == 0:
                    prompt_body = user_content
                else:
                    prompt_body = (
                        f"{user_content}\n\n"
                        "Your previous response was rejected.\n"
                        "Return raw Python source only.\n"
                        "Do not include markdown fences.\n"
                        "Do not include explanatory prose.\n"
                        "If you write comments, prefix them with '#'.\n"
                        f"Previous validation error: {last_error}\n"
                    )
                resp = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt_body),
                ])
                code = extract_python_source(resp.content)
                try:
                    validate_python_source(code, expected_class_name=class_name)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_syntax_retries:
                        raise
                    log.warning(
                        "implement_node | Retrying %r after invalid code response (%d/%d): %s",
                        class_name,
                        attempt + 1,
                        max_syntax_retries,
                        exc,
                    )

            cache_path.write_text(code, encoding="utf-8")
            log.info("implement_node | Saved %d lines -> %s", len(code.splitlines()), cache_path)
            implementation = {
                "script_path": str(cache_path),
                "class_name": class_name,
                "proposal_name": proposal_name,
                "plan": plan,
                "cached": False,
            }
            register_implementation_artifact(profile, implementation, errors)
            implementations.append(implementation)
        except Exception as exc:
            log.error("implement_node | Code generation failed for %r: %s", class_name, exc, exc_info=True)
            if resp is not None:
                log.error(
                    "implement_node | Raw LLM response for %r follows:\n%s",
                    class_name,
                    _truncate_text(str(getattr(resp, "content", "") or ""), 12000),
                )
            implementations.append({
                "script_path": "",
                "class_name": class_name,
                "proposal_name": proposal_name,
                "plan": plan,
                "error": str(exc),
            })
            errors.append(f"implement: {class_name} failed: {exc}")

    delta = {"implementations": implementations}
    if errors:
        delta["errors"] = errors
    try:
        persist_memory_records_for_state(profile, {**state, **delta})
    except Exception as exc:
        log.warning("implement_node | Memory persistence failed: %s", exc)
        delta["errors"] = list(delta.get("errors") or []) + [f"implement: memory persistence failed: {exc}"]
    return delta


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```python ... ```) from LLM output."""
    return extract_python_source(text)


def _load_implementation_examples(profile: dict) -> str:
    """Load profile-declared implementation examples for code-generation context."""
    examples = profile.get("implementation_examples") or []
    if not examples:
        return "(none)"

    rendered: list[str] = []
    for example in examples:
        path = Path(example.get("path", ""))
        purpose = example.get("purpose", "")
        if not path.exists():
            rendered.append(f"[missing example: {path}] {purpose}")
            continue
        try:
            code = path.read_text(encoding="utf-8")
        except Exception as exc:
            rendered.append(f"[unreadable example: {path}] {exc}")
            continue
        excerpt = _truncate_text(code, 3500)
        rendered.append(
            f"Example path: {path}\n"
            f"Purpose: {purpose}\n"
            "Use this as API/style reference only; do not copy feature logic unless explicitly requested.\n"
            f"```python\n{excerpt}\n```"
        )
    return "\n\n".join(rendered)


def _base_class_context(profile: dict) -> str:
    base_classes = profile.get("base_classes") or []
    rendered: list[str] = []
    for bc in base_classes:
        rendered.append(
            f"Base class: {bc['name']}\n"
            f"Module: {bc.get('module', 'n/a')}\n"
            f"Description: {bc.get('description', '')}\n"
            f"Key interface excerpt:\n{_truncate_text(bc.get('key_interface', ''), 2600)}"
        )
    return "\n\n".join(rendered)


def _scan_constraints_context(profile: dict) -> str:
    datasets = profile.get("datasets") or []
    parts: list[str] = []
    for ds in datasets:
        asf = ds.get("available_scan_fields") or {}
        layer_patterns = ds.get("layer_name_patterns") or {}
        parts.append(
            f"Dataset '{ds['name']}':\n"
            f"  Guaranteed: {asf.get('guaranteed', [])}\n"
            f"  NOT available: {asf.get('not_available', [])}\n"
            f"  FFN patterns: {layer_patterns.get('ffn', [])[:6]}\n"
            f"  Attn patterns: {layer_patterns.get('attn', [])[:6]}"
        )
    return "\n".join(parts)


def _proposal_execution_context(state: ResearchState, proposal_name: str) -> str:
    proposals = state.get("proposals") or []
    proposal = next(
        (
            item for item in proposals
            if isinstance(item, dict) and str(item.get("name") or "") == str(proposal_name or "")
        ),
        None,
    )
    if not isinstance(proposal, dict):
        return (
            "No matching proposal found in state. "
            "If the strategy expects fixed symbols, declare them through cfg['symbol'] or cfg['symbols']; "
            "otherwise iterate over the incoming PriceData symbols."
        )

    explicit_symbol = proposal.get("symbol")
    explicit_symbols = proposal.get("symbols")
    explicit_universe = proposal.get("universe")
    if isinstance(explicit_symbols, str):
        explicit_symbols = [explicit_symbols]
    elif isinstance(explicit_symbols, (tuple, set)):
        explicit_symbols = list(explicit_symbols)
    if isinstance(explicit_universe, str):
        explicit_universe = [explicit_universe]
    elif isinstance(explicit_universe, (tuple, set)):
        explicit_universe = list(explicit_universe)

    expected_symbols: list[str] = []
    if explicit_symbol:
        expected_symbols.append(str(explicit_symbol))
    if isinstance(explicit_symbols, list):
        expected_symbols.extend(str(item) for item in explicit_symbols if item)
    if isinstance(explicit_universe, list):
        expected_symbols.extend(str(item) for item in explicit_universe if item)
    expected_symbols = list(dict.fromkeys(expected_symbols))

    context = {
        "proposal_name": proposal.get("name"),
        "symbol": explicit_symbol,
        "symbols": explicit_symbols,
        "universe": explicit_universe,
        "data_source": proposal.get("data_source"),
        "timeframe": proposal.get("timeframe"),
        "mode": proposal.get("mode"),
        "history_length": proposal.get("history_length"),
        "guidance": [
            "If expected_symbols is non-empty, the implementation may target those symbols, but it must read them from cfg/proposal-facing fields rather than hardcoding ticker literals deep in the logic.",
            "If expected_symbols is empty, the implementation should work from the symbols present in the incoming PriceData tick.",
            "Do not reimplement base-class history tracking or warmup gating.",
        ],
        "expected_symbols": expected_symbols,
    }
    return json.dumps(context, indent=2)


def _truncate_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
