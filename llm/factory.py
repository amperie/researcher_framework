"""LLM provider factory.

Model resolution order (first non-None wins):
  1. profile['llm']['step_overrides'][step_name]  (per-step override in profile)
  2. profile['llm']['default_model']              (profile-level default)
  3. Config.llm_model                             (global .env override)
  4. Provider built-in default                   (claude-opus-4-6 / gpt-4o)
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from configs.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


def get_llm(
    step_name: str | None = None,
    profile: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """Return a configured chat model.

    Args:
        step_name: Pipeline step name (e.g. 'implement'). Used to resolve
                   per-step model overrides from the profile.
        profile:   Loaded profile dict. If given, model is resolved from
                   profile['llm']['step_overrides'][step_name] or
                   profile['llm']['default_model'].
        provider:  'anthropic' or 'openai'. Falls back to Config.llm_provider.
        model:     Explicit model ID override (highest priority).
    """
    cfg = get_config()
    provider = provider or cfg.llm_provider

    # Resolve model: explicit arg → step override → profile default → global → provider default
    resolved_model = model
    if resolved_model is None and profile:
        llm_cfg = profile.get("llm") or {}
        overrides = llm_cfg.get("step_overrides") or {}
        if step_name and step_name in overrides:
            resolved_model = overrides[step_name]
            log.debug("get_llm | step override — step=%r model=%r", step_name, resolved_model)
        elif llm_cfg.get("default_model"):
            resolved_model = llm_cfg["default_model"]
            log.debug("get_llm | profile default — model=%r", resolved_model)
    if resolved_model is None:
        resolved_model = cfg.llm_model

    log.info(
        "get_llm | provider=%r model=%r (step=%r)",
        provider, resolved_model or "(provider default)", step_name,
    )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=resolved_model or "claude-opus-4-6",
            api_key=cfg.anthropic_api_key,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=resolved_model or "gpt-4o",
            api_key=cfg.openai_api_key,
        )
    raise ValueError(
        f"Unknown LLM provider {provider!r}. Expected 'anthropic' or 'openai'."
    )
