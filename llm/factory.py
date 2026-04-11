"""LLM provider factory.

Returns a configured LangChain chat model for the requested provider,
defaulting to the values in Config.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from configs.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Return a configured chat model for the given provider.

    Args:
        provider: 'anthropic' or 'openai'. Falls back to Config.llm_provider.
        model: Model ID override. Falls back to Config.llm_model, then provider default.

    Returns:
        An instantiated BaseChatModel ready for invocation.

    TODO: implement Anthropic branch (ChatAnthropic, default model claude-opus-4-6).
    TODO: implement OpenAI branch (ChatOpenAI, default model gpt-4o).
    TODO: raise ValueError with clear message for unknown providers.
    """
    cfg = get_config()
    provider = provider or cfg.llm_provider
    model = model or cfg.llm_model

    log.info("Creating LLM — provider=%r, model=%r", provider, model or "(provider default)")
    log.debug("API key present: %s",
              bool(cfg.anthropic_api_key if provider == "anthropic" else cfg.openai_api_key))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-opus-4-6",
            api_key=cfg.anthropic_api_key,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=cfg.openai_api_key,
        )
    raise ValueError(
        f"Unknown LLM provider {provider!r}. Expected 'anthropic' or 'openai'."
    )
