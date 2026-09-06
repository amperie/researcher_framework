"""LLM provider factory.

Model resolution order (first non-None wins):
  1. profile['llm']['step_overrides'][step_name]  (per-step override in profile)
  2. profile['llm']['default_model']              (profile-level default)
  3. Config.llm_model                             (global .env override)
  4. Provider built-in default                   (claude-opus-4-6 / gpt-4o)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from configs.config import get_config
from core.utils.logger import get_logger

log = get_logger(__name__)

_TOKEN_TOTALS: dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
_TOKEN_EVENTS: list[dict[str, Any]] = []


def reset_usage() -> None:
    _TOKEN_EVENTS.clear()
    for key in _TOKEN_TOTALS:
        _TOKEN_TOTALS[key] = 0


def get_usage_report() -> dict[str, Any]:
    providers = sorted(
        {str(item.get("provider") or "") for item in _TOKEN_EVENTS if item.get("provider")}
    )
    models = sorted({str(item.get("model") or "") for item in _TOKEN_EVENTS if item.get("model")})
    return {
        "provider": providers[0] if len(providers) == 1 else None,
        "model": models[0] if len(models) == 1 else None,
        "promptTokens": _TOKEN_TOTALS["prompt_tokens"],
        "completionTokens": _TOKEN_TOTALS["completion_tokens"],
        "totalTokens": _TOKEN_TOTALS["total_tokens"],
        "calls": len(_TOKEN_EVENTS),
        "steps": list(_TOKEN_EVENTS),
    }


@dataclass
class LoggedChatModel:
    """Thin proxy that logs usage metadata from model responses."""

    inner: BaseChatModel
    step_name: str | None = None
    model_name: str = ""
    provider: str = ""

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        response = self.inner.invoke(*args, **kwargs)
        _log_usage(
            step_name=self.step_name or "unknown",
            provider=self.provider,
            model_name=self.model_name,
            response=response,
        )
        return response

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


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

    llm_cfg = (profile or {}).get("llm") or {}
    resolved_step_cfg = _resolve_step_config(step_name=step_name, llm_cfg=llm_cfg)

    resolved_model = model or resolved_step_cfg.get("model")
    if resolved_model is None:
        resolved_model = cfg.llm_model
    max_tokens = resolved_step_cfg.get("max_output_tokens")

    log.info(
        "get_llm | provider=%r model=%r step=%r max_output_tokens=%r",
        provider,
        resolved_model or "(provider default)",
        step_name,
        max_tokens,
    )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        inner = ChatAnthropic(
            model=resolved_model or "claude-opus-4-6",
            api_key=cfg.anthropic_api_key,
            max_tokens=max_tokens,
        )
        return LoggedChatModel(
            inner=inner,
            step_name=step_name,
            model_name=resolved_model or "claude-opus-4-6",
            provider=provider,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        inner = ChatOpenAI(
            model=resolved_model or "gpt-4o",
            api_key=cfg.openai_api_key,
            max_tokens=max_tokens,
        )
        return LoggedChatModel(
            inner=inner,
            step_name=step_name,
            model_name=resolved_model or "gpt-4o",
            provider=provider,
        )
    raise ValueError(
        f"Unknown LLM provider {provider!r}. Expected 'anthropic' or 'openai'."
    )


def _resolve_step_config(*, step_name: str | None, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    default_model = llm_cfg.get("default_model")
    if isinstance(default_model, dict):
        resolved.update(_normalize_model_config(default_model))
    elif default_model:
        resolved["model"] = str(default_model)

    if llm_cfg.get("default_max_output_tokens") is not None:
        resolved["max_output_tokens"] = int(llm_cfg["default_max_output_tokens"])

    overrides = llm_cfg.get("step_overrides") or {}
    if step_name and step_name in overrides:
        override = overrides[step_name]
        resolved.update(_normalize_model_config(override))
        log.debug("get_llm | step override step=%r config=%r", step_name, resolved)
    elif resolved.get("model"):
        log.debug("get_llm | profile default model=%r", resolved["model"])

    return resolved


def _normalize_model_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        if value.get("model"):
            normalized["model"] = str(value["model"])
        if value.get("max_output_tokens") is not None:
            normalized["max_output_tokens"] = int(value["max_output_tokens"])
        return normalized
    if value:
        return {"model": str(value)}
    return {}


def _log_usage(*, step_name: str, provider: str, model_name: str, response: Any) -> None:
    usage = _extract_usage_metadata(response)
    if not usage:
        log.debug(
            "llm usage | step=%r provider=%r model=%r usage=unavailable",
            step_name,
            provider,
            model_name,
        )
        return

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    _TOKEN_TOTALS["prompt_tokens"] += prompt_tokens
    _TOKEN_TOTALS["completion_tokens"] += completion_tokens
    _TOKEN_TOTALS["total_tokens"] += total_tokens
    _TOKEN_EVENTS.append(
        {
            "step": step_name,
            "provider": provider,
            "model": model_name,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
        }
    )

    log.info(
        "llm usage | step=%r provider=%r model=%r prompt_tokens=%d completion_tokens=%d total_tokens=%d cumulative_total_tokens=%d",
        step_name,
        provider,
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        _TOKEN_TOTALS["total_tokens"],
    )


def _extract_usage_metadata(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        return {
            "prompt_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "completion_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    if token_usage:
        return {
            "prompt_tokens": int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0),
            "completion_tokens": int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0),
            "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
        }

    return {}
