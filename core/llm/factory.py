"""LLM provider factory.

Model resolution order (first non-None wins):
  1. profile['llm']['step_overrides'][step_name]  (per-step override in profile)
  2. profile['llm']['default_model']              (profile-level default)
  3. Config.llm_model                             (global .env override)
  4. Provider built-in default                   (claude-opus-4-6 / gpt-4o)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from configs.config import get_config
from core.llm.usage import finish_call, get_usage_report, reset_usage, start_call
from core.utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class LoggedChatModel:
    """Thin proxy that logs usage metadata from model responses."""

    inner: BaseChatModel
    step_name: str | None = None
    model_name: str = ""
    provider: str = ""

    def _start(self):
        return start_call(self.step_name or "unknown", self.provider, self.model_name)

    def _finish(self, event, response=None, status="succeeded"):
        result = finish_call(event, response, status)
        log.info("llm usage | %s", result)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        event = self._start()
        try:
            response = self.inner.invoke(*args, **kwargs)
        except BaseException:
            self._finish(event, status="failed")
            raise
        self._finish(event, response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        event = self._start()
        try:
            response = await self.inner.ainvoke(*args, **kwargs)
        except asyncio.CancelledError:
            self._finish(event, status="interrupted")
            raise
        except BaseException:
            self._finish(event, status="failed")
            raise
        self._finish(event, response)
        return response

    def stream(self, *args: Any, **kwargs: Any):
        event, response, status = self._start(), None, "interrupted"
        try:
            for chunk in self.inner.stream(*args, **kwargs):
                response = chunk if response is None else response + chunk
                yield chunk
            status = "succeeded"
        finally:
            self._finish(event, response, status)

    async def astream(self, *args: Any, **kwargs: Any):
        event, response, status = self._start(), None, "interrupted"
        try:
            async for chunk in self.inner.astream(*args, **kwargs):
                response = chunk if response is None else response + chunk
                yield chunk
            status = "succeeded"
        finally:
            self._finish(event, response, status)

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
    llm_cfg = (profile or {}).get("llm") or {}
    provider = provider or llm_cfg.get("provider") or cfg.llm_provider
    resolved_step_cfg = _resolve_step_config(step_name=step_name, llm_cfg=llm_cfg)

    resolved_model = model or resolved_step_cfg.get("model")
    if resolved_model is None:
        resolved_model = cfg.llm_model
    max_tokens = resolved_step_cfg.get("max_output_tokens")
    runtime = {key: llm_cfg[key] for key in ("timeout", "max_retries") if key in llm_cfg}

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
            **runtime,
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
            **runtime,
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
    finish_call(start_call(step_name, provider, model_name), response)
