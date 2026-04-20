"""Shared experiment adapter contract.

Domain plugins implement this interface so graph nodes can stay domain-agnostic.
The legacy module-level functions are still supported by the generic nodes, but
new plugins should expose `get_adapter()` returning a ResearchAdapter instance.
"""
from __future__ import annotations

from typing import Any, Protocol


class ResearchAdapter(Protocol):
    """Protocol implemented by domain experiment adapters."""

    def validate_environment(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Return adapter readiness information without running an experiment."""
        ...

    def build_context(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Return domain-specific context to inject into planning/prompts."""
        ...

    def prepare_experiment(
        self,
        profile: dict[str, Any],
        proposal: dict[str, Any],
        implementation: dict[str, Any] | None,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Prepare reusable artifacts needed before experiment execution."""
        ...

    def execute_experiment(
        self,
        profile: dict[str, Any],
        proposal: dict[str, Any],
        implementation: dict[str, Any] | None,
        artifact: dict[str, Any] | None,
        experiment_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Run the domain experiment and return a normalized result dict."""
        ...

    def summarize_result(
        self,
        profile: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a compact domain-specific summary for storage/evaluation."""
        ...

