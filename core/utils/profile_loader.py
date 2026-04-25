"""Research profile loader.

Loads and validates a research profile YAML from configs/profiles/<name>.yaml.
The profile is the single source of truth for a research domain — it controls
pipeline steps, LLM models, prompts, datasets, base classes, and storage targets.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from core.utils.logger import get_logger

log = get_logger(__name__)

_PROFILES_DIR = Path("configs/profiles")

_REQUIRED_KEYS = {"name", "pipeline", "llm", "prompts"}


def load_profile(name: str) -> dict:
    """Load a research profile by name.

    Looks for ``configs/profiles/<name>.yaml``.

    Args:
        name: Profile name (e.g. 'neuralsignal', 'trading').

    Returns:
        Parsed profile dict.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError: If required top-level keys are missing.
    """
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Profile {name!r} not found at {path}. "
            f"Available profiles: {list_profiles()}"
        )

    with path.open(encoding="utf-8") as fh:
        profile = yaml.safe_load(fh) or {}

    missing = _REQUIRED_KEYS - set(profile.keys())
    if missing:
        raise ValueError(
            f"Profile {name!r} is missing required keys: {sorted(missing)}"
        )

    log.info(
        "profile_loader | Loaded profile %r — steps=%s",
        name,
        profile.get("pipeline", {}).get("steps", []),
    )
    return profile


@lru_cache(maxsize=8)
def load_profile_cached(name: str) -> dict:
    """Cached version of load_profile for repeated calls within a run."""
    return load_profile(name)


def list_profiles() -> list[str]:
    """Return names of all available profiles (without .yaml extension)."""
    if not _PROFILES_DIR.exists():
        return []
    return [p.stem for p in sorted(_PROFILES_DIR.glob("*.yaml"))]


def get_prompt(profile: dict, step: str, key: str = "system") -> str:
    """Retrieve a prompt string from a profile.

    Args:
        profile: Loaded profile dict.
        step:    Step name (e.g. 'ideate', 'implement').
        key:     Prompt key within the step (default: 'system').

    Returns:
        The prompt string.

    Raises:
        KeyError: If the step or key is not defined in the profile.
    """
    prompts = profile.get("prompts") or {}
    step_prompts = prompts.get(step)
    if step_prompts is None:
        raise KeyError(
            f"No prompts defined for step {step!r} in profile {profile.get('name')!r}. "
            f"Available steps: {list(prompts.keys())}"
        )
    if key not in step_prompts:
        raise KeyError(
            f"No prompt key {key!r} for step {step!r} in profile {profile.get('name')!r}. "
            f"Available keys: {list(step_prompts.keys())}"
        )
    return step_prompts[key]


def get_step_datasets(profile: dict) -> list[dict]:
    """Return the dataset list from the profile."""
    return profile.get("datasets") or []


def get_primary_dataset(profile: dict) -> dict | None:
    """Return the first dataset in the profile, or None if empty."""
    datasets = get_step_datasets(profile)
    return datasets[0] if datasets else None
