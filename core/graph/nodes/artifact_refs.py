"""Artifact registration helpers for graph node outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.artifacts import get_artifact_store
from core.utils.logger import get_logger

log = get_logger(__name__)


def register_implementation_artifact(
    profile: dict[str, Any],
    implementation: dict[str, Any],
    errors: list[str] | None = None,
) -> None:
    """Persist a generated implementation file and attach artifact references."""
    script_path = implementation.get("script_path")
    if not isinstance(script_path, str) or not script_path:
        return
    path = Path(script_path)
    if not path.exists():
        return

    try:
        record = get_artifact_store().store_file(
            path,
            artifact_type="implementation",
            profile_name=profile.get("name", ""),
            proposal_name=implementation.get("proposal_name", ""),
            experiment_id=implementation.get("experiment_id", ""),
            artifact_name=path.name,
            metadata={
                "class_name": implementation.get("class_name", ""),
                "proposal_name": implementation.get("proposal_name", ""),
                "cached": bool(implementation.get("cached")),
                "validated": implementation.get("validated"),
            },
            tags=["implementation", profile.get("name", "")],
        )
        implementation["stored_artifact_id"] = record["artifact_id"]
        implementation["stored_artifact_uri"] = record["uri"]
    except Exception as exc:
        log.warning(
            "artifact_refs | implementation artifact storage failed for %s: %s",
            implementation.get("class_name"),
            exc,
        )
        if errors is not None:
            errors.append(
                f"artifact_store: implementation {implementation.get('class_name', 'unknown')} failed: {exc}"
            )


def register_validation_test_artifact(
    profile: dict[str, Any],
    *,
    proposal_name: str,
    class_name: str,
    test_file: str,
    test_source: str,
    errors: list[str] | None = None,
) -> dict[str, Any] | None:
    """Persist a generated validation test file and return its artifact record."""
    path = Path(test_file)
    if not path.exists():
        return None

    try:
        return get_artifact_store().store_file(
            path,
            artifact_type="validation_test",
            profile_name=profile.get("name", ""),
            proposal_name=proposal_name,
            artifact_name=path.name,
            metadata={
                "class_name": class_name,
                "test_source": test_source,
            },
            tags=["validation", "test", profile.get("name", "")],
        )
    except Exception as exc:
        log.warning("artifact_refs | validation test artifact storage failed for %s: %s", class_name, exc)
        if errors is not None:
            errors.append(f"artifact_store: validation_test {class_name} failed: {exc}")
        return None


def register_validation_result_artifact(
    profile: dict[str, Any],
    validation_result: dict[str, Any],
    errors: list[str] | None = None,
) -> dict[str, Any] | None:
    """Persist validation output as JSON and return its artifact record."""
    class_name = str(validation_result.get("class_name") or "unknown")
    proposal_name = str(validation_result.get("proposal_name") or class_name)
    artifact_name = f"{class_name}_validation.json"
    payload = {
        "validation_result": validation_result,
    }
    try:
        return get_artifact_store().store_json(
            payload,
            artifact_type="validation_result",
            profile_name=profile.get("name", ""),
            proposal_name=proposal_name,
            experiment_id=validation_result.get("experiment_id", ""),
            artifact_name=artifact_name,
            metadata={
                "class_name": class_name,
                "passed": validation_result.get("passed"),
                "attempts": validation_result.get("attempts", 0),
                "test_source": validation_result.get("test_source", ""),
                "test_file_artifact_id": validation_result.get("test_file_artifact_id", ""),
                "test_file_artifact_uri": validation_result.get("test_file_artifact_uri", ""),
            },
            tags=["validation", "result", profile.get("name", "")],
        )
    except Exception as exc:
        log.warning("artifact_refs | validation result artifact storage failed for %s: %s", class_name, exc)
        if errors is not None:
            errors.append(f"artifact_store: validation_result {class_name} failed: {exc}")
        return None
