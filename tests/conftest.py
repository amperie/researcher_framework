"""Shared pytest fixtures for the researcher_framework test suite."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal profile fixture (no file I/O needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_profile() -> dict:
    """A minimal valid research profile dict usable by all nodes."""
    return {
        "name": "test_profile",
        "description": "Test profile",
        "pipeline": {
            "steps": ["research", "ideate"],
        },
        "llm": {
            "default_model": "claude-test",
            "step_overrides": {
                "research": "claude-fast",
            },
        },
        "research": {
            "tools": [
                {
                    "name": "arxiv",
                    "tool": "tools.research_tools.collect_arxiv",
                    "max_results": 5,
                    "relevance_score_threshold": 6,
                    "max_papers_to_digest": 1,
                }
            ],
            "domain_context": "Testing domain context",
            "max_summary_artifacts": 5,
        },
        "base_classes": [
            {
                "name": "TestBase",
                "module": "test.base",
                "description": "A test base class",
                "key_interface": "class TestBase:\n    def run(self): ...",
            }
        ],
        "datasets": [
            {
                "name": "test_dataset",
                "description": "A test dataset",
                "storage": {
                    "type": "mongodb",
                    "application_name": "test_db",
                    "sub_application_name": "test_col",
                },
                "available_detectors": ["hallucination"],
                "available_scan_fields": {
                    "guaranteed": ["field_a", "field_b"],
                    "optional": ["field_c"],
                    "not_available": ["field_x"],
                },
                "layer_name_patterns": {
                    "ffn": ["mlp"],
                    "attn": ["attn"],
                },
            }
        ],
        "experiment_adapter": "plugins.test_adapter",
        "evaluation": {
            "primary_metric": "test_auc",
            "metrics": ["test_auc", "test_f1"],
            "thresholds": {"test_auc": 0.65},
        },
        "storage": {
            "mlflow_experiment": "test_experiment",
            "mongodb_results_db": "test_results",
            "chroma_collection": "test_collection",
        },
        "validate": {
            "auto_run": True,
            "max_fix_retries": 2,
            "test_runner": "uv run pytest",
            "test_output_dir": "dev/experiments/test_profile/tests",
        },
        "prompts": {
            "research": {
                "system": "You are a researcher.",
                "score_system": "Score this artifact.",
                "artifact_score_system": "Score this artifact.",
                "summary_system": "Summarise these artifacts.",
                "artifact_summary_system": "Summarise these artifacts.",
                "digest_system": "Digest this paper.",
            },
            "ideate": {"system": "Brainstorm ideas."},
            "refine": {"system": "Refine ideas."},
            "propose_experiments": {"system": "Propose experiments."},
            "plan_implementation": {"system": "Plan implementation."},
            "implement": {"system": "Implement the plan."},
            "validate": {
                "system": "Write tests.",
                "fix_system": "Fix the code.",
            },
            "evaluate": {"system": "Evaluate results."},
            "store_results": {"system": "Store results."},
            "propose_next_steps": {"system": "Propose next steps."},
        },
    }


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """A mock LLM that returns a configurable response."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='[{"name": "idea1", "description": "desc"}]')
    return llm


def make_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with the given content."""
    resp = MagicMock()
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Mock config
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider="anthropic",
        llm_model=None,
        anthropic_api_key="test-key",
        openai_api_key=None,
        mongo_url="mongodb://localhost:27017",
        mlflow_uri="http://localhost:5000",
        chroma_host="localhost",
        chroma_port=8000,
        chroma_ssl=False,
        chroma_auth_token=None,
        chroma_collection="test_collection",
        experiment_timeout_seconds=300,
        validate_timeout_seconds=60,
        max_arxiv_papers=20,
        experiments_dir="dev/experiments",
        artifacts_db_name="researcher_artifacts",
        artifacts_collection="artifacts",
        artifact_store_backend="filesystem",
        artifact_store_root="dev/artifacts",
        s3_endpoint_url=None,
        s3_access_key_id=None,
        s3_secret_access_key=None,
        s3_bucket=None,
        s3_region_name="us-east-1",
        s3_prefix=None,
        s3_secure=True,
    )


@pytest.fixture(autouse=False)
def patch_get_config(mock_cfg):
    """Patch get_config() globally with mock_cfg. Use explicitly where needed."""
    with patch("configs.config.get_config", return_value=mock_cfg):
        yield mock_cfg
