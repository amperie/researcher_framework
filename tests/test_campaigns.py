from __future__ import annotations

from core.campaigns import materialize_variants


def test_materialize_variants_applies_defaults_and_variant_patch():
    base_proposal = {
        "name": "per_projection_residual_entropy",
        "dataset": "HaluBench",
        "hyperparameters": {
            "max_features": 64,
            "regularization": "light",
        },
    }
    cfg = {
        "research_direction": "claim-ratio feature sweep",
        "variant_defaults": {
            "planning_notes": "Keep the family comparable.",
            "proposal_patch": {
                "hyperparameters": {
                    "regularization": "medium",
                }
            },
        },
        "variants": [
            {
                "key": "small-q",
                "title": "Small Q-only",
                "name_suffix": "small-q",
                "planning_notes": "Use the smallest feature budget.",
                "proposal_patch": {
                    "hyperparameters": {
                        "max_features": 16,
                    },
                    "projection_subset": "q",
                },
            }
        ],
    }

    variants = materialize_variants(base_proposal, cfg)

    assert len(variants) == 1
    variant = variants[0]
    assert variant["key"] == "small-q"
    assert variant["research_direction"] == "claim-ratio feature sweep"
    assert variant["proposal_template"]["name"] == "per_projection_residual_entropy_small_q"
    assert variant["proposal_template"]["hyperparameters"]["max_features"] == 16
    assert variant["proposal_template"]["hyperparameters"]["regularization"] == "medium"
    assert "Keep the family comparable." in variant["planning_notes"]
    assert "smallest feature budget" in variant["planning_notes"]
