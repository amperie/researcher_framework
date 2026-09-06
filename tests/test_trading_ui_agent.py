from core.plugins.trading.ui_agent import create_component_draft, create_idea_batch


def test_create_idea_batch_returns_reviewable_ideas():
    batch = create_idea_batch(
        {
            "direction": "SPY mean reversion with volatility filters",
            "symbols": ["SPY", "VIX"],
            "maxIdeas": 2,
        }
    )

    assert batch["status"] == "succeeded"
    assert batch["direction"] == "SPY mean reversion with volatility filters"
    assert len(batch["ideas"]) == 2
    assert batch["ideas"][0]["proposedSymbols"] == ["SPY", "VIX"]
    assert batch["ideas"][0]["name"] == "volatility_filtered_mean_reversion"


def test_create_component_draft_returns_valid_component_source_shape():
    batch = create_idea_batch({"direction": "SPY mean reversion", "maxIdeas": 1})
    draft = create_component_draft(
        {
            "direction": batch["direction"],
            "ideaBatchId": batch["id"],
            "idea": batch["ideas"][0],
            "generationMode": "algorithm-and-portfolio",
        }
    )

    assert draft["status"] == "succeeded"
    assert draft["algorithm"]["content"].count("crucible_metadata") == 1
    assert "class VolatilityFilteredMeanReversionAlgorithm(Algorithm)" in draft["algorithm"]["content"]
    assert draft["portfolio"]["content"].count("crucible_metadata") == 1
