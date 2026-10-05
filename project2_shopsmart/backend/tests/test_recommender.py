"""Unit tests for preprocessing, the item-item model, the popularity fallback and the holdout split."""
import pandas as pd
import pytest

from src.evaluate import holdout_split
from src.preprocess import build_user_item, prepare_events
from src.recommender import (
    AdaptiveHybridRecommender,
    ItemItemRecommender,
    build_content_similarity,
    content_share,
    popular_items,
)


@pytest.fixture
def toy_events():
    # u1 and u2 both like A and B; u2 also bought C, so C should be recommended to u1.
    return pd.DataFrame(
        [
            ("u1", "A", "purchase", 1),
            ("u1", "B", "view", 2),
            ("u2", "A", "purchase", 1),
            ("u2", "B", "cart", 2),
            ("u2", "C", "purchase", 3),
            ("u3", "D", "view", 1),
        ],
        columns=["user_id", "item_id", "event_type", "timestamp"],
    )


def test_event_weights_follow_event_type(toy_events):
    weighted = prepare_events(toy_events)
    assert weighted.set_index("event_type")["weight"].to_dict() == {
        "purchase": 6.0,
        "view": 1.0,
        "cart": 3.0,
    }


def test_unknown_event_type_is_rejected(toy_events):
    toy_events.loc[0, "event_type"] = "wishlist"
    with pytest.raises(ValueError, match="Unknown event types"):
        prepare_events(toy_events)


def test_repeated_events_are_aggregated(toy_events):
    """Two u1 views of B (weight 1 each) sum to 2 in the matrix."""
    repeated = pd.concat([toy_events, toy_events.iloc[[1]]], ignore_index=True)
    matrix = build_user_item(prepare_events(repeated))
    assert matrix.loc["u1", "B"] == 2.0


def test_recommends_unseen_co_interacted_item(toy_events):
    model = ItemItemRecommender().fit(build_user_item(prepare_events(toy_events)))
    recommendations = model.recommend("u1", k=5, explain=True)
    recommended_ids = [recommendation["item_id"] for recommendation in recommendations]
    assert recommended_ids[0] == "C"
    assert "A" not in recommended_ids and "B" not in recommended_ids
    assert "D" not in recommended_ids  # no collaborative evidence linking D to u1
    assert set(recommendations[0]["because_item_ids"]) <= {"A", "B"}


def test_unknown_user_gets_no_personal_recommendations(toy_events):
    model = ItemItemRecommender().fit(build_user_item(prepare_events(toy_events)))
    assert model.recommend("nobody") == []


def test_popular_items_ranks_by_total_weight(toy_events):
    """A has two purchases (12), C one (6); B only 4 (view + cart)."""
    popular = popular_items(prepare_events(toy_events), k=2)
    assert [item["item_id"] for item in popular] == ["A", "C"]


def test_holdout_hides_all_events_for_the_test_pair(toy_events):
    """Only u2 has two positive items; their latest (C) is hidden and removed from training."""
    train_events, test_events = holdout_split(prepare_events(toy_events))
    assert list(zip(test_events.user_id, test_events.item_id)) == [("u2", "C")]
    assert not ((train_events.user_id == "u2") & (train_events.item_id == "C")).any()


@pytest.fixture
def toy_products():
    """A and B are snacks, C is another snack brand, D is a laptop, E is a never-interacted snack."""
    return pd.DataFrame(
        [
            ("A", "Crisps - Small Pack", "Food", "Snacks", "Crunchy", "Salted potato crisps."),
            ("B", "Crisps - Value Pack", "Food", "Snacks", "Crunchy", "Salted potato crisps."),
            ("C", "Corn Chips", "Food", "Snacks", "Nacho", "Cheese corn chips."),
            ("D", "Laptop Pro", "Electronics", "Laptops", "Techco", "Powerful laptop for work."),
            ("E", "Cheese Puffs", "Food", "Snacks", "Puffy", "Cheese puffed corn snack."),
        ],
        columns=["item_id", "name", "category", "subcategory", "brand", "description"],
    )


def test_content_similarity_prefers_same_subcategory(toy_products):
    similarity = build_content_similarity(toy_products)
    # Row A: the other snacks (B, C, E) all score above the laptop (D); nothing is similar to itself.
    assert min(similarity[0, [1, 2, 4]]) > similarity[0, 3]
    assert similarity[0, 0] == 0


def test_content_share_shrinks_as_history_grows():
    assert content_share(1) == 0.8
    assert content_share(10) == 0.5
    assert content_share(100) == 0.2


def test_hybrid_reaches_never_interacted_items_and_explains(toy_events, toy_products):
    """E has no interactions, so item-item can never recommend it, but content similarity can."""
    model = AdaptiveHybridRecommender(toy_products).fit(build_user_item(prepare_events(toy_events)))
    recommendations = model.recommend("u1", k=5, explain=True)
    recommended_ids = [r["item_id"] for r in recommendations]
    assert "E" in recommended_ids
    assert "A" not in recommended_ids and "B" not in recommended_ids
    by_id = {r["item_id"]: r for r in recommendations}
    assert by_id["E"]["source"] == "similar_products"
    assert by_id["E"]["reasons"][0] == {
        "type": "similar_product", "item_id": by_id["E"]["reasons"][0]["item_id"],
        "shared": "subcategory", "value": "Snacks",
    }


def test_shoppers_in_common_counts_co_interactions(toy_events, toy_products):
    """u1 and u2 both touched A and B; only u2 touched C."""
    model = AdaptiveHybridRecommender(toy_products).fit(build_user_item(prepare_events(toy_events)))
    assert model.shoppers_in_common("A", "B") == 2
    assert model.shoppers_in_common("A", "C") == 1
    assert model.shoppers_in_common("A", "E") == 0  # E was never interacted with
