"""API tests: run the FastAPI app in-process with all file writes redirected to a temp directory."""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API client whose new interactions, feedback log and artifacts go to tmp_path."""
    import src.config
    import src.data
    import src.train

    # Patch the module attributes api.main reads, then reload api.main so it picks them up
    # (it trains a fresh model into tmp_path because no artifacts exist there yet).
    monkeypatch.setattr(src.data, "NEW_INTERACTIONS_PATH", tmp_path / "new_interactions.csv")
    monkeypatch.setattr(src.config, "FEEDBACK_LOG_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(src.config, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(src.train, "ARTIFACT_DIR", tmp_path / "artifacts")

    import api.main

    api_module = importlib.reload(api.main)
    return TestClient(api_module.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_known_user_gets_hybrid_recommendations(client):
    """A known user gets the full enriched response, and nothing from their history is recommended."""
    response = client.get("/v1/recommendations/user/R00001?limit=5")
    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "adaptive_hybrid"
    assert len(body["recommendations"]) == 5
    first = body["recommendations"][0]
    assert set(first) == {
        "item_id", "score", "name", "category", "subcategory", "brand", "price", "source", "reasons"
    }
    assert first["source"] in {"similar_products", "shoppers_like_you"}
    # Every reason names one of the shopper's own products and their action on it.
    for reason in first["reasons"]:
        assert reason["name"] and reason["your_event"] in {"view", "cart", "purchase"}

    seen_item_ids = {event["item_id"] for event in client.get("/v1/users/R00001?history_limit=100").json()["history"]}
    assert not seen_item_ids & {recommendation["item_id"] for recommendation in body["recommendations"]}


def test_unknown_user_gets_popular_fallback(client):
    body = client.get("/v1/recommendations/user/unknown-user").json()
    assert body["strategy"] == "popular_fallback"
    assert len(body["recommendations"]) == 10


def test_new_interactions_make_a_new_user_known_after_retrain(client):
    """Events stay pending (fallback still used) until retraining makes the user known to the model."""
    for item_id in ["P00100", "P00117"]:
        response = client.post(
            "/v1/events",
            json={"user_id": "NEW_SHOPPER", "item_id": item_id, "event_type": "purchase"},
        )
        assert response.json()["becomes_training_data"] is True

    assert client.get("/v1/model").json()["pending_new_interactions"] == 2
    assert client.get("/v1/recommendations/user/NEW_SHOPPER").json()["strategy"] == "popular_fallback"

    retrained = client.post("/v1/model/retrain").json()
    assert retrained["pending_new_interactions"] == 0
    assert client.get("/v1/recommendations/user/NEW_SHOPPER").json()["strategy"] == "adaptive_hybrid"
    assert client.get("/v1/users/NEW_SHOPPER").json()["known_to_model"] is True


def test_new_shopper_buying_one_category_gets_that_category(client):
    """Regression: a new shopper who only buys Food & Beverage should mostly see Food & Beverage."""
    food_items = ["P00093", "P00366", "P00382", "P00414"]  # Doritos, Pop-Tarts, Ritz, Activia
    for item_id in food_items:
        client.post("/v1/events", json={"user_id": "FOOD_FAN", "item_id": item_id, "event_type": "purchase"})
    client.post("/v1/model/retrain")

    recommendations = client.get("/v1/recommendations/user/FOOD_FAN?limit=10").json()["recommendations"]
    food_share = sum(r["category"] == "Food & Beverage" for r in recommendations) / len(recommendations)
    assert food_share >= 0.7
    assert all(r["reasons"] for r in recommendations)
    assert not {r["item_id"] for r in recommendations} & set(food_items)


def test_recommendation_click_is_logged_but_not_training_data(client):
    """Clicks go to the feedback log only, so they don't increase the pending interaction count."""
    body = client.post(
        "/v1/events",
        json={"user_id": "R00001", "item_id": "P00100", "event_type": "recommendation_click"},
    ).json()
    assert body["becomes_training_data"] is False
    assert body["pending_new_interactions"] == 0


def test_unknown_item_is_rejected(client):
    response = client.post(
        "/v1/events",
        json={"user_id": "R00001", "item_id": "NOPE", "event_type": "view"},
    )
    assert response.status_code == 404


def test_product_search_by_text_and_category(client):
    """Brand search finds Nike products; the category filter narrows; all categories are listed."""
    body = client.get("/v1/products?search=nike").json()
    assert body["total"] > 0
    assert all(product["brand"] == "Nike" or "nike" in product["name"].lower() for product in body["products"])
    assert "Food & Beverage" in body["categories"] and len(body["categories"]) >= 10

    snacks = client.get("/v1/products?search=snacks&category=Food%20%26%20Beverage").json()["products"]
    assert snacks and all(product["category"] == "Food & Beverage" for product in snacks)
    assert client.get("/v1/products?search=zzz-no-such-product").json()["total"] == 0


def test_shopper_across_two_categories_gets_both(client):
    """A new shopper who buys sneakers and malt drinks should see both categories after retraining."""
    for item_id in ["P00217", "P00229", "P00001", "P00107"]:  # Air Force 1, Samba, Milo, Maltina
        client.post("/v1/events", json={"user_id": "MIXED_FAN", "item_id": item_id, "event_type": "purchase"})
    client.post("/v1/model/retrain")

    recommendations = client.get("/v1/recommendations/user/MIXED_FAN?limit=10").json()["recommendations"]
    categories = {recommendation["category"] for recommendation in recommendations}
    assert {"Apparel & Footwear", "Food & Beverage"} <= categories
