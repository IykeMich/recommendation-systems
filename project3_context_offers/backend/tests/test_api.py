"""API tests via FastAPI's TestClient, with all writes redirected to a temporary directory."""
import importlib

import pytest
from fastapi.testclient import TestClient

CONTEXT = {"device_type": "mobile", "region": "Lagos", "hour": 9, "day_of_week": 1}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API client whose new interactions, feedback log and artifacts go to tmp_path."""
    import src.config
    import src.data
    import src.train

    monkeypatch.setattr(src.data, "NEW_INTERACTIONS_PATH", tmp_path / "new.csv")
    monkeypatch.setattr(src.config, "FEEDBACK_LOG_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(src.config, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(src.train, "ARTIFACT_DIR", tmp_path / "artifacts")

    import api.main

    return TestClient(importlib.reload(api.main).app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_known_user_gets_contextual_ranking(client):
    body = client.get("/v1/offers/recommendations/R00001", params={**CONTEXT, "limit": 5}).json()
    assert body["strategy"] == "contextual_ranker"
    assert body["context"]["daypart"] == "morning"
    assert len(body["recommendations"]) == 5
    scores = [offer["score"] for offer in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(offer["region"] in ("Lagos", "all") for offer in body["recommendations"])
    assert all(offer["channel"] in ("mobile", "both") for offer in body["recommendations"])
    assert body["recommendations"][0]["reasons"]


def test_unknown_user_gets_cold_start(client):
    body = client.get("/v1/offers/recommendations/NOBODY", params=CONTEXT).json()
    assert body["strategy"] == "cold_start_popular"
    assert body["recommendations"]


def test_region_changes_candidates(client):
    lagos = client.get("/v1/offers/recommendations/R00001", params=CONTEXT).json()
    abuja = client.get("/v1/offers/recommendations/R00001", params={**CONTEXT, "region": "Abuja"}).json()
    assert {offer["offer_id"] for offer in lagos["recommendations"]} != {
        offer["offer_id"] for offer in abuja["recommendations"]
    }


def test_redeemed_offer_is_removed_immediately(client):
    """Redeeming takes effect without retraining and shows up as a post-training history event."""
    first = client.get("/v1/offers/recommendations/R00001", params=CONTEXT).json()["recommendations"][0]
    response = client.post("/v1/offers/events", json={
        "user_id": "R00001", "offer_id": first["offer_id"], "event_type": "redeem", "context": CONTEXT,
    })
    assert response.json()["pending_new_interactions"] == 1

    after = client.get("/v1/offers/recommendations/R00001", params=CONTEXT).json()
    assert first["offer_id"] not in {offer["offer_id"] for offer in after["recommendations"]}
    assert client.get("/v1/users/R00001").json()["history"][0]["after_training"] is True


def test_retrain_clears_pending(client):
    """Retraining absorbs pending feedback and keeps the previous evaluation in the metadata."""
    client.post("/v1/offers/events", json={
        "user_id": "R00001", "offer_id": "O00001", "event_type": "click", "context": CONTEXT,
    })
    retrained = client.post("/v1/model/retrain").json()
    assert retrained["pending_new_interactions"] == 0
    assert "evaluation" in retrained


def test_invalid_context_is_rejected(client):
    assert client.get("/v1/offers/recommendations/R00001", params={**CONTEXT, "hour": 30}).status_code == 422
    assert client.post("/v1/offers/events", json={
        "user_id": "R00001", "offer_id": "NOPE", "event_type": "click", "context": CONTEXT,
    }).status_code == 404
