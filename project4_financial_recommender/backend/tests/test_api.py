"""API contract tests via FastAPI's TestClient, with all writes redirected to a temp directory."""
import importlib

import pytest
from fastapi.testclient import TestClient

BASE = "/v1/financial-products/recommendations"


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


def test_known_user_contract(client):
    """A dataset user gets ranked, explained results with request, model and rules versions."""
    body = client.get(f"{BASE}/F00001", params={"limit": 3}).json()
    assert body["strategy"] == "eligibility_then_ranking"
    assert body["request_id"].startswith("req-") and body["model_version"]
    assert len(body["recommendations"]) == 3
    first = body["recommendations"][0]
    assert {"product_id", "name", "score", "reasons", "cautions"} <= first.keys()
    assert all({"text", "evidence"} <= reason.keys() for reason in first["reasons"])
    assert body["eligibility"]["rules_version"]


def test_unknown_user_gets_eligible_cold_start(client):
    """Unknown users fall back to popularity but only see no-minimum-income products."""
    body = client.get(f"{BASE}/NOBODY").json()
    assert body["strategy"] == "cold_start_popular"
    assert all(product["min_income"] == 0 for product in body["recommendations"])


def test_what_if_overrides(client):
    """Query-string overrides change the profile and eligibility; invalid values give 422."""
    body = client.get(f"{BASE}/F00001", params={"monthly_income": 20_000, "risk_profile": "low"}).json()
    assert body["profile_source"] == "dataset+what_if"
    assert body["profile"]["monthly_income"] == 20_000
    failed = {rule["rule_id"] for product in body["eligibility"]["ineligible"] for rule in product["failed_rules"]}
    assert failed == {"min_income", "risk_suitability"}
    assert client.get(f"{BASE}/F00001", params={"risk_profile": "reckless"}).status_code == 422


def test_pairwise_eligibility_endpoint(client):
    """A low-risk user fails risk_suitability for high-risk FP015; unknown products give 404."""
    low_risk = client.get("/v1/users", params={"search": "low", "limit": 1}).json()["users"][0]
    body = client.get(f"/v1/eligibility/{low_risk['user_id']}/FP015").json()
    assert body["eligible"] is False
    assert body["failed_rules"][0]["rule_id"] == "risk_suitability"
    assert client.get(f"/v1/eligibility/{low_risk['user_id']}/NOPE").status_code == 404


def test_feedback_and_retrain(client):
    """Apply becomes training data and shows as "new"; clicks do not; retraining clears the backlog."""
    response = client.post("/v1/financial-products/events", json={
        "user_id": "F00001", "product_id": "FP001", "event_type": "apply", "request_id": "req-test",
    }).json()
    assert response["becomes_training_data"] and response["pending_new_interactions"] == 1
    assert response["event"]["eligible_at_event_time"] is True
    assert client.get("/v1/users/F00001").json()["history"][0]["after_training"] is True

    click = client.post("/v1/financial-products/events", json={
        "user_id": "F00001", "product_id": "FP001", "event_type": "recommendation_click",
    }).json()
    assert click["becomes_training_data"] is False

    retrained = client.post("/v1/model/retrain").json()
    assert retrained["pending_new_interactions"] == 0 and "evaluation" in retrained
