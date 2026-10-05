"""FastAPI endpoint tests via TestClient (needs the trained artifact in artifacts/)."""
import importlib

import pytest
from fastapi.testclient import TestClient

VALID = {
    "transaction_id": "TXN-DEMO-001",
    "customer_id": "F00001",
    "amount": 1500.0,
    "currency": "NGN",
    "country": "NG",
    "device": "android",
    "merchant_category": "travel",
    "hour": 3,
    "velocity_1h": 9,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Uses the trained artifact, but a throwaway decision store and policy file."""
    from src import config

    monkeypatch.setattr(config, "DECISION_DB_PATH", tmp_path / "decisions.sqlite3")
    monkeypatch.setattr(config, "POLICY_PATH", tmp_path / "policy.json")
    import api.main

    return TestClient(importlib.reload(api.main).app)


def test_health_endpoint(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"].startswith("fraud-")


def test_score_endpoint_returns_decision(client):
    body = client.post("/score", json=VALID).json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert body["model_version"] and body["policy_version"] == "policy-v1"
    assert body["reasons"] and body["idempotent_replay"] is False


def test_suspicious_pattern_scores_higher_than_daytime(client):
    risky = client.post("/score", json=VALID).json()["risk_score"]
    daytime = client.post("/score", json={**VALID, "transaction_id": "TXN-DAY", "hour": 14, "velocity_1h": 2}).json()["risk_score"]
    assert risky > 0.5 > daytime


def test_duplicate_transaction_is_idempotent(client):
    first = client.post("/score", json=VALID).json()
    second = client.post("/score", json=VALID).json()
    assert second["idempotent_replay"] is True
    assert {key: second[key] for key in ("risk_score", "decision")} == {key: first[key] for key in ("risk_score", "decision")}
    assert client.get("/v1/stats").json()["scored"] == 1
    conflict = client.post("/score", json={**VALID, "amount": 99.0})
    assert conflict.status_code == 409


@pytest.mark.parametrize("change", [
    {"amount": 0}, {"amount": -5}, {"hour": 24}, {"velocity_1h": 0}, {"currency": "naira"},
    {"risk_score": 0.01},                      # client must not supply a score
    {"hour": None},                            # no hour and no timestamp
])
def test_malformed_records_are_rejected(client, change):
    payload = {**VALID, **change}
    payload = {key: value for key, value in payload.items() if value is not None}
    assert client.post("/score", json=payload).status_code == 422
    assert client.get("/v1/stats").json()["rejected_requests"] == 1


def test_unknown_categories_do_not_crash(client):
    body = client.post("/score", json={**VALID, "transaction_id": "TXN-X", "device": "smart-fridge",
                                       "country": "ZZ", "merchant_category": "crypto", "customer_id": "NEVER-SEEN"}).json()
    assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert "Customer has no reference history" in body["context"]


def test_velocity_computed_server_side_from_timestamps(client):
    """Three timestamped transactions 20 minutes apart: the third sees the two earlier ones (+ itself)."""
    base = {key: value for key, value in VALID.items() if key not in ("hour", "velocity_1h")}
    for index, minute in enumerate(["00", "20", "40"]):
        body = client.post("/score", json={**base, "transaction_id": f"TXN-V{index}",
                                           "transaction_timestamp": f"2026-09-30T02:{minute}:00Z"}).json()
    assert body["features"]["velocity_1h"] == 3 and body["features"]["velocity_source"] == "decision_store"
    assert body["features"]["hour"] == 2


def test_policy_change_affects_new_decisions_only(client):
    """Same model score, new thresholds: only the new decision changes; the old keeps policy-v1."""
    before = client.post("/score", json=VALID).json()
    updated = client.put("/v1/policy", json={"review_threshold": 0.2, "block_threshold": 0.5}).json()
    assert updated["version"] == "policy-v2"
    after = client.post("/score", json={**VALID, "transaction_id": "TXN-DEMO-002"}).json()
    assert before["risk_score"] == pytest.approx(after["risk_score"])   # same model
    assert after["decision"] == "BLOCK" and after["policy_version"] == "policy-v2"
    assert client.get(f"/v1/decisions/{VALID['transaction_id']}").json()["policy_version"] == "policy-v1"
    assert client.put("/v1/policy", json={"review_threshold": 0.9, "block_threshold": 0.5}).status_code == 422


def test_replay_outcome_and_detail(client):
    """Replaying the same test rows twice is fully idempotent; outcomes and detail view work."""
    replay = client.post("/v1/replay", json={"count": 50}).json()
    assert replay["processed"] == 50 and sum(replay["decisions"].values()) == 50
    again = client.post("/v1/replay", json={"count": 50}).json()
    assert again["duplicates_ignored"] == 50
    first_id = client.get("/v1/decisions", params={"limit": 1}).json()["decisions"][0]["transaction_id"]
    assert client.post(f"/v1/decisions/{first_id}/outcome", json={"outcome": "legitimate"}).json()["outcome"] == "legitimate"
    detail = client.get(f"/v1/decisions/{first_id}").json()
    assert detail["customer_profile"]["customer_id"] == detail["customer_id"]


def test_policy_simulation(client):
    body = client.get("/v1/policy/simulate", params={"review_threshold": 0.3, "block_threshold": 0.9}).json()
    assert sum(body["decision_mix"].values()) == pytest.approx(1.0)
    assert 0 <= body["flagged"]["recall"] <= 1
