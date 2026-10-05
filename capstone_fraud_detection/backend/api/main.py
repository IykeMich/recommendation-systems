"""Fraud Detection API.

Run from backend/: uvicorn api.main:app --reload --port 8005

The browser is an untrusted client: it sends a transaction and gets a decision.
Features, the risk score and the policy outcome are all computed server-side;
extra client fields (e.g. a "risk_score") are rejected.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import config
from src.evaluate import classification_metrics
from src.explain import explain_transaction
from src.policy import Policy, load_policy, save_policy
from src.predict import FraudScorer

from .store import DecisionStore, DuplicateTransaction, utc_now

app = FastAPI(title="Fraud Detection API", version="1.0.0")
# Allow the Next.js dashboard (a different origin/port) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3005,http://127.0.0.1:3005").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# First run convenience: train if no model artifact exists yet. Everything below is loaded
# once at import time and shared by all requests.
if not config.PIPELINE_PATH.exists():
    from src.train import train

    train()

scorer = FraudScorer.load(config.PIPELINE_PATH)
manifest = json.loads(config.MANIFEST_PATH.read_text())
policy = load_policy(config.POLICY_PATH)
store = DecisionStore(config.DECISION_DB_PATH)


class Transaction(BaseModel):
    """Guide contract, adapted to the dataset: the feature inputs the model needs."""

    # Unknown fields (e.g. a client-supplied risk_score) cause a 422 instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64)
    customer_id: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=100_000_000)
    currency: str = Field(default="NGN", pattern=r"^[A-Z]{3}$")
    country: str = Field(min_length=2, max_length=2, description="ISO country code, e.g. NG")
    device: str = Field(min_length=1, max_length=32, description="android / ios / web; unknown values are allowed")
    merchant_category: str = Field(min_length=1, max_length=32)
    transaction_timestamp: Optional[datetime] = None
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    velocity_1h: Optional[int] = Field(
        default=None, ge=1, le=1000,
        description="Transactions by this customer in the last hour, including this one. "
                    "Computed from the decision store when omitted (needs transaction_timestamp).",
    )

    @model_validator(mode="after")
    def time_context_is_available(self):
        """Runs after field validation: hour and velocity must each be given or derivable from a timestamp."""
        if self.hour is None and self.transaction_timestamp is None:
            raise ValueError("Provide hour or transaction_timestamp")
        if self.velocity_1h is None and self.transaction_timestamp is None:
            raise ValueError("Provide velocity_1h or transaction_timestamp")
        return self


@app.exception_handler(RequestValidationError)
async def record_rejections(request: Request, validation_error: RequestValidationError):
    """Malformed records are rejected at the boundary and counted for the data-quality panel."""
    # loc looks like ("body", "amount"); dropping the first part leaves the field path.
    # Only /score rejections are counted, so bad policy/replay calls do not skew the panel.
    errors = [{"field": ".".join(str(part) for part in error["loc"][1:]), "message": error["msg"]} for error in validation_error.errors()]
    if request.url.path == "/score":
        store.record_rejection(errors)
    return JSONResponse(status_code=422, content={"detail": errors})


def payload_hash(transaction: Transaction) -> str:
    """SHA-256 of the validated payload; tells a true retry (same hash) from a conflicting reuse of an ID."""
    return hashlib.sha256(transaction.model_dump_json().encode()).hexdigest()


def resolve_time_context(transaction: Transaction) -> dict:
    """Work out hour and velocity_1h, using client values when given, else the timestamp/store.

    Timestamps are normalised to UTC (naive ones are assumed UTC) so stored strings compare correctly.
    """
    if transaction.transaction_timestamp is None:
        return {"hour": transaction.hour, "velocity_1h": transaction.velocity_1h, "velocity_source": "client", "timestamp_iso": None}
    timestamp = transaction.transaction_timestamp
    timestamp = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    velocity, velocity_source = transaction.velocity_1h, "client"
    if velocity is None:
        # Server-side velocity: this customer's scored transactions in the previous hour, plus this one.
        # Window is [timestamp - 1h, timestamp): only earlier stored rows match, so +1 adds this one.
        velocity = 1 + store.recent_customer_count(
            transaction.customer_id,
            (timestamp - timedelta(hours=1)).isoformat(timespec="seconds"),
            timestamp.isoformat(timespec="seconds"),
        )
        velocity_source = "decision_store"
    return {
        "hour": transaction.hour if transaction.hour is not None else timestamp.hour,
        "velocity_1h": velocity,
        "velocity_source": velocity_source,
        "timestamp_iso": timestamp.isoformat(timespec="seconds"),
    }


def response_from(record: dict, replay: bool) -> dict:
    """Shape a stored decision record into the /score response; `replay` marks an idempotent repeat."""
    return {
        "transaction_id": record["transaction_id"],
        "risk_score": round(record["risk_score"], 4),
        "decision": record["decision"],
        "model_version": record["model_version"],
        "policy_version": record["policy_version"],
        "reasons": [reason["text"] for reason in record["reasons"]],
        "reason_evidence": record["reasons"],
        "context": record["context"],
        "features": record["features"],
        "idempotent_replay": replay,
    }


def score_transaction(transaction: Transaction, source: str = "api", synthetic_label: Optional[int] = None) -> dict:
    """Scoring path shared by /score and /v1/replay: idempotency check, features,
    score, policy decision, explanation, then store.
    """
    # Fast idempotency path: a retry returns the stored decision instead of re-scoring.
    # (A concurrent duplicate that slips past this check is caught by store.insert.)
    existing = store.get(transaction.transaction_id)
    if existing:
        if existing["payload_hash"] != payload_hash(transaction):
            raise DuplicateTransaction(transaction.transaction_id)
        return response_from(existing, replay=True)

    time_context = resolve_time_context(transaction)
    row = {
        "transaction_id": transaction.transaction_id,
        "customer_id": transaction.customer_id,
        "amount": transaction.amount,
        "currency": transaction.currency,
        "hour": time_context["hour"],
        "velocity_1h": time_context["velocity_1h"],
        # Normalise case to match the training data's categories.
        "country": transaction.country.upper(),
        "device": transaction.device.lower(),
        "merchant_category": transaction.merchant_category.lower(),
    }
    risk_score, features = scorer.score(row)
    features["velocity_source"] = time_context["velocity_source"]
    explanation = explain_transaction({**row, **features}, scorer.artifact["explanation_reference"])
    stored, replay = store.insert({
        "transaction_id": transaction.transaction_id,
        "payload_hash": payload_hash(transaction),
        "payload": json.loads(transaction.model_dump_json()),
        "customer_id": transaction.customer_id,
        "amount": transaction.amount,
        "risk_score": risk_score,
        "decision": policy.decide(risk_score),
        "reasons": explanation["reasons"],
        "context": explanation["context"],
        "features": features,
        "model_version": scorer.model_version,
        "policy_version": policy.version,
        "transaction_timestamp": time_context["timestamp_iso"],
        "created_at": utc_now(),
        "source": source,
        "synthetic_label": synthetic_label,
    })
    return response_from(stored, replay)


@app.get("/health")
def health():
    """Liveness check that also reports the loaded model and policy versions."""
    return {"status": "ok", "model_version": scorer.model_version, "policy_version": policy.version}


@app.post("/score")
def score(transaction: Transaction):
    """Score one transaction; 409 if its transaction_id was already used with a different payload."""
    try:
        return score_transaction(transaction)
    except DuplicateTransaction:
        raise HTTPException(409, f"transaction_id {transaction.transaction_id} was already scored with a different payload")


class ReplayRequest(BaseModel):
    """Which slice of data/processed/test.csv to replay."""
    count: int = Field(default=200, ge=1, le=3000)
    offset: int = Field(default=0, ge=0)


@app.post("/v1/replay")
def replay_test_transactions(replay_request: ReplayRequest):
    """Simulate a stream: push held-out test transactions through the same /score path.
    Their synthetic labels are stored separately, as if outcomes arrived later."""
    test_rows = pd.read_csv(config.PROCESSED_DIR / "test.csv").iloc[replay_request.offset: replay_request.offset + replay_request.count]
    decisions = {"ALLOW": 0, "REVIEW": 0, "BLOCK": 0}
    replays = 0
    for row in test_rows.to_dict("records"):
        result = score_transaction(Transaction(
            transaction_id=row["transaction_id"], customer_id=row["customer_id"], amount=row["amount"],
            currency=row["currency"], country=row["country"], device=row["device"],
            merchant_category=row["merchant_category"], hour=int(row["hour"]), velocity_1h=int(row["velocity_1h"]),
        ), source="replay", synthetic_label=int(row["is_fraud"]))
        decisions[result["decision"]] += 1
        replays += result["idempotent_replay"]
    return {"processed": len(test_rows), "duplicates_ignored": replays, "decisions": decisions,
            "next_offset": replay_request.offset + len(test_rows)}


@app.get("/v1/decisions")
def list_decisions(decision: Optional[Literal["ALLOW", "REVIEW", "BLOCK"]] = None, limit: int = Query(default=50, ge=1, le=500)):
    """Recent decisions (summary fields only) for the dashboard, optionally filtered by decision."""
    return {"decisions": [
        {key: record[key] for key in ("transaction_id", "customer_id", "amount", "risk_score", "decision", "created_at",
                                      "transaction_timestamp", "source", "outcome", "synthetic_label")}
        | {"reasons": [reason["text"] for reason in record["reasons"]]}
        for record in store.list(decision, limit)
    ]}


@app.get("/v1/decisions/{transaction_id}")
def decision_detail(transaction_id: str):
    """Full stored decision plus the customer's training-profile summary for the alert drawer."""
    record = store.get(transaction_id)
    if record is None:
        raise HTTPException(404, "Unknown transaction_id")
    return {**record, "customer_profile": scorer.profiles.profile(record["customer_id"])}


class OutcomeIn(BaseModel):
    """Analyst verdict on an alert, with an optional note."""
    outcome: Literal["confirmed_fraud", "legitimate"]
    note: Optional[str] = Field(default=None, max_length=500)


@app.post("/v1/decisions/{transaction_id}/outcome")
def record_outcome(transaction_id: str, outcome: OutcomeIn):
    """An alert is a signal, not confirmed fraud: the analyst's outcome is recorded separately."""
    record = store.set_outcome(transaction_id, outcome.outcome, outcome.note)
    if record is None:
        raise HTTPException(404, "Unknown transaction_id")
    return {"transaction_id": transaction_id, "outcome": record["outcome"], "outcome_at": record["outcome_at"]}


@app.get("/v1/stats")
def stats():
    """Live totals from the decision store for the dashboard."""
    return store.stats()


@app.get("/v1/model")
def model_info():
    """Training lineage, metrics, threshold table and data-quality report from artifacts/."""
    metrics = json.loads(config.METRICS_PATH.read_text())
    return {
        "manifest": manifest,
        "metrics": metrics,
        "threshold_table": pd.read_csv(config.THRESHOLD_TABLE_PATH).to_dict("records"),
        "data_quality": json.loads(config.DATA_QUALITY_PATH.read_text()),
    }


@app.get("/v1/policy")
def get_policy():
    """Current thresholds and policy version."""
    return {"review_threshold": policy.review_threshold, "block_threshold": policy.block_threshold, "version": policy.version}


class PolicyIn(BaseModel):
    """New thresholds; their ordering is checked by Policy itself."""
    review_threshold: float = Field(ge=0, le=1)
    block_threshold: float = Field(ge=0, le=1)


@app.put("/v1/policy")
def update_policy(new_thresholds: PolicyIn):
    """Change thresholds without retraining. Affects NEW decisions only; stored decisions keep
    the policy version they were made under. (Protect this endpoint in a real deployment.)"""
    global policy
    try:
        # Bump the numeric suffix, e.g. policy-v3 -> policy-v4; any other format restarts at v2.
        version_number = int(policy.version.rsplit("-v", 1)[1]) + 1
    except (IndexError, ValueError):
        version_number = 2
    try:
        policy = Policy(new_thresholds.review_threshold, new_thresholds.block_threshold, f"policy-v{version_number}")
    except ValueError as error:
        raise HTTPException(422, str(error))
    save_policy(policy, config.POLICY_PATH)
    return get_policy()


@app.get("/v1/policy/simulate")
def simulate_policy(review_threshold: float = Query(ge=0, le=1), block_threshold: float = Query(ge=0, le=1)):
    """What a policy would do on the validation set, before applying it."""
    if review_threshold > block_threshold:
        raise HTTPException(422, "review_threshold must be <= block_threshold")
    # Uses validation scores saved at training time, so nothing is re-scored here.
    labels = scorer.artifact["validation_labels"]
    scores = scorer.artifact["validation_scores"]
    candidate = Policy(review_threshold, block_threshold, "simulation")
    decisions = pd.Series([candidate.decide(score) for score in scores])
    return {
        "validation_transactions": int(len(scores)),
        "decision_mix": {decision: round(float((decisions == decision).mean()), 4) for decision in ("ALLOW", "REVIEW", "BLOCK")},
        "flagged": classification_metrics(labels, scores, review_threshold),
        "blocked": classification_metrics(labels, scores, block_threshold),
    }


@app.get("/v1/data-quality")
def data_quality():
    """Training-data quality report next to live store stats (including rejections)."""
    return {"training_data": json.loads(config.DATA_QUALITY_PATH.read_text()), "live": store.stats()}

