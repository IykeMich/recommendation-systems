"""SageMaker inference handler (guide section 20).

The model artifact carries its own preprocessing, customer profiles and feature
list, so training and serving cannot drift. Package src/ alongside (see
package.sh). The endpoint returns a risk score only; the POLICY is applied by
the caller (Lambda), so thresholds change without redeploying the model.

Request:  {"transaction_id", "customer_id", "amount", "currency", "hour",
           "velocity_1h", "country", "device", "merchant_category"}
Response: {"transaction_id", "risk_score", "model_version", "features"}
"""
import json
import os

from src.predict import FraudScorer

REQUIRED_FIELDS = ["transaction_id", "customer_id", "amount", "hour", "velocity_1h", "country", "device", "merchant_category"]


def model_fn(model_dir):
    """Called once when the container starts: load the artifact from the unpacked model.tar.gz."""
    return FraudScorer.load(os.path.join(model_dir, "fraud_pipeline.joblib"))


def input_fn(request_body, request_content_type="application/json"):
    """Parse and minimally validate the JSON request body (raising ValueError rejects the request)."""
    if request_content_type != "application/json":
        raise ValueError("Expected application/json")
    payload = json.loads(request_body)
    missing = [field for field in REQUIRED_FIELDS if payload.get(field) is None]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    if float(payload["amount"]) <= 0:
        raise ValueError("amount must be positive")
    return payload


def predict_fn(payload, scorer):
    """Score one transaction with the same normalisation as the local API (case of categories)."""
    risk_score, features = scorer.score({
        "transaction_id": payload["transaction_id"],
        "customer_id": payload["customer_id"],
        "amount": float(payload["amount"]),
        "currency": payload.get("currency", "NGN"),
        "hour": int(payload["hour"]),
        "velocity_1h": int(payload["velocity_1h"]),
        "country": str(payload["country"]).upper(),
        "device": str(payload["device"]).lower(),
        "merchant_category": str(payload["merchant_category"]).lower(),
    })
    return {"transaction_id": payload["transaction_id"], "risk_score": round(risk_score, 4),
            "model_version": scorer.model_version, "features": features}


def output_fn(prediction, accept="application/json"):
    """Serialise the prediction as JSON; returns (body, content type)."""
    return json.dumps(prediction), "application/json"
