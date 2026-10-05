"""SageMaker inference handler (guide section 30).

Request:  {"user_id": "R00001", "context": {"device_type": "mobile", "region": "Lagos",
           "hour": 18, "day_of_week": 2}, "limit": 10}
Response: {"user_id": "R00001", "strategy": "...", "recommendations": [{"offer_id": "O00012", "score": 0.91}]}

model_fn runs once at container start; requests only do inference.
"""
import json
import os

import joblib
import pandas as pd

from src.data import prepare_offers, to_internal_schema
from src.recommender import ContextualRecommender

REQUIRED_CONTEXT = {"device_type", "region", "hour", "day_of_week"}


def model_fn(model_dir):
    """Load the ranker and the data snapshot that train.py copied into the model directory."""
    with open(os.path.join(model_dir, "feature_columns.json")) as feature_file:
        feature_columns = json.load(feature_file)
    return ContextualRecommender(
        model=joblib.load(os.path.join(model_dir, "ranker.joblib")),
        numeric_features=feature_columns["numeric"],
        categorical_features=feature_columns["categorical"],
        offers=prepare_offers(pd.read_csv(os.path.join(model_dir, "offers.csv"))),
        users=pd.read_csv(os.path.join(model_dir, "users.csv")),
        interactions=to_internal_schema(pd.read_csv(os.path.join(model_dir, "offer_interactions.csv"))),
    )


def input_fn(request_body, content_type="application/json"):
    """Parse the JSON body and check user_id plus all four context fields are present."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    missing_context = REQUIRED_CONTEXT - set(payload.get("context", {}))
    if not payload.get("user_id") or missing_context:
        raise ValueError(f"user_id and context {sorted(REQUIRED_CONTEXT)} are required")
    return payload


def predict_fn(payload, recommender):
    """Recommend for the payload's user and context; limit is clamped to 1-50 like the local API.
    Only offer_id and score are returned per offer."""
    context = {
        "device_type": payload["context"]["device_type"],
        "region": payload["context"]["region"],
        "hour": int(payload["context"]["hour"]),
        "day_of_week": int(payload["context"]["day_of_week"]),
    }
    limit = max(1, min(int(payload.get("limit", 10)), 50))
    result = recommender.recommend(payload["user_id"], context, k=limit)
    return {
        "user_id": payload["user_id"],
        "strategy": result["strategy"],
        "context": context,
        "recommendations": [
            {"offer_id": offer["offer_id"], "score": offer["score"]} for offer in result["recommendations"]
        ],
    }


def output_fn(prediction, accept="application/json"):
    """Serialise the prediction as the JSON response body."""
    return json.dumps(prediction), "application/json"
