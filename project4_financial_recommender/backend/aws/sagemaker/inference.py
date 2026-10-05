"""SageMaker inference handler (guide section 38).

Request:  {"user_id": "F00001", "limit": 5}
Response: {"strategy": "eligibility_then_ranking", "user_id": "F00001",
           "recommendations": [{"product_id": "FP001", "score": 0.84, "reasons": ["..."]}]}

Eligibility runs inside the endpoint, before ranking, exactly as locally.
"""
import json
import os

import joblib
import pandas as pd

from src.data import prepare_interactions, prepare_products, prepare_users
from src.recommender import FinancialRecommender


def model_fn(model_dir):
    """Load the ranker, feature lists, rules version and data snapshot packaged by train.py."""
    with open(os.path.join(model_dir, "feature_columns.json")) as feature_file:
        feature_columns = json.load(feature_file)
    with open(os.path.join(model_dir, "metadata.json")) as metadata_file:
        rules_version = json.load(metadata_file)["eligibility_rules_version"]
    recommender = FinancialRecommender(
        model=joblib.load(os.path.join(model_dir, "ranker.joblib")),
        numeric_features=feature_columns["numeric"],
        categorical_features=feature_columns["categorical"],
        users=prepare_users(pd.read_csv(os.path.join(model_dir, "users.csv"))),
        products=prepare_products(pd.read_csv(os.path.join(model_dir, "financial_products.csv"))),
        interactions=prepare_interactions(pd.read_csv(os.path.join(model_dir, "interactions.csv"))),
    )
    return {"recommender": recommender, "rules_version": rules_version}


def input_fn(request_body, content_type="application/json"):
    """Parse and validate the JSON request body."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if not payload.get("user_id"):
        raise ValueError("user_id is required")
    return payload


def predict_fn(payload, loaded):
    """Run eligibility -> ranking -> explanation and return a compact response (reason texts only)."""
    # Clamp limit to 1..20, matching the local API's bounds.
    limit = max(1, min(int(payload.get("limit", 5)), 20))
    result = loaded["recommender"].recommend(payload["user_id"], k=limit)
    return {
        "strategy": result["strategy"],
        "user_id": payload["user_id"],
        "rules_version": loaded["rules_version"],
        "recommendations": [
            {
                "product_id": product["product_id"],
                "score": product["score"],
                "reasons": [reason["text"] for reason in product["reasons"]],
            }
            for product in result["recommendations"]
        ],
    }


def output_fn(prediction, accept="application/json"):
    """Serialise the prediction as JSON."""
    return json.dumps(prediction, default=str), "application/json"
