"""SageMaker inference handler (section 19 contract).

Input:  {"user_id": "R00001", "limit": 10}
Output: {"user_id": "R00001", "strategy": "...",
         "recommendations": [{"item_id": "P00123", "score": 0.82, "source": "similar_products"}]}

model_fn runs once when the container starts; each request only does inference.
"""
import json
import os

import joblib


def model_fn(model_dir):
    """Load the pickled model and the precomputed popular-items list from the unpacked model.tar.gz."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "popular_items.json")) as popular_file:
        popular = json.load(popular_file)
    return {"model": model, "popular": popular}


def input_fn(request_body, content_type="application/json"):
    """Parse and validate the JSON request body."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if not payload.get("user_id"):
        raise ValueError("user_id is required")
    return payload


def predict_fn(payload, loaded):
    """Recommend for the user, using the popularity list when the model has nothing to offer."""
    # Clamp limit to 1..50 (popular_items.json holds at most 50 items).
    limit = max(1, min(int(payload.get("limit", 10)), 50))
    recommendations = loaded["model"].recommend(payload["user_id"], k=limit)
    strategy = "adaptive_hybrid"
    if not recommendations:
        strategy = "popular_fallback"
        recommendations = loaded["popular"][:limit]
    return {
        "user_id": payload["user_id"],
        "strategy": strategy,
        "recommendations": recommendations,
    }


def output_fn(prediction, accept="application/json"):
    """Serialise the prediction as JSON, returning (body, content type) as SageMaker expects."""
    return json.dumps(prediction), "application/json"
