"""FastAPI backend for the financial recommender.

Serves recommendations (eligibility -> ranker -> explanations), pairwise eligibility checks,
user search/detail, feedback capture and retraining. Every recommendation is audit-logged with
request ID, strategy, model version and rules version, but never the financial profile itself.
Run with: uvicorn api.main:app --reload --port 8004
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import AGE_BANDS, ARTIFACT_DIR, EMPLOYMENT_TYPES, FEEDBACK_LOG_PATH, RISK_PROFILES, TOP_K
from src.data import append_new_interaction, prepare_interactions
from src.eligibility import RULES, RULES_VERSION, check_eligibility
from src.model import describe_feature
from src.recommender import load_recommender
from src.train import train

app = FastAPI(title="Financial Product Recommendation API", version="1.0.0")

# Allow the Next.js dev server (or origins from CORS_ORIGINS) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3004,http://127.0.0.1:3004").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_state():
    """Load the recommender and model metadata, training first if no artifacts exist yet."""
    if not (ARTIFACT_DIR / "ranker.joblib").exists():
        train()
    return load_recommender(), json.loads((ARTIFACT_DIR / "metadata.json").read_text())


# Loaded once at import; /v1/model/retrain replaces both globals.
recommender, model_metadata = load_state()

# Literal types make FastAPI reject invalid what-if values with a 422.
AgeBand = Literal["18-24", "25-34", "35-44", "45-54", "55+"]
Employment = Literal["salary", "self_employed", "business_owner", "student"]
RiskProfile = Literal["low", "medium", "high"]


def log_feedback(feedback_event: dict) -> None:
    """Append one JSON event (impression or feedback) to the JSONL audit/feedback log."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a") as feedback_log:
        feedback_log.write(json.dumps(feedback_event, default=str) + "\n")


def pending_interaction_count() -> int:
    """Interactions recorded since the current model was trained."""
    return max(0, len(recommender.interactions) - model_metadata["num_interactions"])


def product_or_404(product_id: str) -> dict:
    """Catalog row for product_id, or HTTP 404."""
    matches = recommender.products[recommender.products["product_id"] == product_id]
    if matches.empty:
        raise HTTPException(404, "Unknown product_id")
    return matches.iloc[0].to_dict()


@app.get("/health")
def health():
    """Liveness check used by the UI's status indicator."""
    return {"status": "ok"}


@app.get("/v1/financial-products")
def product_catalog():
    """The full product catalog."""
    return {"products": recommender.products.to_dict("records")}


@app.get("/v1/financial-products/recommendations/{user_id}")
def recommendations(
    user_id: str,
    limit: int = Query(default=TOP_K, ge=1, le=20),
    include_unfiltered: bool = Query(default=False, description="Also return the ranker's order WITHOUT eligibility (teaching comparison)."),
    age_band: Optional[AgeBand] = None,
    monthly_income: Optional[float] = Query(default=None, ge=0),
    employment: Optional[Employment] = None,
    risk_profile: Optional[RiskProfile] = None,
    existing_products: Optional[int] = Query(default=None, ge=0, le=20),
):
    """Eligibility first, then ranking. Optional profile fields are what-if overrides."""
    # None means "not overridden"; resolve_profile drops those before merging onto the dataset profile.
    profile_overrides = {
        "age_band": age_band,
        "monthly_income": monthly_income,
        "employment": employment,
        "risk_profile": risk_profile,
        "existing_products": existing_products,
    }
    result = recommender.recommend(user_id, k=limit, profile_overrides=profile_overrides, include_unfiltered=include_unfiltered)
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    # Audit trail: request, model and rule versions. The financial profile itself is not logged.
    log_feedback({
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "request_id": request_id,
        "user_id": user_id,
        "event_type": "recommendation_impression",
        "recommendation_strategy": result["strategy"],
        "profile_source": result["profile_source"],
        "product_ids": [product["product_id"] for product in result["recommendations"]],
        "model_version": model_metadata["model_version"],
        "rules_version": RULES_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return {
        **result,
        "request_id": request_id,
        "model_version": model_metadata["model_version"],
    }


@app.get("/v1/eligibility/rules")
def eligibility_rules():
    """The active rule set and its version."""
    return {"rules_version": RULES_VERSION, "rules": RULES}


@app.get("/v1/eligibility/{user_id}/{product_id}")
def eligibility_check(user_id: str, product_id: str):
    """Check one user/product pair against the rules, without the model."""
    profile, _ = recommender.resolve_profile(user_id)
    product = product_or_404(product_id)
    failed_rules = check_eligibility(profile or {"user_id": user_id}, product)
    return {
        "user_id": user_id,
        "product_id": product_id,
        "eligible": not failed_rules,
        "failed_rules": failed_rules,
        "rules_version": RULES_VERSION,
    }


@app.get("/v1/users")
def search_users(search: str = "", limit: int = Query(default=10, ge=1, le=100)):
    """Find users by ID prefix, employment, exact risk profile, region or age band."""
    users = recommender.users
    matches = users
    if search.strip():
        search_term = search.strip().lower()
        matches = users[
            users.user_id.str.lower().str.startswith(search_term)
            | users.employment.str.lower().str.contains(search_term, regex=False)
            | users.risk_profile.str.lower().eq(search_term)
            | users.region.str.lower().str.contains(search_term, regex=False)
            | users.age_band.str.contains(search_term, regex=False)
        ]
    return {"total": int(len(matches)), "users": matches.head(limit).to_dict("records")}


@app.get("/v1/users/{user_id}")
def user_detail(user_id: str, history_limit: int = Query(default=12, ge=1, le=100)):
    """Dataset profile, event counts and recent history; 404 only if there is neither."""
    profile, _ = recommender.resolve_profile(user_id)
    history = recommender.interactions[recommender.interactions["user_id"] == user_id]
    if profile is None and history.empty:
        raise HTTPException(404, "Unknown user_id")

    names = recommender.products.set_index("product_id")["name"]
    # Interactions are stored in arrival order; rows past the trained count are new.
    trained_interaction_count = model_metadata["num_interactions"]
    recent_history = history.sort_values("timestamp", ascending=False).head(history_limit)
    return {
        "user_id": user_id,
        "profile": profile,
        "event_type_counts": {event_type: int(count) for event_type, count in history["event_type"].value_counts().items()},
        "history": [
            {
                "product_id": event.product_id,
                "name": names.get(event.product_id),
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                # Position-based: relies on rows past the trained count being the newest ones.
                "after_training": bool(event.Index >= trained_interaction_count),
            }
            for event in recent_history.itertuples()
        ],
    }


@app.get("/v1/profile/options")
def profile_options():
    """Allowed values for the what-if profile editor."""
    return {"age_bands": AGE_BANDS, "employment_types": EMPLOYMENT_TYPES, "risk_profiles": RISK_PROFILES}


class FeedbackEventIn(BaseModel):
    """Feedback body. request_id/strategy link the action to the recommendation that preceded it."""
    user_id: str
    product_id: str
    event_type: Literal["recommendation_click", "view", "learn", "apply"]
    request_id: Optional[str] = None
    recommendation_strategy: Optional[str] = None


@app.post("/v1/financial-products/events")
def record_event(feedback_event: FeedbackEventIn):
    """Log a feedback event; view/learn/apply also become new training interactions."""
    product = product_or_404(feedback_event.product_id)
    profile, _ = recommender.resolve_profile(feedback_event.user_id)

    logged_event = {
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        **feedback_event.model_dump(),
        "eligible_at_event_time": is_eligible_now(profile, product, feedback_event.user_id),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log_feedback(logged_event)

    # view / learn / apply are business interactions and become training data.
    # A recommendation_click stays in the feedback log only.
    becomes_training_data = feedback_event.event_type != "recommendation_click"
    if becomes_training_data:
        raw_interaction = {
            "user_id": feedback_event.user_id,
            "product_id": feedback_event.product_id,
            "event_type": feedback_event.event_type,
            "timestamp": int(time.time()),
        }
        # Persist to new_interactions.csv for the next retrain, and update the in-memory history so
        # history features and reasons reflect the event immediately (the model weights do not).
        append_new_interaction(raw_interaction)
        recommender.add_interaction(prepare_interactions(pd.DataFrame([raw_interaction])).iloc[0].to_dict())

    return {
        "accepted": True,
        "event": logged_event,
        "becomes_training_data": becomes_training_data,
        "pending_new_interactions": pending_interaction_count(),
    }


def is_eligible_now(profile, product, user_id) -> bool:
    """Eligibility of the pair under the current rules (unknown user -> empty profile)."""
    return not check_eligibility(profile or {"user_id": user_id}, product)


@app.get("/v1/model")
def model_info():
    """Model metadata with readable coefficient labels and the pending-feedback count."""
    return {
        **model_metadata,
        "top_coefficients": [
            {**coefficient, "label": describe_feature(coefficient["feature"])}
            for coefficient in model_metadata["top_coefficients"]
        ],
        "pending_new_interactions": pending_interaction_count(),
    }


@app.post("/v1/model/retrain")
def retrain_model():
    """Refit the ranker on all interactions (the last offline evaluation is kept;
    re-run `python -m src.train` to re-evaluate)."""
    global recommender, model_metadata
    # Retraining skips the slow evaluation, so carry the previous evaluation into the new metadata.
    previous_evaluation = model_metadata.get("evaluation")
    _, new_metadata = train(include_new_interactions=True, run_evaluation=False)
    if previous_evaluation:
        new_metadata["evaluation"] = previous_evaluation
        (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(new_metadata, indent=2))
    recommender, model_metadata = load_recommender(), new_metadata
    return model_info()
