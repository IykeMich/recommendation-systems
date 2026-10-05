"""FastAPI backend for the offer recommender: context-aware recommendations, user search and
history, click/redeem feedback (applied to history features at once) and on-demand retraining."""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import ARTIFACT_DIR, DEVICE_TYPES, FEEDBACK_LOG_PATH, REGIONS, TOP_K
from src.data import append_new_interaction, to_internal_schema
from src.features import daypart_for_hour
from src.model import describe_feature
from src.recommender import load_recommender
from src.train import train

app = FastAPI(title="ShopSmart Contextual Offer API", version="1.0.0")

# Let the Next.js frontend call this API from the browser. Always allowed: any localhost / 127.0.0.1
# port (local testing) and any https://*.vercel.app URL (Vercel production and preview deployments).
# Other deployed frontends (e.g. a custom domain) go in CORS_ORIGINS, comma-separated.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip().rstrip("/") for origin in os.environ.get("CORS_ORIGINS", "").split(",") if origin.strip()
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?|https://[a-z0-9-]+\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_state():
    """Train once if no saved model exists, then load the recommender and its training metadata."""
    if not (ARTIFACT_DIR / "ranker.joblib").exists():
        train()
    return load_recommender(), json.loads((ARTIFACT_DIR / "metadata.json").read_text())


# Module-level state shared by every request; /v1/model/retrain replaces both.
recommender, model_metadata = load_state()

DeviceType = Literal["mobile", "tablet", "web"]


def log_feedback(feedback_event: dict) -> None:
    """Append one event as a JSON line to the feedback log (impressions, clicks and redeems)."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a") as feedback_log:
        feedback_log.write(json.dumps(feedback_event) + "\n")


def pending_interaction_count() -> int:
    """Interactions recorded since the current model was trained (in memory minus trained count)."""
    return max(0, len(recommender.interactions) - model_metadata["num_interactions"])


@app.get("/health")
def health():
    """Liveness check; the UI polls it for its 'API online' indicator."""
    return {"status": "ok"}


@app.get("/v1/context/options")
def context_options():
    """Values the UI offers for each context field."""
    return {"device_types": DEVICE_TYPES, "regions": REGIONS}


@app.get("/v1/offers/recommendations/{user_id}")
def recommendations(
    user_id: str,
    device_type: DeviceType = "web",
    region: str = "Lagos",
    hour: int = Query(default=12, ge=0, le=23),
    day_of_week: int = Query(default=0, ge=0, le=6),
    limit: int = Query(default=TOP_K, ge=1, le=50),
    enforce_region: bool = True,
    enforce_channel: bool = True,
    max_per_category: Optional[int] = Query(default=None, ge=1, le=10),
):
    """Top-K offers for a user in the given context. Rule toggles relax region/channel filtering;
    max_per_category caps repeats. Each call also logs a recommendation_impression event."""
    context = {"device_type": device_type, "region": region, "hour": hour, "day_of_week": day_of_week}
    result = recommender.recommend(
        user_id,
        context,
        k=limit,
        enforce_region=enforce_region,
        enforce_channel=enforce_channel,
        max_per_category=max_per_category,
    )
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    # Capture the context that produced the recommendation (guide section 32).
    log_feedback({
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "request_id": request_id,
        "user_id": user_id,
        "event_type": "recommendation_impression",
        "strategy": result["strategy"],
        "offer_ids": [offer["offer_id"] for offer in result["recommendations"]],
        "context": context,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    return {
        "strategy": result["strategy"],
        "user_id": user_id,
        "request_id": request_id,
        "context": {**context, "daypart": daypart_for_hour(hour)},
        "candidate_funnel": result["funnel"],
        "model_trained_at": model_metadata["trained_at"],
        "recommendations": result["recommendations"],
    }


@app.get("/v1/users")
def search_users(search: str = "", limit: int = Query(default=10, ge=1, le=100)):
    """Find users by ID prefix, or by substring of segment, preferred category or region."""
    users = recommender.users
    matches = users
    if search.strip():
        search_term = search.strip().lower()
        matches = users[
            users.user_id.str.lower().str.startswith(search_term)
            | users.segment.str.lower().str.contains(search_term, regex=False)
            | users.preferred_category.str.lower().str.contains(search_term, regex=False)
            | users.region.str.lower().str.contains(search_term, regex=False)
        ]
    return {"total": int(len(matches)), "users": matches.head(limit).to_dict("records")}


@app.get("/v1/users/{user_id}")
def user_detail(user_id: str, history_limit: int = Query(default=12, ge=1, le=100)):
    """Profile, event counts and recent history; `after_training` marks events the model hasn't seen."""
    profile_rows = recommender.users[recommender.users.user_id == user_id]
    history = recommender.interactions[recommender.interactions.user_id == user_id]
    if profile_rows.empty and history.empty:
        # Unknown only if there is neither a profile nor any history for this ID.
        raise HTTPException(404, "Unknown user_id")

    offer_lookup = recommender.offers.set_index("offer_id")
    recent_history = history.sort_values("timestamp", ascending=False).head(history_limit)
    # Interactions are stored in arrival order; rows past the trained count are new.
    trained_interaction_count = model_metadata["num_interactions"]
    return {
        "user_id": user_id,
        "profile": None if profile_rows.empty else profile_rows.iloc[0].to_dict(),
        "event_type_counts": {
            event_type: int(count) for event_type, count in history["event_type"].value_counts().items()
        },
        "history": [
            {
                "offer_id": event.offer_id,
                "title": offer_lookup.loc[event.offer_id, "title"] if event.offer_id in offer_lookup.index else None,
                "category": offer_lookup.loc[event.offer_id, "category"] if event.offer_id in offer_lookup.index else None,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "device_type": event.device_type,
                "region": event.region,
                "after_training": bool(event.Index >= trained_interaction_count),
            }
            for event in recent_history.itertuples()
        ],
    }


class FeedbackContext(BaseModel):
    """Context the user was in when they clicked/redeemed (same limits as the GET query params)."""
    device_type: DeviceType
    region: str
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)


class FeedbackEventIn(BaseModel):
    """Body of POST /v1/offers/events; request_id links the event to the list that showed the offer."""
    user_id: str
    offer_id: str
    event_type: Literal["click", "redeem"]
    request_id: Optional[str] = None
    context: FeedbackContext


@app.post("/v1/offers/events")
def record_event(feedback_event: FeedbackEventIn):
    """Validate a click/redeem, write it to the feedback log and new-interactions CSV, and add it
    to the in-memory history so the next recommendation reflects it."""
    if not recommender.offers["offer_id"].eq(feedback_event.offer_id).any():
        raise HTTPException(404, "Unknown offer_id")

    logged_event = {
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        **feedback_event.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log_feedback(logged_event)

    # Raw-schema row with the server's current time. Training derives hour/day from this timestamp,
    # so context.hour/day_of_week are kept only in the feedback log, not in the interaction.
    raw_interaction = {
        "user_id": feedback_event.user_id,
        "offer_id": feedback_event.offer_id,
        "event_type": feedback_event.event_type,
        "timestamp": int(time.time()),
        "device_type": feedback_event.context.device_type,
        "region": feedback_event.context.region,
    }
    append_new_interaction(raw_interaction)
    # History features (affinity, popularity, redeemed offers) update immediately;
    # the model's weights only change on retrain.
    recommender.add_interaction(to_internal_schema(pd.DataFrame([raw_interaction])).iloc[0].to_dict())

    return {
        "accepted": True,
        "event": logged_event,
        "pending_new_interactions": pending_interaction_count(),
    }


@app.get("/v1/model")
def model_info():
    """Training metadata, readable coefficient labels and the count of feedback not yet trained on."""
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
    """Refit the ranker on all interactions. The last offline evaluation is kept
    (re-run it with `python -m src.train`)."""
    global recommender, model_metadata
    # Retraining skips the slow evaluation, so the previous evaluation is copied into the new
    # metadata (and metadata.json) to keep the UI's metrics table populated.
    previous_evaluation = model_metadata.get("evaluation")
    _, new_metadata = train(include_new_interactions=True, run_evaluation=False)
    if previous_evaluation:
        new_metadata["evaluation"] = previous_evaluation
        (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(new_metadata, indent=2))
    recommender, model_metadata = load_recommender(), new_metadata
    return model_info()
