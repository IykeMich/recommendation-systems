"""ShopSmart FastAPI backend: serves recommendations, shopper data, feedback events and retraining.

Run with: uvicorn api.main:app --reload --port 8002 (Swagger UI at /docs).
The model is held in memory; POST /v1/model/retrain swaps in a freshly trained one.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import ARTIFACT_DIR, EVENT_WEIGHTS, FEEDBACK_LOG_PATH
from src.data import append_new_interaction, load_data, load_new_interactions
from src.preprocess import prepare_events
from src.recommender import AdaptiveHybridRecommender, popular_items
from src.train import train

app = FastAPI(title="ShopSmart Recommendation API", version="1.0.0")

# Let the Next.js dev server (a different origin) call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "http://localhost:3002,http://127.0.0.1:3002"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at start-up. original_event_count lets us tell original rows apart from new interactions.
users, products, original_events = load_data()
original_event_count = len(original_events)
product_lookup = products.set_index("item_id")


def load_model_state():
    """Load the trained artifact (training it first if it is missing or from an older model type)."""
    model_path = ARTIFACT_DIR / "model.joblib"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        return train()
    loaded_model = joblib.load(model_path)
    if not isinstance(loaded_model, AdaptiveHybridRecommender):
        return train()
    return loaded_model, json.loads(metadata_path.read_text())


# Module-level state shared by all requests; replaced by retrain_model().
model, model_metadata = load_model_state()


def load_weighted_training_events() -> pd.DataFrame:
    """The events the current model was trained on: original data + incorporated new ones."""
    training_events = original_events
    # new_interactions.csv is append-only, so the model's first N new rows are the ones it was trained on,
    # where N = events in the model minus original events.
    incorporated_count = model_metadata["num_events"] - original_event_count
    if incorporated_count > 0:
        training_events = pd.concat(
            [original_events, load_new_interactions().head(incorporated_count)],
            ignore_index=True,
        )
    return prepare_events(training_events)


# Used for the popularity fallback and for the non-pending part of a shopper's history.
weighted_events = load_weighted_training_events()


def pending_interaction_count() -> int:
    """New interactions recorded since the current model was trained."""
    incorporated_count = model_metadata["num_events"] - original_event_count
    return max(0, len(load_new_interactions()) - incorporated_count)


def product_details(item_id: str) -> dict:
    """Catalog fields for an item, or all-None fields if the item isn't in products.csv."""
    if item_id not in product_lookup.index:
        return {"name": None, "category": None, "subcategory": None, "brand": None, "price": None}
    product = product_lookup.loc[item_id]
    return {
        "name": product["name"],
        "category": product["category"],
        "subcategory": product["subcategory"],
        "brand": product["brand"],
        "price": float(product["price"]),
    }


def log_feedback(feedback_event: dict) -> None:
    """Append one event as a JSON line to the feedback log (JSONL: one JSON object per line)."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a") as feedback_log:
        feedback_log.write(json.dumps(feedback_event) + "\n")


@app.get("/health")
def health():
    """Liveness check used by the frontend's API status indicator."""
    return {"status": "ok"}


@app.get("/v1/model")
def model_info():
    """Training metadata and metrics, plus how many new interactions are waiting for a retrain."""
    return {**model_metadata, "pending_new_interactions": pending_interaction_count()}


@app.post("/v1/model/retrain")
def retrain_model():
    """Fold new interactions into the training data and swap in the new model."""
    # Rebind the module-level state so later requests use the new model.
    global model, model_metadata, weighted_events
    model, model_metadata = train(include_new_interactions=True)
    weighted_events = load_weighted_training_events()
    return model_info()


@app.get("/v1/users")
def search_users(search: str = "", limit: int = Query(default=12, ge=1, le=100)):
    """Case-insensitive shopper search: user_id prefix, or substring of segment/category/region."""
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


@app.get("/v1/products")
def search_products(
    search: str = "",
    category: str = "",
    limit: int = Query(default=24, ge=1, le=100),
):
    """Catalog search: case-insensitive substring of name/brand/category/subcategory, or item_id prefix.

    `category` is an optional exact filter; `categories` lists every category for the UI's filter.
    """
    matches = products
    if category:
        matches = matches[matches.category == category]
    if search.strip():
        search_term = search.strip().lower()
        matches = matches[
            matches.item_id.str.lower().str.startswith(search_term)
            | matches.name.str.lower().str.contains(search_term, regex=False)
            | matches.brand.str.lower().str.contains(search_term, regex=False)
            | matches.category.str.lower().str.contains(search_term, regex=False)
            | matches.subcategory.str.lower().str.contains(search_term, regex=False)
        ]
    return {
        "total": int(len(matches)),
        "categories": sorted(products.category.unique().tolist()),
        "products": [
            {"item_id": item_id, **product_details(item_id)} for item_id in matches.item_id.head(limit)
        ],
    }


@app.get("/v1/users/{user_id}")
def user_detail(user_id: str, history_limit: int = Query(default=15, ge=1, le=100)):
    """Profile plus recent history; 404 only if the user has neither a profile nor any events."""
    profile_rows = users[users.user_id == user_id]
    # Combine events the model was trained on (pending=False) with all new interactions (pending=True).
    # New rows already folded into the model appear in both; keep="first" keeps the pending=False copy.
    new_events = load_new_interactions()
    new_events = new_events[new_events.user_id == user_id].assign(pending=True)
    history = pd.concat(
        [weighted_events[weighted_events.user_id == user_id].assign(pending=False), new_events],
        ignore_index=True,
    ).drop_duplicates(subset=["user_id", "item_id", "event_type", "timestamp"], keep="first")

    if profile_rows.empty and history.empty:
        raise HTTPException(404, "Unknown user_id")

    recent_history = history.sort_values("timestamp", ascending=False).head(history_limit)
    return {
        "user_id": user_id,
        "profile": None if profile_rows.empty else profile_rows.iloc[0].to_dict(),
        "known_to_model": model.knows_user(user_id),
        "event_count": int(len(history)),
        "event_type_counts": {
            event_type: int(count)
            for event_type, count in history["event_type"].value_counts().items()
        },
        "history": [
            {
                "item_id": event.item_id,
                "event_type": event.event_type,
                "timestamp": int(event.timestamp),
                "pending": bool(event.pending),
                **product_details(event.item_id),
            }
            for event in recent_history.itertuples()
        ],
    }


def strongest_events(user_id: str) -> dict:
    """item_id -> the user's strongest trained-on action on it ("purchase" > "cart" > "view")."""
    user_events = weighted_events[weighted_events.user_id == user_id]
    strongest_rows = user_events.sort_values("weight").drop_duplicates("item_id", keep="last")
    return dict(zip(strongest_rows.item_id, strongest_rows.event_type))


def readable_reasons(reasons: list, your_events: dict) -> list:
    """Add the anchor product's name and the shopper's own action on it to each model reason."""
    return [
        {**reason, "name": product_details(reason["item_id"])["name"],
         "your_event": your_events.get(reason["item_id"])}
        for reason in reasons
    ]


@app.get("/v1/recommendations/user/{user_id}")
def recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50),
):
    """Personal Top-K for a user, falling back to popular items when the model returns nothing."""
    # The request id ties later clicks/purchases back to this exact list of recommendations.
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    recs = model.recommend(user_id, k=limit, explain=True)
    if recs:
        strategy = "adaptive_hybrid"
    else:
        strategy = "popular_fallback"
        recs = popular_items(weighted_events, k=limit)

    # Attach catalog details and readable "why this?" reasons to each recommendation.
    your_events = strongest_events(user_id) if strategy == "adaptive_hybrid" else {}
    enriched_recommendations = [
        {
            "item_id": recommendation["item_id"],
            "score": recommendation["score"],
            **product_details(recommendation["item_id"]),
            "source": recommendation.get("source", "popular"),
            "reasons": readable_reasons(recommendation.get("reasons", []), your_events),
        }
        for recommendation in recs
    ]

    # Log what was shown (an impression) so offline analysis can join it with later clicks/purchases.
    log_feedback({
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "event_type": "recommendation_impression",
        "recommendation_request_id": request_id,
        "user_id": user_id,
        "strategy": strategy,
        "item_ids": [recommendation["item_id"] for recommendation in enriched_recommendations],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    return {
        "strategy": strategy,
        "user_id": user_id,
        "recommendation_request_id": request_id,
        "model_trained_at": model_metadata.get("trained_at"),
        "recommendations": enriched_recommendations,
    }


class FeedbackEventIn(BaseModel):
    """Request body for POST /v1/events; pydantic rejects unknown event types with a 422."""

    user_id: str
    item_id: str
    event_type: Literal["recommendation_click", "view", "cart", "purchase"]
    recommendation_request_id: Optional[str] = None
    device_type: str = "web"
    region: str = "unknown"


@app.post("/v1/events")
def record_event(feedback_event: FeedbackEventIn):
    """Log a feedback event; view/cart/purchase are also queued as training data for the next retrain."""
    if feedback_event.item_id not in product_lookup.index:
        raise HTTPException(404, "Unknown item_id")

    logged_event = {
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        **feedback_event.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log_feedback(logged_event)

    # Business interactions become training data on the next retrain;
    # recommendation clicks stay in the feedback log only.
    becomes_training_data = feedback_event.event_type in EVENT_WEIGHTS
    if becomes_training_data:
        append_new_interaction({
            "user_id": feedback_event.user_id,
            "item_id": feedback_event.item_id,
            "event_type": feedback_event.event_type,
            "event_value": EVENT_WEIGHTS[feedback_event.event_type],
            "timestamp": int(time.time()),
            "device_type": feedback_event.device_type,
            "region": feedback_event.region,
        })

    return {
        "accepted": True,
        "event": logged_event,
        "becomes_training_data": becomes_training_data,
        "pending_new_interactions": pending_interaction_count(),
    }
