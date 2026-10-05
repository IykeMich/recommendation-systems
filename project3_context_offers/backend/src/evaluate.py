"""Time-aware offline evaluation (guide sections 17-18 and 25).

Train on the earliest 80% of events; for every click/redeem in the last 20%,
rank the whole catalog in that event's context and check where the offer the
user actually engaged with lands.
"""
import numpy as np
import pandas as pd

from .config import NEGATIVES_PER_POSITIVE, RANDOM_SEED, TEST_FRACTION, TOP_K
from .features import (
    CATEGORICAL_FEATURES,
    CONTEXT_FEATURES,
    NUMERIC_FEATURES,
    add_time_features,
    build_exposure_examples,
    build_feature_frame,
    build_training_examples,
)
from .model import build_model


def time_split(interactions: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split at the (1 - test_fraction) timestamp quantile: earlier events train, later ones test."""
    interactions = interactions.sort_values("timestamp")
    cutoff = interactions["timestamp"].quantile(1 - test_fraction)
    return interactions[interactions["timestamp"] <= cutoff], interactions[interactions["timestamp"] > cutoff], cutoff


def recall_at_k(actual_items, recommended_items, k):
    """Share of the actual items found in the first k recommendations (general helper; unused here)."""
    recommended = set(recommended_items[:k])
    actual = set(actual_items)
    return len(recommended.intersection(actual)) / len(actual) if actual else 0.0


def without_context(features):
    """The feature list minus the context features (for the 'no_context' ablation)."""
    return [feature for feature in features if feature not in CONTEXT_FEATURES]


def fit_variant(training_requests, offers, users, history, numeric, categorical):
    """Build point-in-time features for the training requests and fit one ranker variant."""
    training_frame = build_feature_frame(training_requests, offers, users, history)
    model = build_model(numeric, categorical)
    model.fit(training_frame[numeric + categorical], training_frame["label"])
    return model


def ranking_metrics(
    scored: pd.DataFrame, score_column: str, offers: pd.DataFrame, k: int, max_per_category=None
) -> dict:
    """scored has one row per (test request, candidate offer) and an is_target flag."""
    request_ids = scored["request_id"].unique()
    # Rank candidates inside each request; offer_id breaks score ties so results are deterministic.
    scored = scored.sort_values(["request_id", score_column, "offer_id"], ascending=[True, False, True])
    # Optional diversity cap: keep only the first N rows per category before assigning ranks.
    if max_per_category:
        scored = scored[scored.groupby(["request_id", "category"]).cumcount() < max_per_category]
    scored = scored.assign(rank=scored.groupby("request_id").cumcount() + 1)
    top_k = scored[scored["rank"] <= k]

    # Rank of the offer the user really engaged with; inf if the cap removed it (counts as a miss).
    target_ranks = (
        scored.loc[scored["is_target"], ["request_id", "rank"]]
        .set_index("request_id")["rank"]
        .reindex(request_ids, fill_value=np.inf)
    )
    # One relevant offer per request, so NDCG@k reduces to 1/log2(rank + 1) for a hit, else 0.
    # Coverage = share of the catalog shown anywhere; diversity = distinct categories per list / k.
    hits = target_ranks <= k
    return {
        f"recall@{k}": round(float(hits.mean()), 4),
        f"ndcg@{k}": round(float((hits / np.log2(target_ranks + 1)).mean()), 4),
        "coverage": round(top_k["offer_id"].nunique() / len(offers), 4),
        "diversity": round(float(top_k.groupby("request_id")["category"].nunique().mean() / k), 4),
    }


def evaluate(interactions, offers, users, k: int = TOP_K) -> dict:
    """Fit each variant on the training period, rank the full catalog for every test click/redeem,
    and return overall metrics plus recall per daypart and per device."""
    train_events, test_events, cutoff = time_split(interactions)

    # Ablations: the full model, the same examples without context features, and the guide's
    # literal setup (every logged event labelled engaged / not engaged).
    variants = {
        "contextual": (build_training_examples(train_events, offers, NEGATIVES_PER_POSITIVE, RANDOM_SEED),
                       NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "no_context": (build_training_examples(train_events, offers, NEGATIVES_PER_POSITIVE, RANDOM_SEED),
                       without_context(NUMERIC_FEATURES), without_context(CATEGORICAL_FEATURES)),
        "exposure_labels": (build_exposure_examples(train_events),
                            NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    models = {
        name: (fit_variant(requests, offers, users, train_events, numeric, categorical), numeric + categorical)
        for name, (requests, numeric, categorical) in variants.items()
    }

    # Every test engagement becomes a request over the whole catalog, in its own context.
    test_positives = add_time_features(test_events[test_events["label"] == 1]).reset_index(drop=True)
    test_positives["request_id"] = np.arange(len(test_positives))
    # Cross join: each test request x every offer, so the target competes with the whole catalog.
    test_requests = test_positives[["request_id", "user_id", "device_type", "region", "hour", "day_of_week"]].merge(
        offers[["offer_id"]], how="cross"
    )
    test_requests["request_time"] = test_requests["request_id"].map(test_positives["timestamp"])
    target_offer = test_requests["request_id"].map(test_positives["offer_id"])

    # History = everything before each request, including earlier test-period events.
    scored = build_feature_frame(test_requests, offers, users, interactions)
    scored["request_id"] = test_requests["request_id"].to_numpy()
    scored["is_target"] = (scored["offer_id"] == target_offer.to_numpy())

    for name, (model, columns) in models.items():
        scored[f"score_{name}"] = model.predict_proba(scored[columns])[:, 1]
    # Baseline: rank purely by point-in-time engagement count.
    scored["score_popularity"] = scored["offer_popularity"]

    results = {
        name: ranking_metrics(scored, f"score_{name}", offers, k)
        for name in ["contextual", "no_context", "exposure_labels", "popularity"]
    }
    results["contextual_max_3_per_category"] = ranking_metrics(
        scored, "score_contextual", offers, k, max_per_category=3
    )

    # Recall broken down by daypart and device, to see whether context helps in some segments.
    by_segment = {}
    for segment_column in ["daypart", "device_type"]:
        by_segment[segment_column] = {}
        for segment_value, segment_rows in scored.groupby(segment_column):
            by_segment[segment_column][segment_value] = {
                name: ranking_metrics(segment_rows, f"score_{name}", offers, k)[f"recall@{k}"]
                for name in ["contextual", "no_context", "popularity"]
            } | {"requests": int(segment_rows["request_id"].nunique())}

    return {
        "protocol": "time-based holdout: train on the earliest 80% of events, "
                    "rank the full catalog for each click/redeem in the last 20%",
        "k": k,
        "cutoff": cutoff.isoformat(),
        "train_events": int(len(train_events)),
        "test_requests": int(len(test_positives)),
        "models": results,
        "by_segment": by_segment,
    }
