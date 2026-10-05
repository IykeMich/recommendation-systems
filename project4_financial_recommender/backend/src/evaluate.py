"""Temporal evaluation of four strategies (guide sections 26-28).

Train on the earliest 80% of events. For every application in the last 20%,
rank the catalog as of that moment and check where the applied product lands.
Quality metrics (recall, NDCG, coverage) sit next to policy metrics
(eligibility violations, empty-recommendation rate, eligibility rate).
"""
import numpy as np
import pandas as pd

from .config import NEGATIVES_PER_POSITIVE, RANDOM_SEED, TEST_FRACTION, TOP_K
from .eligibility import eligible_mask
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame, build_training_examples
from .model import build_ranker


def time_split(interactions: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split by time at the (1 - test_fraction) timestamp quantile; returns (train, test, cutoff)."""
    interactions = interactions.sort_values("timestamp")
    cutoff = interactions["timestamp"].quantile(1 - test_fraction)
    return interactions[interactions["timestamp"] <= cutoff], interactions[interactions["timestamp"] > cutoff], cutoff


def recall_at_k(actual_items, recommended_items, k=TOP_K):
    """Fraction of the actual items that appear in the first k recommendations."""
    actual = set(actual_items)
    recommended = set(recommended_items[:k])
    return len(actual & recommended) / len(actual) if actual else 0.0


def eligibility_rate(total_candidates, eligible_candidates):
    """Share of candidates that are eligible (0 when there are none)."""
    return eligible_candidates / total_candidates if total_candidates else 0.0


def strategy_metrics(scored: pd.DataFrame, score_column: str, apply_eligibility: bool, catalog_size: int, k: int) -> dict:
    """Quality and policy metrics for one strategy over all test requests.

    Each request has one target (the product actually applied for). With apply_eligibility the
    ineligible rows are dropped before ranking; without it they can occupy top-k slots.
    """
    request_ids = scored["request_id"].unique()
    pool = scored[scored["eligible"]] if apply_eligibility else scored
    pool = pool.sort_values(["request_id", score_column, "product_id"], ascending=[True, False, True])
    # Rank within each request by score (product_id breaks ties).
    pool = pool.assign(rank=pool.groupby("request_id").cumcount() + 1)
    top_k = pool[pool["rank"] <= k]

    # Target's rank per request; inf when it was filtered out (e.g. target ineligible), i.e. a miss.
    target_rank = pool.loc[pool["is_target"]].set_index("request_id")["rank"].reindex(request_ids, fill_value=np.inf)
    hits = target_rank <= k
    shown_per_request = top_k.groupby("request_id").size().reindex(request_ids, fill_value=0)
    # Policy: count ineligible products that made it into each request's top-k.
    violations_per_request = (~top_k["eligible"]).groupby(top_k["request_id"]).sum().reindex(request_ids, fill_value=0)
    return {
        f"recall@{k}": round(float(hits.mean()), 4),
        # One relevant item per request, so NDCG = 1 / log2(rank + 1) for a hit, else 0.
        f"ndcg@{k}": round(float((hits / np.log2(target_rank + 1)).mean()), 4),
        "coverage": round(top_k["product_id"].nunique() / catalog_size, 4),
        # Share of all shown slots that were ineligible, and share of requests with at least one.
        "eligibility_violation_rate": round(float((~top_k["eligible"]).mean()), 4),
        "requests_with_violation": round(float((violations_per_request > 0).mean()), 4),
        "empty_recommendation_rate": round(float((shown_per_request == 0).mean()), 4),
    }


def evaluate(interactions, users, products, k: int = TOP_K) -> dict:
    """Fit a ranker on the training period, then compare four strategies on later applications."""
    train_events, test_events, cutoff = time_split(interactions)

    training_requests = build_training_examples(train_events, users, products, NEGATIVES_PER_POSITIVE, RANDOM_SEED)
    training_frame = build_feature_frame(training_requests, users, products, train_events)
    ranker = build_ranker(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    ranker.fit(training_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], training_frame["label"])

    # One request per test application: the user is scored against the full catalog at that moment.
    test_applications = test_events[test_events["label"] == 1].reset_index(drop=True)
    test_applications["request_id"] = np.arange(len(test_applications))
    test_requests = test_applications[["request_id", "user_id"]].merge(products[["product_id"]], how="cross")
    test_requests["request_time"] = test_requests["request_id"].map(test_applications["timestamp"])

    # Full history is passed, but features only count events before each request_time (no leakage).
    scored = build_feature_frame(test_requests[["user_id", "product_id", "request_time"]], users, products, interactions)
    scored["request_id"] = test_requests["request_id"].to_numpy()
    scored["is_target"] = scored["product_id"].to_numpy() == test_requests["request_id"].map(test_applications["product_id"]).to_numpy()
    scored["eligible"] = eligible_mask(scored).to_numpy()
    # One score column per strategy; popularity is the point-in-time event count feature.
    scored["score_ranker"] = ranker.predict_proba(scored[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    scored["score_popularity"] = scored["product_popularity"]
    scored["score_random"] = np.random.default_rng(RANDOM_SEED).random(len(scored))

    catalog_size = len(products)
    # name -> (score column, apply eligibility filter?). "ranker_without_eligibility" shows violations.
    strategies = {
        "random_eligible": ("score_random", True),
        "popularity_eligible": ("score_popularity", True),
        "ranker_without_eligibility": ("score_ranker", False),
        "eligibility_then_ranking": ("score_ranker", True),
    }
    results = {
        name: strategy_metrics(scored, score_column, apply_eligibility, catalog_size, k)
        for name, (score_column, apply_eligibility) in strategies.items()
    }

    # Policy view: how restrictive the rules are, per test request and across risk profiles and incomes.
    eligible_counts = scored.groupby("request_id")["eligible"].sum()
    all_pairs = users.merge(products, how="cross")
    all_pairs["eligible"] = eligible_mask(all_pairs)
    return {
        "protocol": "temporal holdout: train on the earliest 80% of events, rank the full catalog "
                    "for each application in the last 20%",
        "k": k,
        "cutoff": cutoff.isoformat(),
        "train_events": int(len(train_events)),
        "test_requests": int(len(test_applications)),
        "applied_products_that_were_ineligible": int((scored["is_target"] & ~scored["eligible"]).sum()),
        "mean_eligible_products_per_request": round(float(eligible_counts.mean()), 2),
        "strategies": results,
        "eligibility_rate_by_risk_profile": all_pairs.groupby("risk_profile")["eligible"].mean().round(4).to_dict(),
        "eligibility_rate_by_income_quartile": all_pairs.groupby(
            pd.qcut(all_pairs["monthly_income"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]), observed=True
        )["eligible"].mean().round(4).to_dict(),
    }
