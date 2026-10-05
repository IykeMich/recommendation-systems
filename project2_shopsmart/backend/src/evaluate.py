"""Leave-one-positive-out holdout evaluation (section 9 of the guide)."""
import numpy as np
import pandas as pd

from .config import DEFAULT_TOP_K, POSITIVE_EVENT_TYPES
from .preprocess import build_user_item
from .recommender import AdaptiveHybridRecommender, ItemItemRecommender, popular_items


def holdout_split(weighted_events: pd.DataFrame, min_positive_items: int = 2):
    """Hide each eligible user's most recent positive item.

    Every event for the hidden (user, item) pair is removed from training,
    otherwise a leftover view would leak the answer.
    """
    # Only users with >= 2 distinct positive items are evaluated, so they keep
    # at least one positive in training and stay known to the model.
    positives = weighted_events[weighted_events["event_type"].isin(POSITIVE_EVENT_TYPES)]
    positive_item_counts = positives.groupby("user_id")["item_id"].nunique()
    eligible_users = positive_item_counts[positive_item_counts >= min_positive_items].index

    # Sort by time and keep the last positive row per user: that (user, item) pair is the test target.
    test_events = (
        positives[positives["user_id"].isin(eligible_users)]
        .sort_values("timestamp")
        .groupby("user_id")
        .tail(1)[["user_id", "item_id"]]
    )
    # Leakage guard: drop *every* event (views and carts too) for each hidden pair, not just the positive.
    hidden_pairs = set(zip(test_events["user_id"], test_events["item_id"]))
    is_hidden = [
        (user_id, item_id) in hidden_pairs
        for user_id, item_id in zip(weighted_events["user_id"], weighted_events["item_id"])
    ]
    train_events = weighted_events[~np.array(is_hidden)]
    return train_events, test_events


def recall_at_k(recommend_fn, test_events: pd.DataFrame, k: int = 10) -> float:
    """Share of test users with at least one target item in their top-k (simpler form of ranking_metrics)."""
    hits = 0
    total = 0
    for user_id, group in test_events.groupby("user_id"):
        target_items = set(group["item_id"])
        recs = recommend_fn(user_id, k=k)
        recommended = {r["item_id"] for r in recs}
        if not target_items:
            continue
        total += 1
        if recommended.intersection(target_items):
            hits += 1
    return hits / total if total else 0.0


def ranking_metrics(recommend_fn, test_events: pd.DataFrame, catalog_size: int, k: int):
    """Recall@K, Precision@K, NDCG@K and catalog coverage for one hidden item per user."""
    hits = 0
    ndcg_total = 0.0
    recommended_items = set()
    for user_id, hidden_item_id in zip(test_events["user_id"], test_events["item_id"]):
        ranked_item_ids = [r["item_id"] for r in recommend_fn(user_id, k=k)]
        recommended_items.update(ranked_item_ids)
        if hidden_item_id in ranked_item_ids:
            hits += 1
            # With a single relevant item the ideal DCG is 1, so NDCG = 1 / log2(rank + 1)
            # (index is 0-based, hence + 2). A hit at rank 1 scores 1.0, lower ranks score less.
            ndcg_total += 1.0 / np.log2(ranked_item_ids.index(hidden_item_id) + 2)

    # One hidden item per user: recall = hit rate; precision divides by the k slots shown.
    # Coverage = share of the catalog that appeared in anyone's top-k.
    evaluated_users = len(test_events)
    return {
        f"recall@{k}": round(hits / evaluated_users, 4),
        f"precision@{k}": round(hits / (evaluated_users * k), 4),
        f"ndcg@{k}": round(ndcg_total / evaluated_users, 4),
        "coverage": round(len(recommended_items) / catalog_size, 4),
    }


def evaluate(weighted_events: pd.DataFrame, k: int = DEFAULT_TOP_K, products: pd.DataFrame = None) -> dict:
    """Split, train on the training part only, and compare hybrid / item-item / popularity.

    The hybrid row is included when the product catalog is passed in (it needs product attributes).
    """
    train_events, test_events = holdout_split(weighted_events)
    train_matrix = build_user_item(train_events)
    model = ItemItemRecommender().fit(train_matrix)
    catalog_size = len(products) if products is not None else weighted_events["item_id"].nunique()

    # Items each user interacted with in training, so the baseline also skips seen items (a fair comparison).
    seen_by_user = train_events.groupby("user_id")["item_id"].agg(set).to_dict()

    def recommend_popular(user_id, k):
        """Baseline: the most popular training items this user hasn't already seen."""
        return popular_items(train_events, k=k, exclude=seen_by_user.get(user_id, ()))

    results = {
        "protocol": "leave-one-positive-out (latest cart/purchase per user)",
        "k": k,
        "evaluated_users": int(len(test_events)),
    }
    if products is not None:
        hybrid = AdaptiveHybridRecommender(products).fit(train_matrix)
        results["hybrid"] = ranking_metrics(hybrid.recommend, test_events, catalog_size, k)
    results["item_item"] = ranking_metrics(model.recommend, test_events, catalog_size, k)
    results["popularity_baseline"] = ranking_metrics(recommend_popular, test_events, catalog_size, k)
    return results
