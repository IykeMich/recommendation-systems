"""Feature engineering shared by training, evaluation and serving.

User history and product popularity count only events strictly before each
request time, so no future information leaks into a prediction.
"""
import numpy as np
import pandas as pd

from .eligibility import eligible_mask

# Feature lists are saved with the model (feature_columns.json) so serving uses the same columns.
NUMERIC_FEATURES = [
    "previously_engaged",
    "prior_product_events",
    "category_affinity",
    "product_popularity",
    "income_headroom",
    "existing_products",
]

CATEGORICAL_FEATURES = [
    "category",
    "risk_level",
    "risk_fit",
    "employment_x_category",
    "age_band",
]

USER_COLUMNS = ["user_id", "age_band", "monthly_income", "employment", "risk_profile", "existing_products"]
PRODUCT_COLUMNS = ["product_id", "name", "category", "risk_level", "min_income"]


def _history_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Per-row counts of the user's earlier events: on this product, and in this category.

    Point-in-time: only history events strictly before each row's request_time are counted.
    """
    # Join every request row to ALL of that user's history events, then drop events that happened
    # at or after the request time. This is what prevents future information leaking in.
    prior = rows[["row_id", "user_id", "product_id", "category", "request_time"]].merge(
        history[["user_id", "product_id", "category", "timestamp"]],
        on="user_id",
        suffixes=("", "_history"),
    )
    prior = prior[prior["timestamp"] < prior["request_time"]]
    prior["same_product"] = prior["product_id"] == prior["product_id_history"]
    prior["same_category"] = prior["category"] == prior["category_history"]
    # Aggregate back to one row per request; rows with no prior events get 0 via reindex.
    stats = prior.groupby("row_id").agg(
        user_event_count=("same_product", "size"),
        prior_product_events=("same_product", "sum"),
        same_category_events=("same_category", "sum"),
    ).reindex(rows["row_id"], fill_value=0)

    user_event_count = stats["user_event_count"].to_numpy(dtype=float)
    return pd.DataFrame({
        "prior_product_events": stats["prior_product_events"].to_numpy(dtype=float),
        "previously_engaged": (stats["prior_product_events"].to_numpy() > 0).astype(int),
        # Share of the user's past events that were in this product's category (avoid divide-by-zero).
        "category_affinity": stats["same_category_events"].to_numpy(dtype=float) / np.clip(user_event_count, 1, None),
    }, index=rows.index)


def _product_popularity(rows: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """Number of events on each product strictly before each row's request_time (any user)."""
    # Sorted event times per product; searchsorted then counts how many fall before request_time
    # (default side='left' excludes events at exactly that time), without a row-by-row loop.
    event_times = {
        product_id: np.sort(group["timestamp"].to_numpy())
        for product_id, group in history.groupby("product_id")
    }
    popularity = pd.Series(0.0, index=rows.index)
    for product_id, product_rows in rows.groupby("product_id"):
        times = event_times.get(product_id)
        if times is not None:
            popularity.loc[product_rows.index] = np.searchsorted(times, product_rows["request_time"].to_numpy())
    return popularity


def build_feature_frame(requests, users, products, history) -> pd.DataFrame:
    """One row per (user, product, request_time) with every model feature.

    `users` may be the dataset profiles or a single what-if profile.
    """
    # row_id keeps a stable key so merges and aggregations can be aligned back to the input order.
    rows = requests.reset_index(drop=True).copy()
    rows["row_id"] = np.arange(len(rows))
    rows = rows.merge(users[USER_COLUMNS], on="user_id", how="left")
    rows = rows.merge(products[PRODUCT_COLUMNS], on="product_id", how="left")
    rows = rows.sort_values("row_id").reset_index(drop=True)

    # History needs each product's category for the category-affinity feature.
    history_with_category = history.merge(products[["product_id", "category"]], on="product_id", how="left")
    rows = rows.join(_history_features(rows, history_with_category))
    rows["product_popularity"] = _product_popularity(rows, history)

    # Log scale so a large income gap doesn't dominate; negative means income is below the minimum.
    rows["income_headroom"] = np.log1p(rows["monthly_income"].clip(lower=0)) - np.log1p(rows["min_income"])
    # Crossed categorical features let a linear model learn interactions (e.g. low|high risk fit).
    rows["risk_fit"] = rows["risk_profile"].fillna("unknown") + "|" + rows["risk_level"]
    rows["employment_x_category"] = rows["employment"].fillna("unknown") + "|" + rows["category"]
    return rows


def build_training_examples(interactions, users, products, negatives_per_positive, random_seed) -> pd.DataFrame:
    """Positives: applications. Negatives: other *eligible* products sampled for the
    same user and moment. The ranker only ever learns to order eligible products."""
    positives = interactions[interactions["label"] == 1][["user_id", "product_id", "timestamp"]]
    positives = positives.rename(columns={"timestamp": "request_time"}).assign(label=1)

    # Every user x product pair, filtered by the vectorised rules, gives each user's eligible list.
    pairs = users[["user_id", "monthly_income", "risk_profile"]].merge(
        products[["product_id", "min_income", "risk_level"]], how="cross"
    )
    eligible_by_user = pairs[eligible_mask(pairs)].groupby("user_id")["product_id"].agg(list)

    # For each application, sample negatives (with replacement) from that user's OTHER eligible
    # products at the same request time. Ineligible products are never used as negatives.
    random_generator = np.random.default_rng(random_seed)
    negative_rows = []
    for positive in positives.itertuples(index=False):
        choices = [product_id for product_id in eligible_by_user.get(positive.user_id, []) if product_id != positive.product_id]
        if not choices:
            continue
        for product_id in random_generator.choice(choices, size=negatives_per_positive):
            negative_rows.append((positive.user_id, product_id, positive.request_time, 0))
    negatives = pd.DataFrame(negative_rows, columns=["user_id", "product_id", "request_time", "label"])
    return pd.concat([positives, negatives], ignore_index=True)
