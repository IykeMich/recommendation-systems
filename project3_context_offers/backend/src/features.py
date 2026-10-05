"""Feature engineering shared by training, evaluation and serving.

Every feature is computed "as of" the request time: user and offer history only
counts events strictly before it. Training, evaluation and the API all call
build_feature_frame, so the features cannot drift between them.
"""
import numpy as np
import pandas as pd

# Features the ranker sees. Context-dependent ones are listed separately so the
# "remove context" experiment can drop exactly these.
NUMERIC_FEATURES = [
    # user history (point-in-time)
    "interaction_count",
    "positive_count",
    "positive_rate",
    "category_affinity",
    "preferred_category_match",
    # offer
    "offer_popularity",
    "discount_pct",
    "min_spend",
    # context
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "region_match",
    "channel_match",
]

CATEGORICAL_FEATURES = [
    "segment",
    "category",
    "channel",
    "device_type",
    "region",
    "category_x_daypart",
    "category_x_device",
]

CONTEXT_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "region_match",
    "channel_match",
    "device_type",
    "region",
    "category_x_daypart",
    "category_x_device",
]

DAYPART_BOUNDARIES = [(0, 6, "night"), (6, 12, "morning"), (12, 18, "afternoon"), (18, 24, "evening")]


def daypart_for_hour(hour: int) -> str:
    """Bucket an hour (0-23) into night/morning/afternoon/evening (used in the category crosses)."""
    return next(name for start, end, name in DAYPART_BOUNDARIES if start <= hour < end)


def add_context_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic time encodings: hour 23 and hour 0 end up close together."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # pandas dayofweek has Monday = 0, so 5 and 6 are Saturday and Sunday.
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["daypart"] = df["hour"].map(daypart_for_hour)
    return df


def add_time_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Derive the request-time context (hour, day of week) from event timestamps."""
    interactions = interactions.copy()
    interactions["hour"] = interactions["timestamp"].dt.hour
    interactions["day_of_week"] = interactions["timestamp"].dt.dayofweek
    return add_context_encodings(interactions)


def _user_history_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Counts of the user's events strictly before each request."""
    # Point-in-time join: pair each request row with ALL of that user's events, then keep only
    # events strictly before the request time. Later events would leak the answer.
    prior_events = rows[["row_id", "user_id", "request_time", "category"]].merge(
        history[["user_id", "timestamp", "label", "category"]],
        on="user_id",
        suffixes=("", "_history"),
    )
    prior_events = prior_events[prior_events["timestamp"] < prior_events["request_time"]]
    # Was the earlier event on an offer in the candidate's category? Feeds category_affinity.
    prior_events["same_category"] = prior_events["category"] == prior_events["category_history"]

    stats = prior_events.groupby("row_id").agg(
        interaction_count=("label", "size"),
        positive_count=("label", "sum"),
        same_category_count=("same_category", "sum"),
    )
    # Rows with no earlier events (new user, first request) get zero counts.
    stats = stats.reindex(rows["row_id"], fill_value=0)
    interaction_count = stats["interaction_count"].to_numpy(dtype=float)
    # Divide by at least 1 so the rates are 0 rather than NaN when there is no history.
    safe_count = np.clip(interaction_count, 1, None)
    return pd.DataFrame(
        {
            "interaction_count": interaction_count,
            "positive_count": stats["positive_count"].to_numpy(dtype=float),
            "positive_rate": stats["positive_count"].to_numpy(dtype=float) / safe_count,
            "category_affinity": stats["same_category_count"].to_numpy(dtype=float) / safe_count,
        },
        index=rows.index,
    )


def _offer_popularity(rows: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """Engagements each offer received strictly before the request time."""
    # For each offer, the sorted times it was clicked or redeemed.
    positive_times = {
        offer_id: np.sort(group["timestamp"].to_numpy())
        for offer_id, group in history[history["label"] == 1].groupby("offer_id")
    }
    popularity = pd.Series(0.0, index=rows.index)
    for offer_id, offer_rows in rows.groupby("offer_id"):
        times = positive_times.get(offer_id)
        if times is not None:
            # searchsorted(side='left') counts engagement times strictly before each request_time,
            # giving point-in-time popularity for all of this offer's rows in one call.
            popularity.loc[offer_rows.index] = np.searchsorted(
                times, offer_rows["request_time"].to_numpy(), side="left"
            )
    return popularity


def channel_matches_device(channel: pd.Series, device_type: pd.Series) -> pd.Series:
    """1 if the offer's channel can be shown on the device: 'both' anywhere, 'mobile' on phones and
    tablets, 'web' on web only. Used as a candidate rule and as the channel_match feature."""
    return (
        channel.eq("both")
        | (channel.eq("mobile") & device_type.isin(["mobile", "tablet"]))
        | (channel.eq("web") & device_type.eq("web"))
    ).astype(int)


def build_feature_frame(
    requests: pd.DataFrame,
    offers: pd.DataFrame,
    users: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (request, candidate offer) with every model feature.

    `requests` needs: user_id, offer_id, request_time, device_type, region, hour, day_of_week.
    `history` is interactions in the internal schema; only events before each
    request_time are used, which is what keeps the features leakage-free.
    """
    rows = requests.reset_index(drop=True).copy()
    # row_id is a stable key so rows can be restored to request order after the merges below.
    rows["row_id"] = np.arange(len(rows))

    # Attach offer attributes (offer region renamed so it doesn't clash with the request's region)
    # and the user's profile; users without a profile get segment 'unknown'.
    offer_columns = offers[["offer_id", "title", "category", "discount_pct", "min_spend", "channel", "region"]]
    rows = rows.merge(offer_columns.rename(columns={"region": "offer_region"}), on="offer_id", how="left")
    rows = rows.merge(users[["user_id", "segment", "preferred_category"]], on="user_id", how="left")
    rows["segment"] = rows["segment"].fillna("unknown")
    rows = rows.sort_values("row_id").reset_index(drop=True)

    rows = add_context_encodings(rows)

    # History events need the category of the offer they refer to, for category_affinity.
    history_with_category = history.merge(offers[["offer_id", "category"]], on="offer_id", how="left")
    rows = rows.join(_user_history_features(rows, history_with_category))
    rows["offer_popularity"] = _offer_popularity(rows, history)

    # Match flags and crosses. A linear model can't combine features itself, so crosses like
    # 'Home|evening' let it learn that a category does better in one daypart or on one device.
    rows["preferred_category_match"] = rows["category"].eq(rows["preferred_category"]).astype(int)
    rows["region_match"] = (rows["offer_region"].eq("all") | rows["offer_region"].eq(rows["region"])).astype(int)
    rows["channel_match"] = channel_matches_device(rows["channel"], rows["device_type"])
    rows["category_x_daypart"] = rows["category"] + "|" + rows["daypart"]
    rows["category_x_device"] = rows["category"] + "|" + rows["device_type"]
    return rows


def build_training_examples(
    interactions: pd.DataFrame,
    offers: pd.DataFrame,
    negatives_per_positive: int,
    random_seed: int,
) -> pd.DataFrame:
    """Positives: offers the user clicked/redeemed, in the context it happened.
    Negatives: other offers sampled for that same user and context."""
    # Each click/redeem becomes a positive request, with the context and time it happened in.
    positives = add_time_features(interactions[interactions["label"] == 1])
    context_columns = ["user_id", "device_type", "region", "hour", "day_of_week"]
    positive_requests = positives[context_columns + ["offer_id"]].assign(
        request_time=positives["timestamp"], label=1
    )

    # Every offer each user engaged with, so those offers are never used as their negatives.
    engaged_by_user = interactions[interactions["label"] == 1].groupby("user_id")["offer_id"].agg(set)
    offer_ids = offers["offer_id"].to_numpy()
    random_generator = np.random.default_rng(random_seed)

    # Copy each positive request N times and swap in a random catalog offer. Negatives share the
    # positive's user, time and context, so the model must learn which *offer* fits that context.
    negative_requests = positive_requests.loc[
        positive_requests.index.repeat(negatives_per_positive)
    ].copy()
    sampled_offer_ids = random_generator.choice(offer_ids, size=len(negative_requests))
    # Resample any draw that hits an offer this user engaged with.
    for position, (user_id, offer_id) in enumerate(zip(negative_requests["user_id"], sampled_offer_ids)):
        while offer_id in engaged_by_user.get(user_id, ()):
            offer_id = random_generator.choice(offer_ids)
        sampled_offer_ids[position] = offer_id
    negative_requests["offer_id"] = sampled_offer_ids
    negative_requests["label"] = 0

    return pd.concat([positive_requests, negative_requests], ignore_index=True)


def build_exposure_examples(interactions: pd.DataFrame) -> pd.DataFrame:
    """The guide's literal setup: every logged event is a row, labelled engaged or not.
    Kept for the comparison experiment."""
    events = add_time_features(interactions)
    return events[["user_id", "offer_id", "device_type", "region", "hour", "day_of_week", "label"]].assign(
        request_time=events["timestamp"]
    )
