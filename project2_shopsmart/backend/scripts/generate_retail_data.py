"""Regenerate the synthetic users.csv and interactions.csv for the current products.csv (seeded, reproducible).

Run from backend/: python scripts/generate_retail_data.py
The previous files are copied to data/archive/ first. products.csv is the input and is never modified.

Shoppers behave like real ones, which is what collaborative filtering needs to learn from:
most activity is in one or two favourite subcategories of a preferred category, some in a
second category, and a little random exploration. Each product "touch" is a view that may
lead to a cart and then a purchase (the view -> cart -> purchase funnel).
"""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, EVENT_WEIGHTS  # noqa: E402

SEED = 42
NUM_USERS = 1000
TOUCHES_PER_USER = (13, 28)          # distinct product touches per shopper (inclusive range)
SHARE_FAVOURITE_SUBCATEGORIES = 0.60  # of touches, inside 1-2 favourite subcategories
SHARE_PREFERRED_CATEGORY = 0.10       # elsewhere in the preferred category
SHARE_SECONDARY_CATEGORY = 0.18       # in a second, segment-related category
# The remainder is random exploration across the whole catalog.
CART_RATE = 0.28                      # P(cart | view)
PURCHASE_RATE = 0.50                  # P(purchase | cart)
MIN_TOUCHES_PER_PRODUCT = 3           # every product gets some signal (no unreachable items)

SEGMENTS = ["creator", "student", "fitness", "family", "professional", "business"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
DEVICES = ["mobile", "tablet", "web"]
REGIONS = ["Lagos", "Port Harcourt", "Abuja", "Ibadan", "Enugu"]

# Categories each segment leans towards (used for both preferred and secondary categories).
SEGMENT_AFFINITY = {
    "creator": ["Electronics", "Beauty", "Apparel & Footwear"],
    "student": ["Office & School", "Electronics", "Food & Beverage", "Apparel & Footwear"],
    "fitness": ["Apparel & Footwear", "Food & Beverage", "Personal Care"],
    "family": ["Food & Beverage", "Baby Care", "Home Care", "Toys & Games", "Pet Care"],
    "professional": ["Electronics", "Appliances", "Home & Kitchen", "Travel"],
    "business": ["Office & School", "Electronics", "Home & Furniture", "Travel"],
}
YEAR_START = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
YEAR_END = int(datetime(2025, 12, 29, tzinfo=timezone.utc).timestamp())


def pick_category(rng, segment, weights_by_category, exclude=None):
    """A category, biased (3x) towards the segment's affinities and by catalog size."""
    categories = [c for c in weights_by_category.index if c != exclude]
    weights = np.array([
        weights_by_category[c] * (3.0 if c in SEGMENT_AFFINITY[segment] else 1.0) for c in categories
    ])
    return str(rng.choice(categories, p=weights / weights.sum()))


def pick_product(rng, pool, appeal, taken):
    """One product from the pool, weighted by its appeal, avoiding products already touched."""
    candidates = [item_id for item_id in pool if item_id not in taken]
    if not candidates:
        return None
    weights = np.array([appeal[item_id] for item_id in candidates])
    return str(rng.choice(candidates, p=weights / weights.sum()))


def touch_events(rng, user, item_id):
    """The funnel for one product touch: always a view, maybe a cart, maybe a purchase."""
    timestamp = int(rng.integers(YEAR_START, YEAR_END))
    event_types = ["view"]
    if rng.random() < CART_RATE:
        event_types.append("cart")
        if rng.random() < PURCHASE_RATE:
            event_types.append("purchase")
    rows = []
    for event_type in event_types:
        rows.append({
            "user_id": user["user_id"],
            "item_id": item_id,
            "event_type": event_type,
            "event_value": int(EVENT_WEIGHTS[event_type]),
            "timestamp": timestamp,
            "device_type": user["device_type"],
            "region": user["region"],
        })
        timestamp += int(rng.integers(60, 3 * 86400))  # later funnel steps happen after the view
    return rows


def main():
    rng = np.random.default_rng(SEED)
    products = pd.read_csv(DATA_DIR / "products.csv")
    # Square root of category size so big categories lead without drowning out small ones.
    category_weights = np.sqrt(products.category.value_counts())
    by_category = products.groupby("category").item_id.apply(list).to_dict()
    by_subcategory = products.groupby(["category", "subcategory"]).item_id.apply(list).to_dict()
    # Skewed product appeal: a few bestsellers per shelf, a long tail of rarer picks.
    appeal = dict(zip(products.item_id, rng.pareto(1.5, len(products)) + 0.2))

    users = []
    for number in range(1, NUM_USERS + 1):
        segment = str(rng.choice(SEGMENTS))
        users.append({
            "user_id": f"R{number:05d}",
            "segment": segment,
            "age_band": str(rng.choice(AGE_BANDS)),
            "preferred_category": pick_category(rng, segment, category_weights),
            "device_type": str(rng.choice(DEVICES)),
            "region": str(rng.choice(REGIONS)),
        })

    rows = []
    for user in users:
        preferred = user["preferred_category"]
        secondary = pick_category(rng, user["segment"], category_weights, exclude=preferred)
        subcategories = [s for (c, s) in by_subcategory if c == preferred]
        favourite_count = min(len(subcategories), int(rng.integers(1, 3)))
        favourites = rng.choice(subcategories, size=favourite_count, replace=False)
        favourite_pool = [i for s in favourites for i in by_subcategory[(preferred, s)]]

        taken = set()
        for _ in range(int(rng.integers(TOUCHES_PER_USER[0], TOUCHES_PER_USER[1] + 1))):
            draw = rng.random()
            if draw < SHARE_FAVOURITE_SUBCATEGORIES:
                pool = favourite_pool
            elif draw < SHARE_FAVOURITE_SUBCATEGORIES + SHARE_PREFERRED_CATEGORY:
                pool = by_category[preferred]
            elif draw < SHARE_FAVOURITE_SUBCATEGORIES + SHARE_PREFERRED_CATEGORY + SHARE_SECONDARY_CATEGORY:
                pool = by_category[secondary]
            else:
                pool = list(products.item_id)
            # Small favourite shelves run out; fall back to the wider preferred category.
            item_id = pick_product(rng, pool, appeal, taken) or pick_product(rng, by_category[preferred], appeal, taken)
            if item_id is None:
                continue
            taken.add(item_id)
            rows.extend(touch_events(rng, user, item_id))

    # Coverage pass: products with too little signal get extra touches from shoppers who prefer
    # that category, so every product is reachable by the collaborative model.
    users_df = pd.DataFrame(users)
    events = pd.DataFrame(rows)
    touched = events.groupby("item_id").user_id.nunique()
    product_category = products.set_index("item_id").category
    for item_id in products.item_id:
        missing = MIN_TOUCHES_PER_PRODUCT - int(touched.get(item_id, 0))
        if missing <= 0:
            continue
        already = set(events.loc[events.item_id == item_id, "user_id"])
        fans = users_df[(users_df.preferred_category == product_category[item_id]) & ~users_df.user_id.isin(already)]
        for _, user in fans.sample(min(missing, len(fans)), random_state=int(rng.integers(1_000_000))).iterrows():
            rows.extend(touch_events(rng, user, item_id))

    events = pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Keep the previous files (once) so the regeneration can be undone.
    archive = DATA_DIR / "archive"
    archive.mkdir(exist_ok=True)
    for name in ["users.csv", "interactions.csv"]:
        if (DATA_DIR / name).exists() and not (archive / name).exists():
            shutil.copy2(DATA_DIR / name, archive / name)
    users_df.to_csv(DATA_DIR / "users.csv", index=False)
    events.to_csv(DATA_DIR / "interactions.csv", index=False)

    print(f"users: {len(users_df)}  interactions: {len(events)}  "
          f"products covered: {events.item_id.nunique()}/{len(products)}")
    print("event types:", events.event_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
