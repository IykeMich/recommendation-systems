"""Regenerate the synthetic offers.csv and offer_interactions.csv for the current users.csv (seeded, reproducible).

Run from backend/: python scripts/generate_offer_data.py
users.csv is the input (a copy of Project 2's shoppers) and is never modified. The previous
offers.csv and offer_interactions.csv are copied to data/archive/ first (once).

The data keeps the design of the original Project 3 offer dataset:
- Offers span the same categories as the shoppers' preferred categories, with more offers in
  categories more shoppers prefer. Discount, minimum spend, channel, region and expiry are random.
- Each shopper has 4-10 events; 80% are offers in their preferred category, the rest are random.
- Whether an event is an impression, click or redeem is random and independent of the context,
  so engagement is flat across hour, day, device, region and discount.
- Events use the shopper's own device and region. Offer region/channel rules are not applied to
  the log, so it also contains offers shown outside their region or channel.
"""
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR  # noqa: E402

SEED = 42
NUM_OFFERS = 120
MIN_OFFERS_PER_CATEGORY = 3
EVENTS_PER_USER = (4, 10)           # inclusive range
SHARE_PREFERRED_CATEGORY = 0.80     # of a shopper's events
EVENT_TYPE_SHARES = {"impression": 0.555, "click": 0.346, "redeem": 0.099}

TITLE_PREFIXES = ["Bonus", "Save", "Flash", "Bundle"]
DISCOUNTS = [5, 10, 15, 20, 25, 30]
MIN_SPENDS = [0, 25, 50, 100, 250]
CHANNELS = ["both", "web", "mobile"]
OFFER_REGIONS = ["all", "Lagos", "Abuja", "Port Harcourt"]
EXPIRY_RANGE = (date(2026, 9, 23), date(2026, 12, 31))
YEAR_START = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
YEAR_END = int(datetime(2025, 12, 29, tzinfo=timezone.utc).timestamp())


def offers_per_category(users: pd.DataFrame) -> dict:
    """Split NUM_OFFERS across preferred categories in proportion to shoppers (largest remainder)."""
    counts = users.preferred_category.value_counts().sort_index()
    spare = NUM_OFFERS - MIN_OFFERS_PER_CATEGORY * len(counts)
    exact = counts / counts.sum() * spare
    allocation = np.floor(exact).astype(int)
    leftover = spare - allocation.sum()
    for category in (exact - allocation).sort_values(ascending=False).index[:leftover]:
        allocation[category] += 1
    return (allocation + MIN_OFFERS_PER_CATEGORY).to_dict()


def build_offers(rng, users: pd.DataFrame) -> pd.DataFrame:
    """The offer catalog: categories in shuffled order, random attributes per offer."""
    categories = [c for c, n in offers_per_category(users).items() for _ in range(n)]
    rng.shuffle(categories)
    expiry_days = (EXPIRY_RANGE[1] - EXPIRY_RANGE[0]).days
    rows = []
    for number, category in enumerate(categories, start=1):
        rows.append({
            "offer_id": f"O{number:05d}",
            "title": f"{rng.choice(TITLE_PREFIXES)} {category} Offer {number}",
            "category": category,
            "discount_pct": int(rng.choice(DISCOUNTS)),
            "min_spend": int(rng.choice(MIN_SPENDS)),
            "channel": str(rng.choice(CHANNELS)),
            "region": str(rng.choice(OFFER_REGIONS)),
            "active": True,
            "expires_at": (EXPIRY_RANGE[0] + timedelta(days=int(rng.integers(0, expiry_days + 1)))).isoformat(),
        })
    return pd.DataFrame(rows)


def build_interactions(rng, users: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """4-10 events per shopper, mostly in their preferred category, with random event types."""
    by_category = offers.groupby("category").offer_id.apply(list).to_dict()
    all_offers = list(offers.offer_id)
    event_types = list(EVENT_TYPE_SHARES)
    event_probabilities = list(EVENT_TYPE_SHARES.values())
    rows = []
    for user in users.itertuples():
        for _ in range(int(rng.integers(EVENTS_PER_USER[0], EVENTS_PER_USER[1] + 1))):
            pool = by_category[user.preferred_category] if rng.random() < SHARE_PREFERRED_CATEGORY else all_offers
            rows.append({
                "user_id": user.user_id,
                "offer_id": str(rng.choice(pool)),
                "event_type": str(rng.choice(event_types, p=event_probabilities)),
                "timestamp": int(rng.integers(YEAR_START, YEAR_END)),
                "device_type": user.device_type,
                "region": user.region,
            })
    return pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    users = pd.read_csv(DATA_DIR / "users.csv")
    offers = build_offers(rng, users)
    interactions = build_interactions(rng, users, offers)

    # Keep the previous files (once) so the regeneration can be undone.
    archive = DATA_DIR / "archive"
    archive.mkdir(exist_ok=True)
    for name in ["offers.csv", "offer_interactions.csv"]:
        if (DATA_DIR / name).exists() and not (archive / name).exists():
            shutil.copy2(DATA_DIR / name, archive / name)
    offers.to_csv(DATA_DIR / "offers.csv", index=False)
    interactions.to_csv(DATA_DIR / "offer_interactions.csv", index=False)

    print(f"offers: {len(offers)}  interactions: {len(interactions)}  users: {interactions.user_id.nunique()}")
    print("offers per category:", offers.category.value_counts().to_dict())
    print("event types:", interactions.event_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
