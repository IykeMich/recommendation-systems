"""Loading the raw CSVs and mapping them to the internal schema (first step of the pipeline).

Also stores feedback interactions captured by the API in a separate CSV, so the supplied
dataset is never modified.
"""
import pandas as pd

from .config import DATA_DIR, INTERACTION_MAP, NEW_INTERACTIONS_PATH, POSITIVE_EVENT_TYPES, PRODUCT_MAP, USER_MAP


def _to_internal(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Rename raw CSV columns to internal names using an {internal: raw} map from config."""
    return df.rename(columns={source: target for target, source in column_map.items()})


def load_data():
    """Read the three raw CSVs exactly as supplied."""
    users = pd.read_csv(DATA_DIR / "users.csv")
    products = pd.read_csv(DATA_DIR / "financial_products.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")
    return users, products, interactions


def prepare_users(raw_users: pd.DataFrame) -> pd.DataFrame:
    """User profiles in the internal schema."""
    return _to_internal(raw_users, USER_MAP)


def prepare_products(raw_products: pd.DataFrame) -> pd.DataFrame:
    """Product catalog in the internal schema."""
    return _to_internal(raw_products, PRODUCT_MAP)


def prepare_interactions(raw_interactions: pd.DataFrame) -> pd.DataFrame:
    """Interactions with UTC timestamps, a binary `label` (1 = apply) and sorted by time."""
    interactions = _to_internal(raw_interactions, INTERACTION_MAP)
    # Raw timestamps are Unix seconds; unparseable values become NaT instead of raising.
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="s", utc=True, errors="coerce")
    interactions["label"] = interactions["event_type"].isin(POSITIVE_EVENT_TYPES).astype(int)
    return interactions.sort_values("timestamp").reset_index(drop=True)


def load_new_interactions() -> pd.DataFrame:
    """Feedback interactions recorded after the original dataset (empty frame if none yet)."""
    if not NEW_INTERACTIONS_PATH.exists():
        return pd.DataFrame(columns=list(INTERACTION_MAP.values()))
    return pd.read_csv(NEW_INTERACTIONS_PATH)


def append_new_interaction(raw_interaction: dict) -> None:
    """Append one raw-schema interaction to the new-interactions CSV (header only on first write)."""
    row = pd.DataFrame([raw_interaction], columns=list(INTERACTION_MAP.values()))
    row.to_csv(NEW_INTERACTIONS_PATH, mode="a", header=not NEW_INTERACTIONS_PATH.exists(), index=False)


def load_prepared(include_new_interactions: bool = True):
    """Users, products and interactions in the internal schema."""
    raw_users, raw_products, raw_interactions = load_data()
    if include_new_interactions:
        new_interactions = load_new_interactions()
        if not new_interactions.empty:
            raw_interactions = pd.concat([raw_interactions, new_interactions], ignore_index=True)
    return prepare_users(raw_users), prepare_products(raw_products), prepare_interactions(raw_interactions)
