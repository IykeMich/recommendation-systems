"""Data loading: read the CSVs, convert raw interactions to the internal schema (with the
click/redeem label) and append feedback recorded through the API to a separate CSV."""
import pandas as pd

from .config import COLUMN_MAP, DATA_DIR, NEW_INTERACTIONS_PATH, POSITIVE_EVENT_TYPES

RAW_INTERACTION_COLUMNS = list(COLUMN_MAP.values())


def load_data():
    """Read the offer catalog and the original interaction log (raw columns, unmodified)."""
    offers = pd.read_csv(DATA_DIR / "offers.csv")
    interactions = pd.read_csv(DATA_DIR / "offer_interactions.csv")
    return offers, interactions


def load_users() -> pd.DataFrame:
    """User profiles. Project 3 shares its user IDs with Project 2's shoppers."""
    return pd.read_csv(DATA_DIR / "users.csv")


def prepare_offers(offers: pd.DataFrame) -> pd.DataFrame:
    """Parse expiry to a `date` and `active` to bool so the candidate rules can filter on them."""
    offers = offers.copy()
    offers["expires_at"] = pd.to_datetime(offers["expires_at"]).dt.date
    offers["active"] = offers["active"].astype(bool)
    return offers


def to_internal_schema(raw_interactions: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to the stable internal schema and derive the label."""
    interactions = raw_interactions.rename(columns={raw: internal for internal, raw in COLUMN_MAP.items()})
    # Raw timestamps are Unix seconds; unparseable values become NaT instead of raising.
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="s", utc=True, errors="coerce")
    interactions["label"] = interactions["event_type"].isin(POSITIVE_EVENT_TYPES).astype(int)
    # Time order matters: the time split and the API's 'after training' flag rely on it.
    return interactions.sort_values("timestamp").reset_index(drop=True)


def load_new_interactions() -> pd.DataFrame:
    """Events recorded via the API since the dataset was built (an empty frame if none yet)."""
    if not NEW_INTERACTIONS_PATH.exists():
        return pd.DataFrame(columns=RAW_INTERACTION_COLUMNS)
    return pd.read_csv(NEW_INTERACTIONS_PATH)


def append_new_interaction(raw_interaction: dict) -> None:
    """Append one raw-schema event to the new-interactions CSV (header only for a new file)."""
    row = pd.DataFrame([raw_interaction], columns=RAW_INTERACTION_COLUMNS)
    write_header = not NEW_INTERACTIONS_PATH.exists()
    row.to_csv(NEW_INTERACTIONS_PATH, mode="a", header=write_header, index=False)


def load_all_interactions(include_new_interactions: bool = True) -> pd.DataFrame:
    """Original plus (optionally) API-recorded interactions, in the internal schema."""
    _, raw_interactions = load_data()
    if include_new_interactions:
        new_interactions = load_new_interactions()
        if not new_interactions.empty:
            raw_interactions = pd.concat([raw_interactions, new_interactions], ignore_index=True)
    return to_internal_schema(raw_interactions)
