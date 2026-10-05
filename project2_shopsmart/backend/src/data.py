"""Data loading: the supplied CSVs plus any new interactions captured later (step 1 of the pipeline)."""
import pandas as pd

from .config import DATA_DIR, NEW_INTERACTIONS_PATH

# Column order shared by interactions.csv and new_interactions.csv, so the two can be concatenated.
INTERACTION_COLUMNS = [
    "user_id",
    "item_id",
    "event_type",
    "event_value",
    "timestamp",
    "device_type",
    "region",
]


def load_data():
    """Return the original (users, products, interactions) DataFrames from data/."""
    users = pd.read_csv(DATA_DIR / "users.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    events = pd.read_csv(DATA_DIR / "interactions.csv")
    return users, products, events


def load_new_interactions() -> pd.DataFrame:
    """Return interactions recorded after the original dataset (an empty frame if there are none yet)."""
    if not NEW_INTERACTIONS_PATH.exists():
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    return pd.read_csv(NEW_INTERACTIONS_PATH)


def append_new_interaction(interaction: dict) -> None:
    """Append one interaction row to new_interactions.csv, writing the header only when creating the file."""
    row = pd.DataFrame([interaction], columns=INTERACTION_COLUMNS)
    write_header = not NEW_INTERACTIONS_PATH.exists()
    row.to_csv(NEW_INTERACTIONS_PATH, mode="a", header=write_header, index=False)


def load_training_events(include_new_interactions: bool = True) -> pd.DataFrame:
    """Original interactions, optionally followed by the new ones (appended after, in file order)."""
    _, _, events = load_data()
    if include_new_interactions:
        new_events = load_new_interactions()
        if not new_events.empty:
            events = pd.concat([events, new_events], ignore_index=True)
    return events
