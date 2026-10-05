"""Preprocessing: turn raw events into weights and a user x item matrix (pipeline steps 2-3)."""
import pandas as pd

from .config import EVENT_WEIGHTS


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Derive the model weight from event_type (not the supplied event_value)."""
    df = events.copy()
    df["weight"] = df["event_type"].map(EVENT_WEIGHTS)
    # .map() yields NaN for event types missing from EVENT_WEIGHTS; fail loudly instead of training on NaN.
    if df["weight"].isna().any():
        unknown = df.loc[df["weight"].isna(), "event_type"].unique()
        raise ValueError(f"Unknown event types: {unknown}")
    return df


def build_user_item(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated user-item events into one user x item strength matrix."""
    # Step 1: sum all weights per (user, item), e.g. view (1) + purchase (6) -> 7.
    user_item = (
        df.groupby(["user_id", "item_id"], as_index=False)["weight"]
        .sum()
    )
    # Step 2: pivot to a dense matrix: one row per user, one column per item, 0 where no interaction.
    return user_item.pivot_table(
        index="user_id",
        columns="item_id",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )


def matrix_sparsity(matrix: pd.DataFrame) -> float:
    """Fraction of user-item cells that are empty (0); close to 1.0 is typical for retail data."""
    density = (matrix.to_numpy() != 0).sum() / matrix.size
    return float(1 - density)
