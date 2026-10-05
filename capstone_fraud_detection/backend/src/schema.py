"""Schema adapter and data-quality checks (guide sections 5 and 7).

The logical names follow the guide's contract. Each maps to the first matching
raw column; a required field that cannot be resolved fails loudly.
"""
import pandas as pd

# Logical field -> raw column names to try, in priority order (first match wins).
ALIASES = {
    "transaction_id": ["transaction_id", "txn_id", "id"],
    "customer_id": ["customer_id", "user_id", "account_id"],
    "amount": ["amount", "transaction_amount", "value"],
    "currency": ["currency"],
    "hour": ["hour", "transaction_hour"],
    "velocity_1h": ["velocity_1h", "txn_velocity_1h"],
    "country": ["country", "location", "city", "transaction_location"],
    "device": ["device", "device_type", "channel"],
    "merchant_category": ["merchant_category", "mcc_category"],
    "is_fraud": ["is_fraud", "fraud", "fraud_label", "label"],
}

# Fields in the guide's contract that this dataset does not have. Features that
# depend on them (temporal split, per-device/per-merchant novelty) are adapted.
UNAVAILABLE_IN_DATASET = ["transaction_timestamp", "merchant_id", "device_id"]


def resolve_column(df: pd.DataFrame, candidates: list, logical_name: str) -> str:
    """Return the first candidate column present in df; raise ValueError if none exists."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not resolve '{logical_name}'. Expected one of: {candidates}")


def resolve_columns(df: pd.DataFrame) -> dict:
    """Map every logical field in ALIASES to its raw column name (fails on the first missing one)."""
    return {logical: resolve_column(df, candidates, logical) for logical, candidates in ALIASES.items()}


def to_internal_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to logical names, keep only those, and coerce numeric fields.

    errors="coerce" turns unparseable values into NaN so the quality checks can count them.
    """
    resolved = resolve_columns(raw)
    # Invert {logical: raw} to {raw: logical} for rename, then select columns in ALIASES order.
    df = raw.rename(columns={source: logical for logical, source in resolved.items()})[list(ALIASES)]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["velocity_1h"] = pd.to_numeric(df["velocity_1h"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Checks that run before any feature engineering."""
    checks = {
        "rows": int(len(df)),
        "duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
        "missing_transaction_id": int(df["transaction_id"].isna().sum()),
        "missing_customer_id": int(df["customer_id"].isna().sum()),
        "missing_amount": int(df["amount"].isna().sum()),
        "non_positive_amount": int((df["amount"] <= 0).sum()),
        "hour_out_of_range": int((~df["hour"].between(0, 23)).sum()),
        "velocity_below_one": int((df["velocity_1h"] < 1).sum()),
        "invalid_label": int((~df["is_fraud"].isin([0, 1])).sum()),
        "currencies": sorted(df["currency"].dropna().unique().tolist()),
    }
    checks["rows_rejected"] = int(invalid_row_mask(df).sum())
    checks["fraud_rate"] = round(float(df["is_fraud"].mean()), 4)
    checks["fraud_count"] = int(df["is_fraud"].sum())
    checks["unavailable_contract_fields"] = UNAVAILABLE_IN_DATASET
    return checks


def invalid_row_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that fail any quality rule; these rows are dropped before training.

    NaN hours/labels also count as invalid because between()/isin() return False for NaN.
    """
    return (
        df["transaction_id"].isna()
        # keep="first": the first copy of a duplicated ID survives, later copies are rejected.
        | df["transaction_id"].duplicated(keep="first")
        | df["customer_id"].isna()
        | df["amount"].isna()
        | (df["amount"] <= 0)
        | ~df["hour"].between(0, 23)
        | (df["velocity_1h"] < 1)
        | ~df["is_fraud"].isin([0, 1])
    )


def load_clean_transactions(path) -> tuple:
    """Returns (clean rows, data-quality report)."""
    df = to_internal_schema(pd.read_csv(path))
    # The report is computed on ALL rows (before filtering) so it shows what was rejected.
    report = data_quality_report(df)
    return df[~invalid_row_mask(df)].reset_index(drop=True), report
