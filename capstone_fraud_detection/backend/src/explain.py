"""Evidence-backed reasons, kept separate from the model.

A reason is only shown when the transaction falls in a bucket whose fraud rate
in the TRAINING data is clearly above the overall rate, and the reason carries
that evidence. Novelty observations (new device/country, unusual amount) are
reported as context: they are true facts about the transaction, but this
dataset gives them no measurable fraud signal, so they are not called reasons.
"""
import numpy as np
import pandas as pd

# A bucket needs at least this many training rows, and a fraud rate at least
# LIFT_FOR_REASON times the overall rate, before it is shown as a reason.
MIN_BUCKET_SIZE = 50
LIFT_FOR_REASON = 1.5

AMOUNT_EDGES = [0, 500, 1000, 2000, 5000, np.inf]

# Each function maps a frame to a bucket label per row (as strings, so they work as JSON keys).
BUCKETERS = {
    "hour": lambda frame: frame["hour"].astype(int).astype(str),
    "velocity_1h": lambda frame: frame["velocity_1h"].astype(int).astype(str),
    "device": lambda frame: frame["device"].astype(str),
    "country": lambda frame: frame["country"].astype(str),
    "merchant_category": lambda frame: frame["merchant_category"].astype(str),
    "amount": lambda frame: pd.cut(frame["amount"], AMOUNT_EDGES, right=False).astype(str),
}

REASON_TEMPLATES = {
    "hour": "Transaction at {value}:00, an hour with a high fraud rate",
    "velocity_1h": "High velocity: {value} transactions in the last hour",
    "device": "Device type '{value}' has a high fraud rate",
    "country": "Country {value} has a high fraud rate",
    "merchant_category": "Merchant category '{value}' has a high fraud rate",
    "amount": "Amount in range {value}, which has a high fraud rate",
}


def build_reference(training_rows: pd.DataFrame) -> dict:
    """Per-feature, per-bucket fraud rate and row count from the training rows (stored in the artifact)."""
    overall_rate = float(training_rows["is_fraud"].mean())
    buckets = {}
    for feature, bucketer in BUCKETERS.items():
        stats = training_rows.groupby(bucketer(training_rows))["is_fraud"].agg(["mean", "size"])
        buckets[feature] = {
            str(bucket): {"fraud_rate": round(float(row["mean"]), 4), "transactions": int(row["size"])}
            for bucket, row in stats.iterrows()
        }
    return {"overall_fraud_rate": round(overall_rate, 4), "buckets": buckets}


def explain_transaction(row: dict, reference: dict, max_reasons: int = 4) -> dict:
    """Return up to `max_reasons` evidence-backed reasons (highest lift first) plus context notes.

    `row` is one transaction merged with its feature snapshot; `reference` comes from build_reference.
    """
    overall_rate = reference["overall_fraud_rate"]
    frame = pd.DataFrame([row])
    reasons = []
    for feature, bucketer in BUCKETERS.items():
        bucket = str(bucketer(frame).iloc[0])
        stats = reference["buckets"][feature].get(bucket)
        if not stats or stats["transactions"] < MIN_BUCKET_SIZE:
            continue
        # Lift = how many times more often this bucket was fraud than the average transaction.
        lift = stats["fraud_rate"] / overall_rate if overall_rate else 0
        if lift >= LIFT_FOR_REASON:
            reasons.append({
                "text": REASON_TEMPLATES[feature].format(value=bucket),
                "evidence": {"feature": feature, "bucket": bucket, "training_fraud_rate": stats["fraud_rate"],
                             "overall_fraud_rate": overall_rate, "training_transactions": stats["transactions"],
                             "lift": round(lift, 1)},
            })
    reasons.sort(key=lambda reason: reason["evidence"]["lift"], reverse=True)
    if not reasons:
        reasons.append({"text": "No single dominant risk signal", "evidence": {"feature": None}})

    # Context: factual novelty notes about the customer's history, not presented as reasons.
    context = []
    if row.get("customer_txn_count", 0) == 0:
        context.append("Customer has no reference history")
    else:
        if row.get("customer_device_share", 0) == 0:
            context.append(f"Device type '{row['device']}' is new for this customer")
        if row.get("customer_country_share", 0) == 0:
            context.append(f"Country {row['country']} is new for this customer")
        ratio = row.get("amount_vs_customer_mean")
        if ratio is not None and not pd.isna(ratio) and ratio >= 3:
            context.append(f"Amount is {ratio:.1f}× the customer's average")
    return {"reasons": reasons[:max_reasons], "context": context}
