"""Feature leakage rules, schema adapter, data-quality mask and evidence-backed reasons."""
import pandas as pd
import pytest

from src.explain import LIFT_FOR_REASON, build_reference, explain_transaction
from src.features import CustomerProfiles, build_features
from src.schema import invalid_row_mask, resolve_column, to_internal_schema


def transactions(rows):
    """Build an internal-schema DataFrame from tuples."""
    return pd.DataFrame(rows, columns=["transaction_id", "customer_id", "amount", "currency", "hour", "velocity_1h",
                                       "country", "device", "merchant_category", "is_fraud"])


@pytest.fixture
def history():
    """Tiny training history: C1 has two android/NG rows, C2 has one web/GB row."""
    return transactions([
        ("T1", "C1", 100.0, "NGN", 10, 1, "NG", "android", "food", 0),
        ("T2", "C1", 300.0, "NGN", 11, 2, "NG", "android", "retail", 0),
        ("T3", "C2", 50.0, "NGN", 12, 1, "GB", "web", "fuel", 0),
    ])


def test_new_device_is_zero_share_for_first_observation(history):
    profiles = CustomerProfiles().fit(history)
    new = build_features(transactions([("T9", "C1", 200.0, "NGN", 3, 7, "NG", "ios", "food", 0)]), profiles).iloc[0]
    assert new["customer_device_share"] == 0          # ios never seen for C1
    assert new["customer_country_share"] == 1.0
    assert new["customer_txn_count"] == 2
    assert new["amount_vs_customer_mean"] == pytest.approx(1.0)  # mean of 100 and 300


def test_training_rows_never_see_themselves(history):
    profiles = CustomerProfiles().fit(history)
    own = build_features(history, profiles, leave_one_out=True)
    first = own[own["transaction_id"] == "T1"].iloc[0]
    assert first["customer_txn_count"] == 1
    assert first["customer_mean_amount"] == 300.0      # only T2 remains
    solo = own[own["transaction_id"] == "T3"].iloc[0]
    assert solo["customer_txn_count"] == 0 and pd.isna(solo["customer_mean_amount"])


def test_unknown_customer_has_empty_history(history):
    profiles = CustomerProfiles().fit(history)
    unknown = build_features(transactions([("T9", "NEW", 10.0, "NGN", 1, 1, "US", "web", "food", 0)]), profiles).iloc[0]
    assert unknown["customer_txn_count"] == 0
    assert unknown["customer_device_share"] == 0 and pd.isna(unknown["amount_vs_customer_mean"])


def test_cyclic_hour_puts_23_next_to_0(history):
    profiles = CustomerProfiles().fit(history)
    frame = build_features(transactions([
        ("A", "C1", 1.0, "NGN", 23, 1, "NG", "web", "food", 0),
        ("B", "C1", 1.0, "NGN", 0, 1, "NG", "web", "food", 0),
        ("C", "C1", 1.0, "NGN", 12, 1, "NG", "web", "food", 0),
    ]), profiles)
    distance = lambda first, second: ((frame.loc[first, ["hour_sin", "hour_cos"]] - frame.loc[second, ["hour_sin", "hour_cos"]]) ** 2).sum()
    assert distance(0, 1) < distance(0, 2)


def test_schema_adapter_resolves_aliases_and_fails_loudly():
    raw = pd.DataFrame({"txn_id": ["T1"], "user_id": ["C1"], "value": [5.0], "currency": ["NGN"], "hour": [1],
                        "velocity_1h": [2], "city": ["NG"], "device": ["web"], "merchant_category": ["food"], "label": [0]})
    assert to_internal_schema(raw).columns.tolist()[:3] == ["transaction_id", "customer_id", "amount"]
    with pytest.raises(ValueError, match="Could not resolve 'amount'"):
        resolve_column(raw.drop(columns="value"), ["amount", "value"], "amount")


def test_data_quality_rejects_bad_rows(history):
    bad = pd.concat([history, transactions([
        ("T1", "C1", 10.0, "NGN", 1, 1, "NG", "web", "food", 0),      # duplicate id
        ("T5", "C1", -5.0, "NGN", 1, 1, "NG", "web", "food", 0),      # negative amount
        ("T6", "C1", 5.0, "NGN", 25, 1, "NG", "web", "food", 0),      # bad hour
        ("T7", "C1", 5.0, "NGN", 1, 1, "NG", "web", "food", 2),       # bad label
    ])], ignore_index=True)
    assert invalid_row_mask(bad).tolist() == [False, False, False, True, True, True, True]


def test_reasons_carry_training_evidence():
    training = transactions(
        [(f"R{i}", "C", 10.0, "NGN", 3, 9, "NG", "web", "food", 1 if i < 40 else 0) for i in range(60)]
        + [(f"S{i}", "C", 10.0, "NGN", 12, 1, "NG", "web", "food", 0) for i in range(540)]
    )
    reference = build_reference(training)
    explanation = explain_transaction({**training.iloc[0].to_dict(), "customer_txn_count": 5,
                                       "customer_device_share": 1, "customer_country_share": 1}, reference)
    features = {reason["evidence"]["feature"] for reason in explanation["reasons"]}
    assert features == {"hour", "velocity_1h"}
    assert all(reason["evidence"]["lift"] >= LIFT_FOR_REASON for reason in explanation["reasons"])
