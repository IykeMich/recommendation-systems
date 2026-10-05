"""Unit tests for the label rule, point-in-time features, negative sampling and candidate rules,
using a tiny hand-made dataset."""
import pandas as pd
import pytest

from src.candidates import generate_candidates
from src.data import prepare_offers, to_internal_schema
from src.features import build_feature_frame, build_training_examples, daypart_for_hour
from src.recommender import select_top_k


@pytest.fixture
def offers():
    return prepare_offers(pd.DataFrame([
        ("O1", "Fashion A", "Fashion", 10, 0, "both", "all", True, "2026-12-31"),
        ("O2", "Audio B", "Audio", 20, 50, "web", "Lagos", True, "2026-12-31"),
        ("O3", "Home C", "Home", 30, 0, "mobile", "Abuja", True, "2026-01-01"),
    ], columns=["offer_id", "title", "category", "discount_pct", "min_spend", "channel", "region", "active", "expires_at"]))


@pytest.fixture
def users():
    return pd.DataFrame([("U1", "family", "Fashion")], columns=["user_id", "segment", "preferred_category"])


@pytest.fixture
def interactions():
    return to_internal_schema(pd.DataFrame([
        ("U1", "O1", "click", 1_700_000_000, "mobile", "Lagos"),
        ("U1", "O2", "impression", 1_700_100_000, "mobile", "Lagos"),
        ("U1", "O1", "redeem", 1_700_200_000, "mobile", "Lagos"),
    ], columns=["user_id", "offer_id", "event_type", "timestamp", "device_type", "region"]))


def test_label_rule_marks_click_and_redeem_positive(interactions):
    assert interactions["label"].tolist() == [1, 0, 1]


def test_daypart_boundaries():
    assert [daypart_for_hour(hour) for hour in (0, 6, 12, 18, 23)] == [
        "night", "morning", "afternoon", "evening", "evening"
    ]


def test_history_features_only_use_earlier_events(offers, users, interactions):
    request_time = interactions["timestamp"].iloc[1]  # between the click and the redeem
    requests = pd.DataFrame({
        "user_id": ["U1"], "offer_id": ["O1"], "request_time": [request_time],
        "device_type": ["mobile"], "region": ["Lagos"], "hour": [9], "day_of_week": [2],
    })
    row = build_feature_frame(requests, offers, users, interactions).iloc[0]
    assert row["interaction_count"] == 1  # only the click is before request_time
    assert row["positive_count"] == 1
    assert row["category_affinity"] == 1.0
    assert row["offer_popularity"] == 1
    assert row["preferred_category_match"] == 1
    assert row["region_match"] == 1 and row["channel_match"] == 1
    assert row["category_x_daypart"] == "Fashion|morning"


def test_negatives_never_use_an_engaged_offer(offers, interactions):
    examples = build_training_examples(interactions, offers, negatives_per_positive=4, random_seed=0)
    negatives = examples[examples["label"] == 0]
    assert len(negatives) == 8
    assert "O1" not in set(negatives["offer_id"])


def test_candidate_rules_and_funnel(offers):
    """O3 is expired by 2026-06-01; O1 and O2 then pass the Lagos region and web channel rules."""
    candidates, funnel = generate_candidates(
        offers, region="Lagos", device_type="web", today=pd.Timestamp("2026-06-01").date(),
    )
    assert candidates["offer_id"].tolist() == ["O1", "O2"]
    assert [step["remaining"] for step in funnel] == [3, 2, 2, 2]


def test_candidate_rules_can_be_relaxed(offers):
    candidates, _ = generate_candidates(offers, region="Enugu", device_type="web", enforce_region=False, enforce_channel=False)
    assert len(candidates) == 3


def test_select_top_k_caps_categories():
    ranked = pd.DataFrame({"offer_id": list("ABCDE"), "category": ["x", "x", "x", "y", "y"]})
    assert select_top_k(ranked, 4, max_per_category=2)["offer_id"].tolist() == ["A", "B", "D", "E"]
