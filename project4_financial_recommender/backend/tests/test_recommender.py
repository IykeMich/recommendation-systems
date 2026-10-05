"""End-to-end recommender tests against the trained artifacts (no API)."""
import pandas as pd
import pytest

from src.eligibility import is_eligible
from src.explanations import verify_reason
from src.recommender import load_recommender

NOW = pd.Timestamp("2026-09-30T12:00:00Z")


@pytest.fixture(scope="module")
def recommender():
    return load_recommender(include_new_interactions=False)


def test_recommendations_are_always_eligible(recommender):
    for user in recommender.users.head(150).to_dict("records"):
        result = recommender.recommend(user["user_id"], k=5, now=NOW)
        for product in result["recommendations"]:
            catalog_row = recommender.products.set_index("product_id").loc[product["product_id"]].to_dict()
            assert is_eligible(user, catalog_row)


def test_every_reason_is_supported_by_evidence(recommender):
    """verify_reason must confirm every reason and caution shown to 50 users."""
    popularity_rank = recommender.popularity_ranks(NOW)
    for user in recommender.users.head(50).to_dict("records"):
        history = recommender.interactions[recommender.interactions["user_id"] == user["user_id"]].merge(
            recommender.products[["product_id", "category"]], on="product_id")
        for product in recommender.recommend(user["user_id"], k=5, now=NOW)["recommendations"]:
            catalog_row = recommender.products[recommender.products["product_id"] == product["product_id"]].iloc[0].to_dict()
            for reason in product["reasons"] + product["cautions"]:
                assert verify_reason(reason, user, catalog_row, history, int(popularity_rank[product["product_id"]])), reason


def test_unknown_user_cold_start_respects_eligibility(recommender):
    result = recommender.recommend("NOBODY", k=20, now=NOW)
    assert result["strategy"] == "cold_start_popular"
    assert result["profile"] is None
    assert all(product["min_income"] == 0 and product["risk_level"] != "high" for product in result["recommendations"])


def test_what_if_profile_changes_eligibility(recommender):
    low = recommender.recommend("F00002", k=20, now=NOW, profile_overrides={"monthly_income": 50_000, "risk_profile": "low"})
    high = recommender.recommend("F00002", k=20, now=NOW, profile_overrides={"monthly_income": 3_000_000, "risk_profile": "high"})
    assert low["profile_source"] == "dataset+what_if"
    assert low["eligibility"]["eligible_products"] < high["eligibility"]["eligible_products"] == 20


def test_no_eligible_products_is_explicit(recommender):
    original_products = recommender.products
    recommender.products = original_products.assign(min_income=10_000_000)
    try:
        result = recommender.recommend("F00001", k=5, now=NOW)
    finally:
        recommender.products = original_products
    assert result["strategy"] == "no_eligible_products"
    assert result["recommendations"] == []


def test_unfiltered_comparison_flags_ineligible_products(recommender):
    """The no-eligibility ranking contains flagged ineligible products that are never served."""
    low_risk_user_id = recommender.users[recommender.users["risk_profile"] == "low"]["user_id"].iloc[0]
    result = recommender.recommend(low_risk_user_id, k=20, now=NOW, include_unfiltered=True)
    flagged = [row for row in result["unfiltered_ranking"] if not row["eligible"]]
    assert flagged and all(row["failed_rules"] for row in flagged)
    served = {product["product_id"] for product in result["recommendations"]}
    assert not served & {row["product_id"] for row in flagged}
