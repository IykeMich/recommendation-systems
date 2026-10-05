"""The policy layer is tested on its own, with no model loaded (guide section 14)."""
import pandas as pd
import pytest

from src.candidates import generate_eligible_candidates
from src.data import load_prepared
from src.eligibility import check_eligibility, eligible_mask, is_eligible

HOME_LOAN = {"product_id": "FP016", "min_income": 500_000, "risk_level": "high"}
SAVINGS = {"product_id": "FP001", "min_income": 0, "risk_level": "low"}


@pytest.mark.parametrize("user, product, expected_rules", [
    ({"monthly_income": 600_000, "risk_profile": "high"}, HOME_LOAN, []),
    ({"monthly_income": 400_000, "risk_profile": "high"}, HOME_LOAN, ["min_income"]),
    ({"monthly_income": 600_000, "risk_profile": "low"}, HOME_LOAN, ["risk_suitability"]),
    ({"monthly_income": 100, "risk_profile": "low"}, HOME_LOAN, ["min_income", "risk_suitability"]),
    ({"monthly_income": 100, "risk_profile": "low"}, SAVINGS, []),
])
def test_rules(user, product, expected_rules):
    """Each rule fires independently, in a fixed order."""
    assert [failure["rule_id"] for failure in check_eligibility(user, product)] == expected_rules


def test_missing_critical_data_is_never_eligible():
    assert not is_eligible({}, HOME_LOAN)
    assert {failure["rule_id"] for failure in check_eligibility({}, HOME_LOAN)} == {"min_income", "risk_suitability"}
    assert is_eligible({}, SAVINGS)  # no requirement needs the missing data


def test_income_exactly_at_minimum_is_eligible():
    assert is_eligible({"monthly_income": 500_000, "risk_profile": "medium"}, HOME_LOAN)


def test_vectorised_mask_matches_rule_function_on_real_data():
    """eligible_mask (training/evaluation) and check_eligibility (serving) must never disagree."""
    users, products, _ = load_prepared(include_new_interactions=False)
    pairs = users.merge(products, how="cross")
    per_pair = [is_eligible(user, product) for user, product in zip(
        pairs[users.columns].to_dict("records"), pairs[products.columns].to_dict("records"))]
    assert eligible_mask(pairs).tolist() == per_pair


def test_logged_behaviour_never_breaks_the_rules():
    """The rules mirror the data: no logged interaction is for an ineligible product."""
    users, products, interactions = load_prepared(include_new_interactions=False)
    events = interactions.merge(users, on="user_id").merge(products, on="product_id")
    assert eligible_mask(events).all()


def test_candidates_are_exactly_the_eligible_products():
    products = pd.DataFrame([HOME_LOAN, SAVINGS]).assign(name=["Home Loan", "Savings"], category=["loan", "savings"])
    candidates, report = generate_eligible_candidates({"monthly_income": 1_000, "risk_profile": "low"}, products)
    assert candidates["product_id"].tolist() == ["FP001"]
    assert report["eligible"].tolist() == [False, True]
