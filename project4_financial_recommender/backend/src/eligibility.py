"""Hard eligibility rules, kept separate from the ranking model.

These are illustrative, educational rules, not financial advice, regulatory
requirements or real bank policy. They mirror the two constraints the
synthetic dataset was generated with: no logged interaction breaks either.

Missing critical data is never treated as "eligible": a rule that needs an
attribute the profile lacks fails, with an explicit reason.
"""
import math

import pandas as pd

from .config import RISK_ORDER

# Logged with every recommendation so an audit can tell which rule set was applied.
RULES_VERSION = "2026-09-30.1"

RULES = [
    {
        "rule_id": "min_income",
        "description": "Monthly income must be at least the product's minimum income.",
    },
    {
        "rule_id": "risk_suitability",
        "description": "High-risk products are not offered to users with a low risk profile.",
    },
]


def _is_missing(value) -> bool:
    """True for None or NaN (pandas turns missing CSV values into NaN)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def check_eligibility(user: dict, product: dict) -> list:
    """Return the rules this user/product pair fails (empty list = eligible)."""
    failures = []

    # Rule 1: income must reach the product's minimum. Only checked when there is a minimum,
    # so products without one stay eligible even when income is unknown.
    min_income = product.get("min_income") or 0
    income = user.get("monthly_income")
    if min_income > 0:
        if _is_missing(income):
            failures.append({
                "rule_id": "min_income",
                "detail": f"Requires monthly income ≥ {min_income:,.0f}; income is unknown.",
            })
        elif income < min_income:
            failures.append({
                "rule_id": "min_income",
                "detail": f"Requires monthly income ≥ {min_income:,.0f}; user has {income:,.0f}.",
            })

    # Rule 2: high-risk products need a known risk profile that is not 'low'.
    # Medium and high profiles pass; only a *caution* is shown later for medium (see explanations).
    if product.get("risk_level") == "high":
        risk_profile = user.get("risk_profile")
        if _is_missing(risk_profile) or risk_profile not in RISK_ORDER:
            failures.append({
                "rule_id": "risk_suitability",
                "detail": "High-risk product; the user's risk profile is unknown.",
            })
        elif risk_profile == "low":
            failures.append({
                "rule_id": "risk_suitability",
                "detail": "High-risk product; the user's risk profile is low.",
            })
    return failures


def is_eligible(user: dict, product: dict) -> bool:
    """True when the pair fails no rule."""
    return not check_eligibility(user, product)


def eligibility_report(user: dict, products: pd.DataFrame) -> pd.DataFrame:
    """One row per product with an `eligible` flag and the failed rules."""
    report = products.copy()
    report["failed_rules"] = [check_eligibility(user, product) for product in products.to_dict("records")]
    report["eligible"] = report["failed_rules"].map(len) == 0
    return report


def eligible_mask(pairs: pd.DataFrame) -> pd.Series:
    """Vectorised check for a frame of user/product pairs (used in training and evaluation).
    Must agree with check_eligibility; tests enforce that."""
    # Same two rules as check_eligibility, but over whole columns at once (fast for cross joins).
    # fillna(-1): unknown income can never meet a positive minimum. isin(...) is False for a
    # missing or unknown risk profile, so high-risk products fail exactly as in the dict version.
    income_ok = (pairs["min_income"] <= 0) | (pairs["monthly_income"].fillna(-1) >= pairs["min_income"])
    risk_ok = (pairs["risk_level"] != "high") | pairs["risk_profile"].isin(["medium", "high"])
    return income_ok & risk_ok
