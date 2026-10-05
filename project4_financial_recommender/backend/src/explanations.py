"""Deterministic, evidence-backed explanations.

Each reason records the exact fields it relies on (`evidence`), so it can be
re-checked against the source data (experiment 7). Nothing here is persuasive
copy: if the evidence isn't there, the reason isn't shown.
"""
from .config import RISK_ORDER

# Past-tense wording for each event type in the "previously engaged" reason.
EVENT_VERBS = {"view": "viewed", "learn": "read about", "apply": "applied for"}


def explain(user: dict, product: dict, history_for_user, popularity_rank: int) -> dict:
    """Reasons and cautions for one eligible product.

    history_for_user: the user's past interactions (internal schema) with a `category` column.
    popularity_rank: 1 = the most-engaged-with product in the catalog.
    """
    reasons, cautions = [], []

    # Reason: the user interacted with this exact product before (cites the latest event).
    product_history = history_for_user[history_for_user["product_id"] == product["product_id"]]
    if not product_history.empty:
        latest = product_history.sort_values("timestamp").iloc[-1]
        reasons.append({
            "text": f"They {EVENT_VERBS.get(latest['event_type'], latest['event_type'])} this product before",
            "evidence": {"source": "history", "product_id": product["product_id"],
                         "event_type": latest["event_type"], "timestamp": latest["timestamp"].isoformat()},
        })

    # Reason: the user engaged with other products in the same category.
    other_in_category = history_for_user[
        (history_for_user["category"] == product["category"])
        & (history_for_user["product_id"] != product["product_id"])
    ]
    if not other_in_category.empty:
        reasons.append({
            "text": f"They engaged with {other_in_category['product_id'].nunique()} other {product['category']} product(s)",
            "evidence": {"source": "history", "category": product["category"],
                         "product_ids": sorted(other_in_category["product_id"].unique().tolist())},
        })

    # Reason: the product is among the 5 most engaged-with in the catalog.
    if popularity_rank <= 5:
        reasons.append({
            "text": f"One of the 5 most engaged-with products (#{popularity_rank})",
            "evidence": {"source": "popularity", "rank": popularity_rank},
        })

    # Reason: why the income rule passed (it must have, since only eligible products get here).
    min_income = product["min_income"]
    if min_income > 0:
        reasons.append({
            "text": f"Meets the minimum monthly income of {min_income:,.0f}",
            "evidence": {"source": "rule", "rule_id": "min_income",
                         "user_monthly_income": user.get("monthly_income"), "product_min_income": min_income},
        })
    else:
        reasons.append({
            "text": "No minimum income requirement",
            "evidence": {"source": "catalog", "product_min_income": 0},
        })

    # Risk fit: a reason when product risk <= profile risk, otherwise a caution (e.g. high-risk
    # product for a medium profile, which the rules allow). Skipped when the profile is unknown.
    risk_profile = user.get("risk_profile")
    if risk_profile in RISK_ORDER:
        if RISK_ORDER[product["risk_level"]] <= RISK_ORDER[risk_profile]:
            reasons.append({
                "text": f"{product['risk_level'].capitalize()} risk, within their {risk_profile} risk profile",
                "evidence": {"source": "profile", "risk_profile": risk_profile, "product_risk_level": product["risk_level"]},
            })
        else:
            cautions.append({
                "text": f"{product['risk_level'].capitalize()} risk is above their {risk_profile} risk profile",
                "evidence": {"source": "profile", "risk_profile": risk_profile, "product_risk_level": product["risk_level"]},
            })

    return {"reasons": reasons, "cautions": cautions}


def verify_reason(reason: dict, user: dict, product: dict, history_for_user, popularity_rank: int) -> bool:
    """Re-check a reason's evidence against the source data (experiment 7)."""
    # Dispatch on the evidence source; each branch re-derives the claim from the inputs.
    evidence = reason["evidence"]
    source = evidence["source"]
    # "Engaged with this product before": a matching event must exist in the history.
    if source == "history" and "event_type" in evidence:
        matches = history_for_user[
            (history_for_user["product_id"] == evidence["product_id"])
            & (history_for_user["event_type"] == evidence["event_type"])
        ]
        return evidence["product_id"] == product["product_id"] and not matches.empty
    # Category reason: every cited product must be in the user's history for that category.
    if source == "history":
        engaged = set(history_for_user.loc[history_for_user["category"] == evidence["category"], "product_id"])
        return evidence["category"] == product["category"] and set(evidence["product_ids"]) <= engaged
    if source == "popularity":
        return evidence["rank"] == popularity_rank
    if source == "rule":
        income = user.get("monthly_income") or 0
        return evidence["product_min_income"] == product["min_income"] and income >= product["min_income"]
    if source == "catalog":
        return product["min_income"] == 0
    if source == "profile":
        return evidence["risk_profile"] == user.get("risk_profile") and evidence["product_risk_level"] == product["risk_level"]
    return False
