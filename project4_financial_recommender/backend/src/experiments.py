"""The Project 4 experiments (guide sections 33 and 44).

Run with: python -m src.experiments   (after python -m src.train)
Nothing is written to disk.
"""
import json

import pandas as pd

from .config import ARTIFACT_DIR
from .eligibility import check_eligibility, eligibility_report
from .explanations import verify_reason
from .recommender import load_recommender

# Fixed request time so the experiments are reproducible.
NOW = pd.Timestamp("2026-09-30T12:00:00Z")


def product_ids(result):
    """Product IDs of a recommend() result, in ranked order."""
    return [product["product_id"] for product in result["recommendations"]]


def main():
    """Run each experiment and print its outcome."""
    recommender = load_recommender()
    users, products = recommender.users, recommender.products

    print("=== 1 / A. Hard-ineligible products can never appear, even with a boosted score ===")
    low_risk_user = users[users["risk_profile"] == "low"].iloc[0].to_dict()
    blocked_product = products[products["risk_level"] == "high"].iloc[0]
    print(f"  {low_risk_user['user_id']} (risk profile low) vs {blocked_product['product_id']} {blocked_product['name']} (high risk):",
          check_eligibility(low_risk_user, blocked_product.to_dict()))
    # Monkeypatch the scorer to force the blocked product to the top score, then restore it.
    original_score = recommender.score

    def boosted_score(profile, catalog, now):
        """Original scores, except the blocked product gets 1.0."""
        scores = original_score(profile, catalog, now)
        scores.loc[scores["product_id"] == blocked_product["product_id"], "score"] = 1.0
        return scores

    recommender.score = boosted_score
    boosted = recommender.recommend(low_risk_user["user_id"], k=5, now=NOW, include_unfiltered=True)
    recommender.score = original_score
    print("  with the product's score forced to 1.0:")
    print("    unfiltered ranker top-5:", [(row["product_id"], row["eligible"]) for row in boosted["unfiltered_ranking"]])
    print("    served recommendations: ", product_ids(boosted), "| blocked product present:",
          blocked_product["product_id"] in product_ids(boosted))

    print("\n=== 2 / B. Change the user profile, observe ranking and eligibility ===")
    user_id = "F00002"
    base = recommender.recommend(user_id, k=5, now=NOW)
    changed = recommender.recommend(user_id, k=5, now=NOW, profile_overrides={"employment": "business_owner", "monthly_income": 2_000_000})
    print(f"  {user_id} as-is:          eligible {base['eligibility']['eligible_products']}/20, top-5 {product_ids(base)}")
    print(f"  business owner, 2M income: eligible {changed['eligibility']['eligible_products']}/20, top-5 {product_ids(changed)}")

    print("\n=== 3. Change a product attribute, observe ranking ===")
    changed_product_id = base["recommendations"][-1]["product_id"]
    product_row = products["product_id"] == changed_product_id

    def position(result):
        """1-based rank and score of the changed product in a result."""
        ranked = product_ids(result)
        index = ranked.index(changed_product_id)
        return index + 1, result["recommendations"][index]["score"]

    before_rank, before_score = position(recommender.recommend(user_id, k=20, now=NOW))
    # Temporarily edit the catalog in place, re-rank, then restore the original value.
    original_category = products.loc[product_row, "category"].iloc[0]
    recommender.products.loc[product_row, "category"] = "savings"
    after_rank, after_score = position(recommender.recommend(user_id, k=20, now=NOW))
    recommender.products.loc[product_row, "category"] = original_category
    print(f"  {changed_product_id} category {original_category} -> savings: rank {before_rank} -> {after_rank}, "
          f"score {before_score:.3f} -> {after_score:.3f}")

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    evaluation = metadata.get("evaluation")
    if evaluation:
        k = evaluation["k"]
        print(f"\n=== 4, 8, 9, 10. Temporal evaluation ({evaluation['test_requests']} test applications) ===")
        print(f"  {'strategy':<28} recall@{k}  ndcg@{k}  coverage  ineligible slots  empty")
        for name, metrics in evaluation["strategies"].items():
            print(f"  {name:<28} {metrics[f'recall@{k}']:.3f}     {metrics[f'ndcg@{k}']:.3f}    "
                  f"{metrics['coverage']:.2f}      {metrics['eligibility_violation_rate']:.3f}             "
                  f"{metrics['empty_recommendation_rate']:.3f}")

    print("\n=== 5 / C. Unknown user ===")
    unknown = recommender.recommend("NOBODY", k=5, now=NOW)
    print(f"  strategy {unknown['strategy']}, eligible {unknown['eligibility']['eligible_products']}/20 "
          "(income/risk unknown, so only no-minimum, non-high-risk products)")

    print("\n=== 6 / D. Zero eligible candidates ===")
    # Swap in a catalog nobody qualifies for to show the explicit empty outcome.
    restrictive_catalog = products.assign(min_income=5_000_000)
    recommender.products = restrictive_catalog
    empty = recommender.recommend("F00001", k=5, now=NOW)
    recommender.products = products
    print(f"  every product requires 5M income: strategy {empty['strategy']}, recommendations {empty['recommendations']}")

    print("\n=== 7 / E. Verify every explanation against source data ===")
    # Re-check every reason and caution against the dataset profile, catalog and history.
    checked = failed = 0
    popularity_rank = recommender.popularity_ranks(NOW)
    for sample_user_id in users["user_id"].head(200):
        result = recommender.recommend(sample_user_id, k=5, now=NOW)
        profile = users[users["user_id"] == sample_user_id].iloc[0].to_dict()
        history = recommender.interactions[recommender.interactions["user_id"] == sample_user_id].merge(
            products[["product_id", "category"]], on="product_id")
        for product in result["recommendations"]:
            catalog_row = products[products["product_id"] == product["product_id"]].iloc[0].to_dict()
            for reason in product["reasons"] + product["cautions"]:
                checked += 1
                failed += not verify_reason(reason, profile, catalog_row, history, int(popularity_rank[product["product_id"]]))
    print(f"  {checked} reasons checked for 200 users, {failed} unsupported")

    print("\n=== Policy: how restrictive is the rule layer? ===")
    rates = [eligibility_report(user, products)["eligible"].mean() for user in users.to_dict("records")]
    print(f"  mean eligibility rate {sum(rates) / len(rates):.3f}; users with every product eligible: "
          f"{sum(rate == 1 for rate in rates)}/{len(rates)}")


if __name__ == "__main__":
    main()
