"""The eight Project 3 experiments (guide section 35).

Run with: python -m src.experiments   (after python -m src.train)
Nothing is written to disk.
"""
import json
from datetime import date

import pandas as pd

from .config import ARTIFACT_DIR
from .model import coefficient_table
from .recommender import load_recommender

# A fixed 'now' and base context keep the experiment output reproducible between runs.
NOW = pd.Timestamp("2026-09-30T12:00:00Z")
BASE_CONTEXT = {"device_type": "mobile", "region": "Lagos", "hour": 9, "day_of_week": 1}


def top_ids(recommender, user_id, context, k=10, **options):
    """Recommend at the fixed NOW and return (list of offer IDs, full result dict)."""
    result = recommender.recommend(user_id, context, k=k, now=NOW, **options)
    return [offer["offer_id"] for offer in result["recommendations"]], result


def compare(title, first_ids, second_ids):
    """Print how many offers two Top-K lists share and whether their order is identical."""
    overlap = len(set(first_ids) & set(second_ids))
    same_order = first_ids == second_ids
    print(f"{title}: {overlap}/{len(first_ids)} offers shared, identical order: {same_order}")


def main():
    """Run the experiments in order and print the results."""
    recommender = load_recommender()
    user_id = "R00001"

    print("=== 1. Same user, different hour (9:00 vs 21:00) ===")
    morning, _ = top_ids(recommender, user_id, BASE_CONTEXT)
    evening, _ = top_ids(recommender, user_id, {**BASE_CONTEXT, "hour": 21})
    compare("  top-10", morning, evening)
    morning_diverse, _ = top_ids(recommender, user_id, BASE_CONTEXT, max_per_category=3)
    evening_diverse, _ = top_ids(recommender, user_id, {**BASE_CONTEXT, "hour": 21}, max_per_category=3)
    compare("  top-10 with max 3 per category", morning_diverse, evening_diverse)

    print("\n=== 2. Same user, different region (Lagos vs Abuja) ===")
    abuja, abuja_result = top_ids(recommender, user_id, {**BASE_CONTEXT, "region": "Abuja"})
    compare("  top-10", morning, abuja)
    print("  candidate funnel in Abuja:", [step["remaining"] for step in abuja_result["funnel"]])

    print("\n=== 3. Same context, different user ===")
    for other_user_id in ["R00002", "R00003"]:
        other, _ = top_ids(recommender, other_user_id, BASE_CONTEXT)
        compare(f"  {user_id} vs {other_user_id}", morning, other)

    print("\n=== 4. Unknown user ===")
    _, unknown_result = top_ids(recommender, "BRAND-NEW-USER", BASE_CONTEXT)
    print("  strategy:", unknown_result["strategy"], "| first offer:", unknown_result["recommendations"][0]["title"])

    print("\n=== 5. New offer with attributes but no interactions ===")
    preferred_category = recommender.users.set_index("user_id").loc[user_id, "preferred_category"]
    new_offer = {
        "offer_id": "O_NEW", "title": f"Launch {preferred_category} Offer", "category": preferred_category,
        "discount_pct": 30, "min_spend": 0, "channel": "both", "region": "all",
        "active": True, "expires_at": date(2026, 12, 31),
    }
    # Added to the in-memory catalog only: the ranker scores it from attributes alone.
    recommender.offers = pd.concat([recommender.offers, pd.DataFrame([new_offer])], ignore_index=True)
    with_new, _ = top_ids(recommender, user_id, BASE_CONTEXT, k=20)
    rank = with_new.index("O_NEW") + 1 if "O_NEW" in with_new else None
    print(f"  O_NEW ({preferred_category}, 30% off, zero history) rank for {user_id}: {rank}")
    print("  A pure collaborative model (Project 2) could not rank it at all.")

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    evaluation = metadata.get("evaluation")
    if evaluation:
        k = evaluation["k"]
        print(f"\n=== 6 & 7. Offline comparison (recall@{k} / ndcg@{k} / diversity) ===")
        for name, metrics in evaluation["models"].items():
            print(f"  {name:<32} {metrics[f'recall@{k}']:.3f} / {metrics[f'ndcg@{k}']:.3f} / {metrics['diversity']:.2f}")

    print("\n=== 8. Model coefficients (largest effects) ===")
    for row in coefficient_table(recommender.model).head(10).itertuples():
        print(f"  {row.coefficient:+.3f}  {row.feature}")


if __name__ == "__main__":
    main()
