"""New-interaction and new-item experiments (sections 14 and 15 of the guide).

Run with: python -m src.experiments
Everything happens in memory; no files are changed.
"""
import pandas as pd

from .data import load_data
from .preprocess import build_user_item, prepare_events
from .recommender import ItemItemRecommender


def fit(events: pd.DataFrame) -> ItemItemRecommender:
    """Run the full pipeline (weights -> matrix -> similarity) on raw events, entirely in memory."""
    return ItemItemRecommender().fit(build_user_item(prepare_events(events)))


def make_event(user_id: str, item_id: str, event_type: str) -> dict:
    """Build one synthetic interaction row in the same shape as interactions.csv, stamped 'now'."""
    return {
        "user_id": user_id,
        "item_id": item_id,
        "event_type": event_type,
        "event_value": {"view": 1, "cart": 3, "purchase": 6}[event_type],
        "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
        "device_type": "web",
        "region": "Lagos",
    }


def print_rank_changes(before: list, after: list, products: pd.DataFrame) -> None:
    """Print the 'after' list with each item's movement vs 'before' (+n = moved up, 'new' = wasn't there)."""
    names = products.set_index("item_id")["name"]
    before_ranks = {recommendation["item_id"]: rank for rank, recommendation in enumerate(before, 1)}
    for rank, recommendation in enumerate(after, 1):
        item_id = recommendation["item_id"]
        previous_rank = before_ranks.get(item_id)
        movement = "new" if previous_rank is None else f"{previous_rank - rank:+d}"
        print(f"  {rank:>2}. {item_id} {names.get(item_id, '?'):<28} score={recommendation['score']:7.2f}  ({movement})")


def new_interaction_experiment(events: pd.DataFrame, products: pd.DataFrame, user_id: str = "R00001"):
    """Show how one new purchase reshapes a user's top-10 after retraining (guide section 14)."""
    print(f"\n=== New interaction experiment for {user_id} ===")
    model = fit(events)
    before = model.recommend(user_id, k=10)

    # Buy a product from a category this user has not engaged with much.
    # (Concretely: exclude the user's single most-engaged category, take the largest remaining
    # catalog category, and pick the first product in it the user hasn't touched.)
    seen_items = set(events.loc[events.user_id == user_id, "item_id"])
    user_categories = products[products.item_id.isin(seen_items)]["category"].value_counts()
    least_seen_category = (
        products.loc[~products.category.isin(user_categories.index[:1]), "category"]
        .value_counts()
        .index[0]
    )
    purchased_item_id = products[
        (products.category == least_seen_category) & ~products.item_id.isin(seen_items)
    ].item_id.iloc[0]
    print(f"Adding a purchase of {purchased_item_id} ({least_seen_category})")

    events_after = pd.concat([events, pd.DataFrame([make_event(user_id, purchased_item_id, "purchase")])], ignore_index=True)
    after = fit(events_after).recommend(user_id, k=10)

    print("Top-10 after the new purchase (movement vs before):")
    print_rank_changes(before, after, products)
    print(f"{purchased_item_id} itself is now 'seen', so it is filtered out:",
          purchased_item_id not in {recommendation["item_id"] for recommendation in after})


def new_item_experiment(events: pd.DataFrame):
    """Show the item cold-start problem: a new product needs co-interactions before CF can recommend it."""
    print("\n=== New item experiment ===")
    new_item_id = "P_NEW"

    # Case 1: zero interactions -> the collaborative model has no column for it at all.
    model = fit(events)
    print(f"Zero interactions: {new_item_id} known to the model?", new_item_id in set(model.item_ids))

    # Case 2: most buyers of a popular anchor product also buy the new product.
    # The first buyer (alphabetically) is held back as the test user; the rest create the signal.
    purchases = events[events.event_type == "purchase"]
    anchor_item_id = purchases.item_id.value_counts().index[0]
    anchor_buyers = sorted(purchases.loc[purchases.item_id == anchor_item_id, "user_id"].unique())
    test_user_id, signal_buyers = anchor_buyers[0], anchor_buyers[1:]

    new_events = pd.DataFrame([make_event(buyer, new_item_id, "purchase") for buyer in signal_buyers])
    model_with_signal = fit(pd.concat([events, new_events], ignore_index=True))

    # A held-out buyer of the anchor has never touched the new product.
    recommendations = model_with_signal.recommend(test_user_id, k=10)
    new_item_rank = next(
        (rank for rank, recommendation in enumerate(recommendations, 1) if recommendation["item_id"] == new_item_id),
        None,
    )
    print(f"After {len(signal_buyers)} {anchor_item_id} buyers purchase {new_item_id}: "
          f"its rank in {test_user_id}'s top-10 = {new_item_rank}")


if __name__ == "__main__":
    _, catalog, interaction_events = load_data()
    new_interaction_experiment(interaction_events, catalog)
    new_item_experiment(interaction_events)
