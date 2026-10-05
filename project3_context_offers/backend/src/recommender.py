"""Serving logic: ContextualRecommender runs candidate rules, scores candidates with the ranker,
takes the Top-K (optionally capped per category) and attaches reasons. Unknown users get popularity."""
from typing import Optional

import pandas as pd

from .candidates import generate_candidates
from .features import build_feature_frame
from .model import feature_contributions, top_reasons

def select_top_k(ranked: pd.DataFrame, k: int, max_per_category: Optional[int] = None) -> pd.DataFrame:
    """Take the best k rows, optionally capping how many come from one category."""
    if not max_per_category:
        return ranked.head(k)
    per_category_rank = ranked.groupby("category").cumcount()
    return ranked[per_category_rank < max_per_category].head(k)


class ContextualRecommender:
    """Candidate generation + contextual ranking + Top-K for one request."""

    def __init__(self, model, numeric_features, categorical_features, offers, users, interactions):
        self.model = model
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.offers = offers
        self.users = users
        self.interactions = interactions

    def knows_user(self, user_id: str) -> bool:
        """Known = has a profile or any interaction; otherwise recommend() uses the cold-start path."""
        return bool(
            self.users["user_id"].eq(user_id).any()
            or self.interactions["user_id"].eq(user_id).any()
        )

    def add_interaction(self, interaction: dict) -> None:
        """New feedback updates history features immediately, without retraining."""
        self.interactions = pd.concat(
            [self.interactions, pd.DataFrame([interaction])], ignore_index=True
        )

    def offer_popularity(self, before: pd.Timestamp) -> pd.Series:
        """Click/redeem counts per offer before `before`; trims the candidate pool and drives cold start."""
        engaged = self.interactions[
            (self.interactions["label"] == 1) & (self.interactions["timestamp"] < before)
        ]
        return engaged["offer_id"].value_counts()

    def recommend(
        self,
        user_id: str,
        context: dict,
        k: int = 10,
        now: Optional[pd.Timestamp] = None,
        enforce_region: bool = True,
        enforce_channel: bool = True,
        max_per_category: Optional[int] = None,
    ) -> dict:
        """Run one request: rules -> (cold start | ranker) -> Top-K with reasons.

        `context` holds device_type, region, hour and day_of_week. Returns strategy, funnel and offers.
        """
        now = now or pd.Timestamp.now(tz="UTC")
        popularity = self.offer_popularity(before=now)
        # Offers this user has redeemed are excluded by the candidate rules.
        redeemed_offer_ids = self.interactions.loc[
            self.interactions["user_id"].eq(user_id) & self.interactions["event_type"].eq("redeem"),
            "offer_id",
        ]
        candidates, funnel = generate_candidates(
            self.offers,
            region=context["region"],
            device_type=context["device_type"],
            today=now.date(),
            exclude_offer_ids=redeemed_offer_ids,
            enforce_region=enforce_region,
            enforce_channel=enforce_channel,
            offer_popularity=popularity,
        )

        if candidates.empty:
            return {"strategy": "no_eligible_offers", "funnel": funnel, "recommendations": []}

        if not self.knows_user(user_id):
            return {
                "strategy": "cold_start_popular",
                "funnel": funnel,
                "recommendations": self._popular(candidates, popularity, k),
            }

        # One request row per candidate, all sharing this user, time and context.
        requests = pd.DataFrame({
            "user_id": user_id,
            "offer_id": candidates["offer_id"].to_numpy(),
            "request_time": now,
            **context,
        })
        features = build_feature_frame(requests, self.offers, self.users, self.interactions)
        model_input = features[self.numeric_features + self.categorical_features]
        features["score"] = self.model.predict_proba(model_input)[:, 1]
        # Highest score first, then Top-K (with the optional per-category cap). Reasons are computed
        # only for the offers actually shown.
        ranked = select_top_k(features.sort_values("score", ascending=False), k, max_per_category)
        contributions, transformed_values = feature_contributions(self.model, model_input.loc[ranked.index])

        offer_details = candidates.set_index("offer_id")
        return {
            "strategy": "contextual_ranker",
            "funnel": funnel,
            "recommendations": [
                {
                    **self._offer_fields(offer_details.loc[row.offer_id], row.offer_id),
                    "score": round(float(row.score), 4),
                    "reasons": top_reasons(contributions.loc[row_index], transformed_values.loc[row_index]),
                }
                for row_index, row in ranked.iterrows()
            ],
        }

    def _popular(self, candidates: pd.DataFrame, popularity: pd.Series, k: int) -> list:
        """Cold-start fallback: rank candidates by engagement count; score is scaled to 0-1 vs the top one."""
        candidate_popularity = candidates["offer_id"].map(popularity).fillna(0)
        ranked = candidates.assign(popularity=candidate_popularity).sort_values(
            "popularity", ascending=False
        ).head(k)
        top_popularity = max(float(ranked["popularity"].max()), 1.0)
        return [
            {
                **self._offer_fields(row, row.offer_id),
                "score": round(float(row.popularity) / top_popularity, 4),
                "reasons": [{"label": f"{int(row.popularity)} engagements from other shoppers", "weight": 0}],
            }
            for row in ranked.itertuples(index=False)
        ]

    @staticmethod
    def _offer_fields(offer, offer_id: str) -> dict:
        """Offer fields returned to clients; numpy numbers become plain ints/strings for JSON."""
        return {
            "offer_id": offer_id,
            "title": offer.title,
            "category": offer.category,
            "discount_pct": int(offer.discount_pct),
            "min_spend": int(offer.min_spend),
            "channel": offer.channel,
            "region": offer.region,
            "expires_at": str(offer.expires_at),
        }


def load_recommender(include_new_interactions: bool = True) -> ContextualRecommender:
    """Build a recommender from the saved artifact and the current data."""
    import json

    import joblib

    from .config import ARTIFACT_DIR
    from .data import load_all_interactions, load_data, load_users, prepare_offers

    feature_columns = json.loads((ARTIFACT_DIR / "feature_columns.json").read_text())
    raw_offers, _ = load_data()
    return ContextualRecommender(
        model=joblib.load(ARTIFACT_DIR / "ranker.joblib"),
        numeric_features=feature_columns["numeric"],
        categorical_features=feature_columns["categorical"],
        offers=prepare_offers(raw_offers),
        users=load_users(),
        interactions=load_all_interactions(include_new_interactions),
    )
