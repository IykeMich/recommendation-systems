"""Serving-time orchestration: eligibility -> candidates -> ranker -> explanations.

Also handles cold start (unknown users, ranked by popularity but still filtered by the rules),
what-if profile overrides, and an optional "no eligibility layer" ranking for teaching.
"""
from typing import Optional

import pandas as pd

from .candidates import generate_eligible_candidates
from .eligibility import RULES_VERSION
from .explanations import explain
from .features import build_feature_frame

# Profile fields returned to the client and fed to the feature builder.
PROFILE_FIELDS = ["age_band", "monthly_income", "employment", "risk_profile", "region", "existing_products"]
PRODUCT_FIELDS = ["product_id", "name", "category", "risk_level", "min_income", "description"]


class FinancialRecommender:
    """Eligibility first, then ranking, then explanation."""

    def __init__(self, model, numeric_features, categorical_features, users, products, interactions):
        """Hold the fitted model, its feature lists and the data used for features and eligibility."""
        self.model = model
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.users = users
        self.products = products
        self.interactions = interactions

    def resolve_profile(self, user_id: str, profile_overrides: Optional[dict] = None):
        """The profile used for this request and where it came from."""
        # Precedence: dataset profile first, what-if overrides on top. None values mean "no override".
        # Unknown user + overrides -> a pure what-if profile; unknown user, no overrides -> (None, "none").
        profile_overrides = {key: value for key, value in (profile_overrides or {}).items() if value is not None}
        matches = self.users[self.users["user_id"] == user_id]
        if not matches.empty:
            profile = matches.iloc[0].to_dict()
            source = "dataset"
        elif profile_overrides:
            profile = {"user_id": user_id}
            source = "what_if"
        else:
            return None, "none"
        if profile_overrides:
            profile.update(profile_overrides)
            source = "dataset+what_if" if source == "dataset" else source
        return profile, source

    def add_interaction(self, interaction: dict) -> None:
        """Append a new (internal-schema) interaction so history features and reasons use it immediately."""
        self.interactions = pd.concat([self.interactions, pd.DataFrame([interaction])], ignore_index=True)

    def score(self, profile: dict, products: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        """Ranker probability for each product in `products`, given the (possibly what-if) profile.

        Features come from a one-row profile frame and from history strictly before `now`.
        """
        requests = pd.DataFrame({"user_id": profile["user_id"], "product_id": products["product_id"].to_numpy(), "request_time": now})
        profile_frame = pd.DataFrame([{field: profile.get(field) for field in ["user_id", *PROFILE_FIELDS]}])
        history = self.interactions[self.interactions["timestamp"] < now]
        features = build_feature_frame(requests, profile_frame, self.products, history)
        features["score"] = self.model.predict_proba(features[self.numeric_features + self.categorical_features])[:, 1]
        return features[["product_id", "score"]]

    def popularity_ranks(self, now: pd.Timestamp) -> pd.Series:
        """Rank of every catalog product by event count before `now`.

        1 = most popular; ties are broken by catalog order.
        """
        counts = self.interactions[self.interactions["timestamp"] < now]["product_id"].value_counts()
        counts = counts.reindex(self.products["product_id"], fill_value=0)
        return counts.rank(method="first", ascending=False).astype(int)

    def recommend(
        self,
        user_id: str,
        k: int = 5,
        profile_overrides: Optional[dict] = None,
        include_unfiltered: bool = False,
        now: Optional[pd.Timestamp] = None,
    ) -> dict:
        """Top-k eligible products with scores and evidence-backed reasons, plus the eligibility report.

        `include_unfiltered` adds the ranker's top-k over ALL products (eligible or not) for comparison.
        """
        now = now or pd.Timestamp.now(tz="UTC")
        profile, profile_source = self.resolve_profile(user_id, profile_overrides)
        # Unknown user: no profile, so rules needing income or risk data fail
        # (missing data is never "eligible"). Cold start still respects eligibility.
        eligibility_input = profile or {"user_id": user_id}
        candidates, report = generate_eligible_candidates(eligibility_input, self.products)
        popularity_rank = self.popularity_ranks(now)

        result = {
            "user_id": user_id,
            "profile": None if profile is None else {field: profile.get(field) for field in PROFILE_FIELDS},
            "profile_source": profile_source,
            "eligibility": {
                "rules_version": RULES_VERSION,
                "total_products": int(len(report)),
                "eligible_products": int(report["eligible"].sum()),
                "ineligible": [
                    {**{field: row[field] for field in PRODUCT_FIELDS}, "failed_rules": row["failed_rules"]}
                    for row in report[~report["eligible"]].to_dict("records")
                ],
            },
        }

        # Never fall back to ineligible products: an empty list is an explicit, honest outcome.
        if candidates.empty:
            return {**result, "strategy": "no_eligible_products", "recommendations": []}

        # This user's past events (with product category) are the evidence for history-based reasons.
        history = self.interactions[(self.interactions["user_id"] == user_id) & (self.interactions["timestamp"] < now)]
        history = history.merge(self.products[["product_id", "category"]], on="product_id", how="left")

        if profile is None:
            # Cold start: score = (catalog_size + 1 - popularity_rank) / catalog_size, so rank 1 -> 1.0.
            ranked = candidates.assign(score=candidates["product_id"].map(popularity_rank).rsub(len(self.products) + 1) / len(self.products))
            strategy = "cold_start_popular"
        else:
            # Score the whole catalog once, then keep only eligible candidates for the served list.
            scores = self.score(profile, self.products, now)
            ranked = candidates.merge(scores, on="product_id")
            strategy = "eligibility_then_ranking"
            if include_unfiltered:
                # Teaching comparison: the same scores with NO eligibility filter, flagged per product.
                unfiltered = report.merge(scores, on="product_id").sort_values("score", ascending=False).head(k)
                result["unfiltered_ranking"] = [
                    {"product_id": row["product_id"], "name": row["name"], "score": round(float(row["score"]), 4),
                     "eligible": bool(row["eligible"]), "failed_rules": row["failed_rules"]}
                    for row in unfiltered.to_dict("records")
                ]

        # Highest score first; product_id breaks ties so results are deterministic.
        top_products = ranked.sort_values(["score", "product_id"], ascending=[False, True]).head(k)
        explained_user = profile or {}
        return {
            **result,
            "strategy": strategy,
            "recommendations": [
                {
                    **{field: row[field] for field in PRODUCT_FIELDS},
                    "min_income": float(row["min_income"]),
                    "score": round(float(row["score"]), 4),
                    **explain(explained_user, row, history, int(popularity_rank[row["product_id"]])),
                }
                for row in top_products.to_dict("records")
            ],
        }


def load_recommender(include_new_interactions: bool = True) -> FinancialRecommender:
    """Build a FinancialRecommender from the trained artifacts and the prepared data."""
    import json

    import joblib

    from .config import ARTIFACT_DIR
    from .data import load_prepared

    feature_columns = json.loads((ARTIFACT_DIR / "feature_columns.json").read_text())
    users, products, interactions = load_prepared(include_new_interactions)
    return FinancialRecommender(
        model=joblib.load(ARTIFACT_DIR / "ranker.joblib"),
        numeric_features=feature_columns["numeric"],
        categorical_features=feature_columns["categorical"],
        users=users,
        products=products,
        interactions=interactions,
    )
