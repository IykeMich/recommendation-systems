"""The recommender models: item-item collaborative filtering, the content + collaborative hybrid,
and the popularity fallback (steps 4-7).

user-item matrix -> item-item cosine similarity -> candidate scores -> remove seen items -> Top-K.
The hybrid blends those scores with product-to-product content similarity.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CONTENT_SHARE_HISTORY_SIZE,
    CONTENT_SHARE_MAX,
    CONTENT_SHARE_MIN,
    SAME_BRAND_BONUS,
    SAME_CATEGORY_BONUS,
    SAME_SUBCATEGORY_BONUS,
)


class ItemItemRecommender:
    """Transparent item-item collaborative filtering over weighted implicit feedback."""

    def __init__(self):
        """Create an untrained model; call fit() before recommend()."""
        self.matrix = None
        self.item_ids = None
        self.item_similarity = None

    def fit(self, user_item_matrix: pd.DataFrame):
        """Compute the item x item cosine similarity matrix and index users; returns self for chaining."""
        self.matrix = user_item_matrix
        self.item_ids = user_item_matrix.columns.to_numpy()
        # Transpose so each row is an item described by its weights across all users;
        # two items are similar when the same users engaged with both.
        self.item_similarity = cosine_similarity(user_item_matrix.T)
        # A product is not a recommendation "because of" itself.
        np.fill_diagonal(self.item_similarity, 0.0)
        # Lookup from user_id to its row number in the matrix, plus the raw rows as a float array.
        self._user_to_row = {
            user_id: row for row, user_id in enumerate(user_item_matrix.index)
        }
        self._user_vectors = user_item_matrix.to_numpy(dtype=float)
        return self

    def knows_user(self, user_id: str) -> bool:
        """True if the user had at least one interaction in the training matrix."""
        return user_id in self._user_to_row

    def recommend(self, user_id: str, k: int = 10, explain: bool = False):
        """Top-k unseen items as [{"item_id", "score"}]; [] for unknown users (the caller falls back).

        With explain=True each result also lists the seen items that contributed most to its score.
        """
        if not self.knows_user(user_id):
            return []

        user_vector = self._user_vectors[self._user_to_row[user_id]]
        # Score every item using the user's known interaction strengths.
        # (score[j] = sum over items i of weight[i] * similarity[i, j])
        scores = user_vector @ self.item_similarity
        # Do not recommend items the user already interacted with.
        seen = user_vector > 0
        scores[seen] = -np.inf

        # Walk items from highest to lowest score, skipping seen (-inf) and items with no evidence (<= 0).
        order = np.argsort(-scores)
        results = []
        for idx in order:
            if not np.isfinite(scores[idx]) or scores[idx] <= 0:
                continue
            recommendation = {
                "item_id": str(self.item_ids[idx]),
                "score": float(scores[idx]),
            }
            if explain:
                recommendation["because_item_ids"] = self._top_contributors(
                    user_vector, idx
                )
            results.append(recommendation)
            if len(results) == k:
                break
        return results

    def _top_contributors(self, user_vector: np.ndarray, candidate_idx: int, top_n: int = 2):
        """Seen items that contributed most to a candidate's score."""
        # Each term of the candidate's score: the user's weight on item i times similarity(i, candidate).
        contributions = user_vector * self.item_similarity[:, candidate_idx]
        top_indices = np.argsort(-contributions)[:top_n]
        return [
            str(self.item_ids[idx]) for idx in top_indices if contributions[idx] > 0
        ]


def build_content_similarity(products: pd.DataFrame) -> np.ndarray:
    """Product x product similarity from catalog data, in products' row order.

    TF-IDF over category, subcategory, brand and description, plus bonuses for sharing a
    subcategory, category or brand. Needs no shopper behaviour, so it works from the first click.
    """
    # The "Variant: small pack." suffix is shared by unrelated products, so leave it out.
    descriptions = products["description"].fillna("").str.replace(r"\s*Variant:.*$", "", regex=True)
    text = (
        products["category"].fillna("") + " " + products["subcategory"].fillna("") + " "
        + products["brand"].fillna("") + " " + descriptions
    )
    similarity = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(text))

    def same(column):
        values = products[column].to_numpy()
        return (values[:, None] == values[None, :]).astype(float)

    similarity += (
        SAME_SUBCATEGORY_BONUS * same("subcategory")
        + SAME_CATEGORY_BONUS * same("category")
        + SAME_BRAND_BONUS * same("brand")
    )
    np.fill_diagonal(similarity, 0.0)
    return similarity


def content_share(distinct_items_seen: int) -> float:
    """How much of the blended score comes from content similarity, given the shopper's history size."""
    share = 1.0 - distinct_items_seen / CONTENT_SHARE_HISTORY_SIZE
    return float(np.clip(share, CONTENT_SHARE_MIN, CONTENT_SHARE_MAX))


class AdaptiveHybridRecommender:
    """Blend of "similar products" (content) and "shoppers like you" (item-item collaborative).

    New shoppers with a short history lean on content similarity; as their history grows the
    blend shifts towards collaborative filtering (see content_share). Every result can explain
    itself: which of the shopper's products it is similar to, and how many shoppers who
    interacted with one of the shopper's products also interacted with it.
    """

    def __init__(self, products: pd.DataFrame):
        """Create an untrained model over the catalog; call fit() before recommend()."""
        self.item_ids = products["item_id"].to_numpy()
        self._item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self.subcategories = products["subcategory"].to_numpy()
        self.categories = products["category"].to_numpy()
        self.brands = products["brand"].to_numpy()
        self.content_similarity = build_content_similarity(products)
        self.collaborative = None

    def fit(self, user_item_matrix: pd.DataFrame):
        """Fit the collaborative part and index who interacted with what; returns self."""
        self.collaborative = ItemItemRecommender().fit(user_item_matrix)
        # Position of each collaborative matrix column in the catalog order (-1 if not in the catalog).
        self._catalog_position = np.array(
            [self._item_index.get(item_id, -1) for item_id in self.collaborative.item_ids]
        )
        self._column_of = {item_id: column for column, item_id in enumerate(self.collaborative.item_ids)}
        # Binary "did this user interact with this item" matrix; column dot products count shoppers.
        self._interacted = sparse.csc_matrix(self.collaborative._user_vectors > 0, dtype=np.int32)
        return self

    def knows_user(self, user_id: str) -> bool:
        """True if the user had at least one interaction in the training matrix."""
        return self.collaborative.knows_user(user_id)

    def _to_catalog(self, values: np.ndarray) -> np.ndarray:
        """Re-index a vector over collaborative columns into catalog order (missing items get 0)."""
        result = np.zeros(len(self.item_ids))
        in_catalog = self._catalog_position >= 0
        result[self._catalog_position[in_catalog]] = values[in_catalog]
        return result

    def shoppers_in_common(self, item_a: str, item_b: str) -> int:
        """Number of training shoppers who interacted with both items."""
        if item_a not in self._column_of or item_b not in self._column_of:
            return 0
        both = self._interacted[:, self._column_of[item_a]].multiply(self._interacted[:, self._column_of[item_b]])
        return int(both.sum())

    def recommend(self, user_id: str, k: int = 10, explain: bool = False):
        """Top-k unseen products as [{"item_id", "score", "source"}]; [] for unknown users.

        With explain=True each result also carries "reasons" (see _reasons).
        """
        if not self.knows_user(user_id):
            return []

        collaborative_vector = self.collaborative._user_vectors[self.collaborative._user_to_row[user_id]]
        user_weights = self._to_catalog(collaborative_vector)
        seen = user_weights > 0

        content_scores = user_weights @ self.content_similarity
        collaborative_scores = self._to_catalog(collaborative_vector @ self.collaborative.item_similarity)

        # Never recommend what the shopper already has. Other sizes of it stay eligible: re-buying a
        # different pack size is common (46% of held-out purchases in the synthetic data).
        excluded = seen

        def normalised(scores):
            """Scale to [0, 1] over the candidates so the two parts are comparable."""
            top = scores[~excluded].max(initial=0.0)
            return scores / top if top > 0 else np.zeros_like(scores)

        share = content_share(int(seen.sum()))
        content_part = share * normalised(content_scores)
        collaborative_part = (1 - share) * normalised(collaborative_scores)
        blended = content_part + collaborative_part
        blended[excluded] = -np.inf

        results = []
        for index in np.argsort(-blended):
            if not np.isfinite(blended[index]) or blended[index] <= 0 or len(results) == k:
                break
            recommendation = {
                "item_id": str(self.item_ids[index]),
                "score": float(blended[index]),
                "source": "similar_products" if content_part[index] >= collaborative_part[index]
                else "shoppers_like_you",
            }
            if explain:
                recommendation["reasons"] = self._reasons(
                    user_weights, collaborative_vector, index, recommendation["source"]
                )
            results.append(recommendation)
        return results

    def _reasons(self, user_weights, collaborative_vector, candidate: int, source: str, top_n: int = 2):
        """Why a candidate was recommended, strongest source first.

        similar_product: the shopper's product it is most similar to, and what they share.
        co_interaction: the shopper's products that drove the collaborative score, with how many
        training shoppers interacted with both.
        """
        reasons = []
        candidate_id = str(self.item_ids[candidate])

        content_contributions = user_weights * self.content_similarity[:, candidate]
        anchor = int(np.argmax(content_contributions))
        if content_contributions[anchor] > 0:
            for attribute, values in [("subcategory", self.subcategories), ("brand", self.brands),
                                      ("category", self.categories)]:
                if values[anchor] == values[candidate]:
                    shared, value = attribute, str(values[candidate])
                    break
            else:
                shared, value = "description", None
            reasons.append({"type": "similar_product", "item_id": str(self.item_ids[anchor]),
                            "shared": shared, "value": value})

        co_interaction_reasons = []
        if candidate_id in self._column_of:
            # Each term of the collaborative score: the shopper's weight on item i x similarity(i, candidate).
            contributions = collaborative_vector * self.collaborative.item_similarity[:, self._column_of[candidate_id]]
            for index in np.argsort(-contributions)[:top_n]:
                if contributions[index] <= 0:
                    break
                anchor_id = str(self.collaborative.item_ids[index])
                count = self.shoppers_in_common(anchor_id, candidate_id)
                if count > 0:
                    co_interaction_reasons.append({"type": "co_interaction", "item_id": anchor_id,
                                                   "shopper_count": count})

        # Lead with the reason that matches the dominant source.
        return co_interaction_reasons + reasons if source == "shoppers_like_you" else reasons + co_interaction_reasons


def popular_items(weighted_events: pd.DataFrame, k: int = 10, exclude=()):
    """Popularity fallback for users the collaborative model has never seen."""
    # Popularity = total event weight an item received across all users, highest first.
    popularity = (
        weighted_events.groupby("item_id")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    # Optionally drop items (e.g. ones the user already saw) before taking the top k.
    popularity = popularity[~popularity.index.isin(list(exclude))]
    return [
        {"item_id": str(item_id), "score": float(score)}
        for item_id, score in popularity.head(k).items()
    ]
