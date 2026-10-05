"""Training entry point: build the hybrid model from all events, evaluate it, and write artifacts/.

Run with: python -m src.train (the API also calls train() on start-up and on retrain).
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy import sparse

from .config import (
    ARTIFACT_DIR,
    CONTENT_SHARE_HISTORY_SIZE,
    CONTENT_SHARE_MAX,
    CONTENT_SHARE_MIN,
    DEFAULT_TOP_K,
    EVENT_WEIGHTS,
)
from .data import load_data, load_training_events
from .evaluate import evaluate
from .preprocess import build_user_item, matrix_sparsity, prepare_events
from .recommender import AdaptiveHybridRecommender


def train(include_new_interactions: bool = True, run_evaluation: bool = True):
    """Fit the hybrid model, save it plus metadata/metrics to ARTIFACT_DIR; return (model, metadata)."""
    _, products, _ = load_data()
    events = prepare_events(load_training_events(include_new_interactions))
    matrix = build_user_item(events)
    model = AdaptiveHybridRecommender(products).fit(matrix)

    # Persist the whole model (for the API) and the similarity matrix + item order separately
    # (sparse .npz, since most item pairs share no users) for inspection or other consumers.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    sparse.save_npz(ARTIFACT_DIR / "item_similarity.npz", sparse.csr_matrix(model.collaborative.item_similarity))
    np.save(ARTIFACT_DIR / "item_ids.npy", model.collaborative.item_ids)

    # num_events is also used by the API to work out how many new interactions this model already includes.
    metadata = {
        "model_type": "adaptive_hybrid",
        "content_share": {
            "max": CONTENT_SHARE_MAX,
            "min": CONTENT_SHARE_MIN,
            "history_size": CONTENT_SHARE_HISTORY_SIZE,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_weights": EVENT_WEIGHTS,
        "num_events": int(len(events)),
        "num_users": int(matrix.shape[0]),
        "num_items": int(matrix.shape[1]),
        "sparsity": round(matrix_sparsity(matrix), 4),
    }
    # Evaluation trains a separate model on a holdout split; the saved model above uses all events.
    if run_evaluation:
        metadata["evaluation"] = evaluate(events, k=DEFAULT_TOP_K, products=products)
        with open(ARTIFACT_DIR / "metrics.json", "w") as metrics_file:
            json.dump(metadata["evaluation"], metrics_file, indent=2)

    with open(ARTIFACT_DIR / "model_metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return model, metadata


if __name__ == "__main__":
    _, trained_metadata = train()
    print(json.dumps(trained_metadata, indent=2))
