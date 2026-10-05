"""Training entry point (python -m src.train): build examples and features, fit the ranker,
and write ranker.joblib, feature_columns.json, metadata.json and metrics.json to artifacts/."""
import json
from datetime import datetime, timezone

import joblib

from .config import ARTIFACT_DIR, NEGATIVES_PER_POSITIVE, POSITIVE_EVENT_TYPES, RANDOM_SEED, TOP_K
from .data import load_all_interactions, load_data, load_users, prepare_offers
from .evaluate import evaluate
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame, build_training_examples
from .model import build_model, coefficient_table


def train(include_new_interactions: bool = True, run_evaluation: bool = True):
    """Fit the ranker on all interactions and save it with its metadata; optionally run evaluate().

    The saved model uses all data; evaluate() fits its own models on the earliest 80% only.
    """
    raw_offers, _ = load_data()
    offers = prepare_offers(raw_offers)
    users = load_users()
    interactions = load_all_interactions(include_new_interactions)

    training_requests = build_training_examples(interactions, offers, NEGATIVES_PER_POSITIVE, RANDOM_SEED)
    training_frame = build_feature_frame(training_requests, offers, users, interactions)

    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model.fit(training_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], training_frame["label"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "ranker.joblib")
    with open(ARTIFACT_DIR / "feature_columns.json", "w") as feature_file:
        # Serving reads this list so it feeds the model the same columns in the same order.
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, feature_file, indent=2)

    coefficients = coefficient_table(model)
    metadata = {
        "model_type": "logistic_regression_contextual_ranker",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_rule": f"positive = {sorted(POSITIVE_EVENT_TYPES)}; "
                      f"{NEGATIVES_PER_POSITIVE} sampled non-engaged offers per positive, same context",
        "num_interactions": int(len(interactions)),
        "num_training_rows": int(len(training_frame)),
        "num_offers": int(len(offers)),
        "num_users": int(interactions["user_id"].nunique()),
        "top_coefficients": coefficients.head(12).round(4).to_dict("records"),
    }
    if run_evaluation:
        metadata["evaluation"] = evaluate(interactions, offers, users)
        with open(ARTIFACT_DIR / "metrics.json", "w") as metrics_file:
            json.dump(metadata["evaluation"], metrics_file, indent=2)

    with open(ARTIFACT_DIR / "metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return model, metadata


if __name__ == "__main__":
    _, trained_metadata = train()
    print(json.dumps(trained_metadata, indent=2))
