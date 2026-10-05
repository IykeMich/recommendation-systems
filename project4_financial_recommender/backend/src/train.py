"""Train the ranker on all data, optionally evaluate it, and write artifacts/.

Run with: python -m src.train. Writes ranker.joblib, feature_columns.json, metadata.json
(model and rules versions, label rule, top coefficients) and metrics.json.
"""
import json
from datetime import datetime, timezone

import joblib

from .config import ARTIFACT_DIR, NEGATIVES_PER_POSITIVE, POSITIVE_EVENT_TYPES, RANDOM_SEED
from .data import load_prepared
from .eligibility import RULES, RULES_VERSION
from .evaluate import evaluate
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame, build_training_examples
from .model import build_ranker, coefficient_table


def train(include_new_interactions: bool = True, run_evaluation: bool = True):
    """Fit on every interaction (plus new feedback if requested) and save the artifacts."""
    users, products, interactions = load_prepared(include_new_interactions)

    training_requests = build_training_examples(interactions, users, products, NEGATIVES_PER_POSITIVE, RANDOM_SEED)
    training_frame = build_feature_frame(training_requests, users, products, interactions)
    model = build_ranker(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model.fit(training_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], training_frame["label"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "ranker.joblib")
    with open(ARTIFACT_DIR / "feature_columns.json", "w") as feature_file:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, feature_file, indent=2)

    # Metadata ties the model to its version, the rule version and the label rule (used in audit logs).
    metadata = {
        "model_type": "logistic_regression_ranker",
        "model_version": datetime.now(timezone.utc).strftime("ranker-%Y%m%dT%H%M%SZ"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_rule": f"positive = {sorted(POSITIVE_EVENT_TYPES)}; "
                      f"{NEGATIVES_PER_POSITIVE} sampled eligible products per positive",
        "eligibility_rules_version": RULES_VERSION,
        "eligibility_rules": RULES,
        "num_interactions": int(len(interactions)),
        "num_training_rows": int(len(training_frame)),
        "num_users": int(len(users)),
        "num_products": int(len(products)),
        "top_coefficients": coefficient_table(model).head(12).round(4).to_dict("records"),
    }
    # The temporal evaluation trains its own separate ranker on the earliest 80% of events.
    if run_evaluation:
        metadata["evaluation"] = evaluate(interactions, users, products)
        with open(ARTIFACT_DIR / "metrics.json", "w") as metrics_file:
            json.dump(metadata["evaluation"], metrics_file, indent=2)

    with open(ARTIFACT_DIR / "metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return model, metadata


if __name__ == "__main__":
    _, trained_metadata = train()
    print(json.dumps(trained_metadata, indent=2))
