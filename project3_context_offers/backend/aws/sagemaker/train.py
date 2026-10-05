"""SageMaker training entry point (guide section 29).

Package with the project's src/ directory as source_dir so SageMaker runs the
same feature and model code that was tested locally.

Training channel must contain offers.csv, offer_interactions.csv and users.csv.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

# Make `src` importable both from the repo layout and from SageMaker's source_dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import NEGATIVES_PER_POSITIVE, RANDOM_SEED
from src.data import prepare_offers, to_internal_schema
from src.evaluate import evaluate
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame, build_training_examples
from src.model import build_model

DATA_FILES = ["offers.csv", "offer_interactions.csv", "users.csv"]


def main():
    """Train on the SageMaker training channel; write the model, feature list and data snapshot."""
    input_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    offers = prepare_offers(pd.read_csv(os.path.join(input_dir, "offers.csv")))
    interactions = to_internal_schema(pd.read_csv(os.path.join(input_dir, "offer_interactions.csv")))
    users = pd.read_csv(os.path.join(input_dir, "users.csv"))

    training_requests = build_training_examples(interactions, offers, NEGATIVES_PER_POSITIVE, RANDOM_SEED)
    training_frame = build_feature_frame(training_requests, offers, users, interactions)
    model = build_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model.fit(training_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], training_frame["label"])

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "ranker.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as feature_file:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, feature_file)
    # The endpoint computes history features at request time, so it ships with
    # a snapshot of the data. Production would read these from a feature store.
    for data_file in DATA_FILES:
        shutil.copy(os.path.join(input_dir, data_file), os.path.join(model_dir, data_file))

    os.makedirs(output_dir, exist_ok=True)
    # Offline metrics go to the output data dir, not into the deployed model artifact.
    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(evaluate(interactions, offers, users), metrics_file, indent=2)


if __name__ == "__main__":
    main()
