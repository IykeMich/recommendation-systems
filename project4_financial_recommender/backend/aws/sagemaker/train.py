"""SageMaker training entry point (guide section 37).

Packages the model WITH its feature configuration and the eligibility rules'
version, so the endpoint never runs a model without its preprocessing
assumptions. Use backend/ as source_dir so src/ ships alongside.
Training channel: users.csv, financial_products.csv, interactions.csv.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import joblib

# Make backend/ importable so `src` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import NEGATIVES_PER_POSITIVE, RANDOM_SEED
from src.data import prepare_interactions, prepare_products, prepare_users
from src.eligibility import RULES, RULES_VERSION
from src.evaluate import evaluate
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame, build_training_examples
from src.model import build_ranker

DATA_FILES = ["users.csv", "financial_products.csv", "interactions.csv"]


def main():
    """Train on the SageMaker training channel and write model artifacts plus metrics."""
    # SageMaker sets these env vars; defaults are the standard container paths.
    input_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    users = prepare_users(pd.read_csv(os.path.join(input_dir, "users.csv")))
    products = prepare_products(pd.read_csv(os.path.join(input_dir, "financial_products.csv")))
    interactions = prepare_interactions(pd.read_csv(os.path.join(input_dir, "interactions.csv")))

    training_requests = build_training_examples(interactions, users, products, NEGATIVES_PER_POSITIVE, RANDOM_SEED)
    training_frame = build_feature_frame(training_requests, users, products, interactions)
    model = build_ranker(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model.fit(training_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], training_frame["label"])

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "ranker.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as feature_file:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, feature_file)
    with open(os.path.join(model_dir, "metadata.json"), "w") as metadata_file:
        json.dump({"eligibility_rules_version": RULES_VERSION, "eligibility_rules": RULES}, metadata_file, indent=2)
    # Profiles, catalog and history snapshot for request-time features and eligibility.
    for data_file in DATA_FILES:
        shutil.copy(os.path.join(input_dir, data_file), os.path.join(model_dir, data_file))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(evaluate(interactions, users, products), metrics_file, indent=2)


if __name__ == "__main__":
    main()
