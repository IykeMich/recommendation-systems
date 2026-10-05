"""SageMaker training entry point.

SageMaker copies the S3 training channel to SM_CHANNEL_TRAINING and uploads
whatever is written to SM_MODEL_DIR as model.tar.gz. Use backend/ as source_dir so
src/ ships with this file and the exact same model code is used.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow `python aws/sagemaker/train.py` from backend/ as well as SageMaker.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_TOP_K, EVENT_WEIGHTS
from src.evaluate import evaluate
from src.preprocess import build_user_item, prepare_events
from src.recommender import AdaptiveHybridRecommender, popular_items


def main():
    """Train on the training channel's interactions.csv + products.csv; write model, fallback list and metrics."""
    # SageMaker sets these env vars; the defaults are the standard container paths.
    input_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    # Same pipeline as src/train.py: event weights -> user-item matrix -> hybrid (content + item-item).
    products = pd.read_csv(os.path.join(input_dir, "products.csv"))
    events = pd.read_csv(os.path.join(input_dir, "interactions.csv"))
    events = prepare_events(events)
    matrix = build_user_item(events)
    model = AdaptiveHybridRecommender(products).fit(matrix)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    # The endpoint needs the cold-start fallback too, so ship it with the model.
    with open(os.path.join(model_dir, "popular_items.json"), "w") as popular_file:
        json.dump(popular_items(events, k=50), popular_file)
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as metadata_file:
        json.dump({
            "model_type": "adaptive_hybrid",
            "event_weights": EVENT_WEIGHTS,
            "num_users": int(matrix.shape[0]),
            "num_items": int(matrix.shape[1]),
        }, metadata_file, indent=2)

    # Offline metrics go to the output dir (uploaded as output.tar.gz), separate from the model artifact.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(evaluate(events, k=DEFAULT_TOP_K, products=products), metrics_file, indent=2)


if __name__ == "__main__":
    main()
