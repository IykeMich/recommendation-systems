"""Central paths and constants shared by training, scoring, the API and tests."""
from pathlib import Path

# Project root (the folder that contains src/); every other path hangs off it.
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
ARTIFACT_DIR = ROOT / "artifacts"

PIPELINE_PATH = ARTIFACT_DIR / "fraud_pipeline.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "feature_manifest.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
THRESHOLD_TABLE_PATH = ARTIFACT_DIR / "threshold_table.csv"
POLICY_PATH = ARTIFACT_DIR / "policy.json"
DATA_QUALITY_PATH = ARTIFACT_DIR / "data_quality.json"
DECISION_DB_PATH = ROOT / "data" / "decisions.sqlite3"

# Written into the saved artifact and feature_manifest.json so a model can be traced to its feature schema.
MODEL_NAME = "fraud-risk"
FEATURE_SCHEMA_VERSION = "features-v1"
RANDOM_STATE = 42

# No timestamps exist in this dataset, so the split is stratified-random (see README).
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Teaching defaults from the guide, not recommendations for a real payment system.
DEFAULT_REVIEW_THRESHOLD = 0.55
DEFAULT_BLOCK_THRESHOLD = 0.85
