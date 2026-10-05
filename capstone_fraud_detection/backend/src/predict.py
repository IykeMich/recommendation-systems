"""Load the artifact once and score transactions with the exact training-time
feature logic. Used by the API, the SageMaker handler and batch scoring."""
import joblib
import numpy as np
import pandas as pd

from .config import PIPELINE_PATH
from .features import build_features
from .train import add_model_inputs

# Raw and derived values stored with each decision so analysts see what the model saw.
SNAPSHOT_FIELDS = [
    "amount", "hour", "velocity_1h", "country", "device", "merchant_category",
    "customer_txn_count", "customer_mean_amount", "amount_vs_customer_mean",
    "customer_device_share", "customer_country_share",
]


class FraudScorer:
    """Wraps the saved training artifact: calibrated pipeline, feature list and customer profiles."""
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.pipeline = artifact["pipeline"]
        self.features = artifact["features"]
        self.profiles = artifact["profiles"]
        self.model_version = artifact["model_version"]

    @classmethod
    def load(cls, path=PIPELINE_PATH) -> "FraudScorer":
        """Load the joblib artifact written by src.train."""
        return cls(joblib.load(path))

    def build(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature steps as training (no leave-one-out: serving rows are not in training)."""
        return add_model_inputs(build_features(transactions, self.profiles))

    def score_frame(self, transactions: pd.DataFrame) -> np.ndarray:
        """Fraud probability (column 1 of predict_proba) for every row of a DataFrame."""
        return self.pipeline.predict_proba(self.build(transactions)[self.features])[:, 1]

    def score(self, transaction: dict) -> tuple:
        """Returns (risk_score, feature snapshot) for one transaction."""
        features = self.build(pd.DataFrame([transaction]))
        risk_score = float(self.pipeline.predict_proba(features[self.features])[:, 1][0])
        snapshot = {field: to_json_value(value) for field, value in features.iloc[0][SNAPSHOT_FIELDS].items()}
        # to_json_value rounds floats, so restore the count to a plain int.
        snapshot["customer_txn_count"] = int(snapshot["customer_txn_count"])
        return risk_score, snapshot


def to_json_value(value):
    """numpy scalars and NaN -> plain JSON-serialisable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 4)
    return value
