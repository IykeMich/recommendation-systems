"""Train, compare and save the fraud-risk model.

Run from backend/: python -m src.train
"""
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ARTIFACT_DIR,
    DATA_QUALITY_PATH,
    FEATURE_SCHEMA_VERSION,
    MANIFEST_PATH,
    METRICS_PATH,
    MODEL_NAME,
    PIPELINE_PATH,
    POLICY_PATH,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_PATH,
    THRESHOLD_TABLE_PATH,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from .evaluate import decision_mix, score_metrics, threshold_table
from .explain import build_reference
from .features import CATEGORICAL, NUMERIC, CustomerProfiles, build_features
from .policy import load_policy, save_policy
from .schema import load_clean_transactions


def build_preprocessor(numeric, categorical):
    """Median-impute + scale numeric columns; mode-impute + one-hot encode categorical columns.

    handle_unknown="ignore" lets serving see unseen categories; min_frequency=2 groups rare ones.
    """
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical),
    ])


def candidate_models(numeric=NUMERIC, categorical=CATEGORICAL, class_weighted=True):
    """Same preprocessing contract for every candidate; only the estimator changes."""
    balanced = "balanced" if class_weighted else None
    return {
        "logistic_regression": (numeric, categorical, LogisticRegression(max_iter=2000, class_weight=balanced)),
        # Hour as a category lets a linear model score each hour separately.
        "logistic_regression_hour_categorical": (
            [feature for feature in numeric if feature not in ("hour", "hour_sin", "hour_cos")],
            categorical + ["hour_category"],
            LogisticRegression(max_iter=2000, class_weight=balanced),
        ),
        "random_forest": (numeric, categorical, RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            class_weight="balanced_subsample" if class_weighted else None,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        "hist_gradient_boosting": (numeric, categorical, HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, class_weight=balanced, random_state=RANDOM_STATE,
        )),
    }


def fit_pipeline(train_features, numeric, categorical, estimator):
    """Fit preprocessing + estimator together on the training features and the is_fraud label."""
    pipeline = Pipeline([("preprocess", build_preprocessor(numeric, categorical)), ("model", estimator)])
    pipeline.fit(train_features[numeric + categorical], train_features["is_fraud"].astype(int))
    return pipeline


def add_model_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Add hour as a string category (used by logistic_regression_hour_categorical)."""
    return features.assign(hour_category=features["hour"].astype(int).astype(str))


def split(transactions: pd.DataFrame):
    """Stratified random 70/15/15. The dataset has no timestamps and its row order
    carries no time signal, so a temporal split is not possible (see README)."""
    # stratify keeps the fraud rate the same in every split. Two steps: 70% train, then the
    # remaining 30% is divided 15/15 (0.15 / 0.30 = 0.5 of the rest goes to validation).
    train, rest = train_test_split(
        transactions, train_size=TRAIN_FRACTION, stratify=transactions["is_fraud"], random_state=RANDOM_STATE
    )
    validation_share_of_rest = VALIDATION_FRACTION / (1 - TRAIN_FRACTION)
    validation, test = train_test_split(
        rest, train_size=validation_share_of_rest, stratify=rest["is_fraud"], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def featurize(train, validation, test):
    """Fit customer profiles on the training split only, then build features for all three splits.

    Training rows use leave-one-out so a row's own transaction is not part of its profile.
    """
    profiles = CustomerProfiles().fit(train)
    return (
        profiles,
        add_model_inputs(build_features(train, profiles, leave_one_out=True)),
        add_model_inputs(build_features(validation, profiles)),
        add_model_inputs(build_features(test, profiles)),
    )


def synthetic_rule_scores(frame: pd.DataFrame) -> np.ndarray:
    """The pattern the synthetic label was generated with, found during inspection.
    A reference ceiling, not a fair competitor: it was read off the whole dataset."""
    return (frame["hour"].between(1, 4) & (frame["velocity_1h"] >= 6)).astype(float).to_numpy()


def train():
    """Full training run: quality checks, split, model comparison, calibration, artifacts.

    Returns the metrics dict that is also written to artifacts/metrics.json.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    transactions, quality_report = load_clean_transactions(RAW_PATH)
    DATA_QUALITY_PATH.write_text(json.dumps(quality_report, indent=2))

    train_rows, validation_rows, test_rows = split(transactions)
    for name, rows in [("train", train_rows), ("validation", validation_rows), ("test", test_rows)]:
        rows.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    profiles, train_features, validation_features, test_features = featurize(train_rows, validation_rows, test_rows)

    # Reuse an existing policy.json (it may have been edited from the dashboard); create it if missing.
    policy = load_policy()
    if not POLICY_PATH.exists():
        save_policy(policy)

    # Fit every candidate and score it on validation only.
    comparison = {}
    fitted = {}
    for model_name, (numeric, categorical, estimator) in candidate_models().items():
        pipeline = fit_pipeline(train_features, numeric, categorical, estimator)
        fitted[model_name] = (pipeline, numeric, categorical)
        comparison[model_name] = {
            "validation": score_metrics(validation_features["is_fraud"], pipeline.predict_proba(validation_features[numeric + categorical])[:, 1], policy),
        }

    # Model selection uses validation PR-AUC only; the test set is scored once, afterwards.
    selected_name = max(comparison, key=lambda name: comparison[name]["validation"]["pr_auc"])
    for model_name, (pipeline, numeric, categorical) in fitted.items():
        comparison[model_name]["test"] = score_metrics(
            test_features["is_fraud"], pipeline.predict_proba(test_features[numeric + categorical])[:, 1], policy
        )
    # The hand-written rule is scored on the same splits as a ceiling for comparison.
    comparison["synthetic_rule_reference"] = {
        split_name: score_metrics(frame["is_fraud"], synthetic_rule_scores(frame), policy)
        for split_name, frame in [("validation", validation_features), ("test", test_features)]
    }

    # Class weighting pushes suspicious scores towards 1, so they stop meaning
    # "probability of fraud". Calibrating (5-fold, on training data only) restores
    # that meaning, so policy thresholds like 0.55 / 0.85 are interpretable.
    uncalibrated_pipeline, numeric, categorical = fitted[selected_name]
    features = numeric + categorical
    pipeline = CalibratedClassifierCV(clone(uncalibrated_pipeline), method="isotonic", cv=5)
    pipeline.fit(train_features[features], train_features["is_fraud"].astype(int))
    for split_name, frame in [("validation", validation_features), ("test", test_features)]:
        comparison.setdefault(f"{selected_name}_calibrated", {})[split_name] = score_metrics(
            frame["is_fraud"], pipeline.predict_proba(frame[features])[:, 1], policy
        )
    uncalibrated_test_scores = uncalibrated_pipeline.predict_proba(test_features[features])[:, 1]
    validation_scores = pipeline.predict_proba(validation_features[features])[:, 1]
    test_scores = pipeline.predict_proba(test_features[features])[:, 1]
    # Threshold table and test predictions come from the calibrated (served) model.
    thresholds = threshold_table(validation_features["is_fraud"], validation_scores)
    thresholds.to_csv(THRESHOLD_TABLE_PATH, index=False)
    test_rows.assign(risk_score=test_scores, decision=[policy.decide(score) for score in test_scores]).to_csv(
        PREDICTIONS_DIR / "test_predictions.csv", index=False
    )

    trained_at = datetime.now(timezone.utc)
    model_version = f"fraud-{trained_at:%Y%m%d%H%M%S}"
    # One artifact holds everything scoring needs: pipeline, feature lists, profiles and the
    # explanation reference. Validation scores/labels are kept for policy simulation in the API.
    joblib.dump({
        "pipeline": pipeline,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "target": "is_fraud",
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "selected_model": f"{selected_name}_calibrated",
        "profiles": profiles,
        "explanation_reference": build_reference(train_features),
        "validation_scores": validation_scores,
        "validation_labels": validation_features["is_fraud"].to_numpy(),
    }, PIPELINE_PATH)

    # Human-readable lineage: what was trained, on which data (file hash) and under which policy.
    manifest = {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "selected_model": f"{selected_name}_calibrated",
        "calibration": "isotonic, 5-fold CalibratedClassifierCV on the training split",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": {"numeric": numeric, "categorical": categorical},
        "training_data_snapshot": {
            "file": str(RAW_PATH.relative_to(RAW_PATH.parents[2])),
            "sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
            "rows": int(len(transactions)),
        },
        "threshold_policy_version": policy.version,
        "trained_at": trained_at.isoformat(timespec="seconds"),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    metrics = {
        "model_version": model_version,
        "selected_model": selected_name,
        "selection_rule": "highest validation PR-AUC; test scored once afterwards",
        "split": {
            "method": "stratified random 70/15/15 (no timestamps in dataset)",
            **{name: {"rows": int(len(rows)), "fraud_rate": round(float(rows["is_fraud"].mean()), 4)}
               for name, rows in [("train", train_rows), ("validation", validation_rows), ("test", test_rows)]},
        },
        "policy": {"version": policy.version, "review_threshold": policy.review_threshold, "block_threshold": policy.block_threshold},
        "models": comparison,
        "served_model": f"{selected_name}_calibrated",
        "test_decision_mix": decision_mix(test_scores, policy),
        "test_decision_mix_uncalibrated": decision_mix(uncalibrated_test_scores, policy),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


# CLI entry point: train, then print a comparison table of all models.
if __name__ == "__main__":
    result = train()
    print(f"selected: {result['selected_model']}, served: {result['served_model']} ({result['model_version']})")
    print(f"{'model':<44} {'val PR-AUC':>10} {'test PR-AUC':>11} {'test ROC-AUC':>12} {'brier':>7} {'recall@review':>14} {'precision@review':>17}")
    for name, splits in result["models"].items():
        test = splits["test"]
        print(f"{name:<44} {splits['validation']['pr_auc']:>10.4f} {test['pr_auc']:>11.4f} {test['roc_auc']:>12.4f} {test['brier_score']:>7.4f} "
              f"{test['at_review_threshold']['recall']:>14.4f} {test['at_review_threshold']['precision']:>17.4f}")
    print("test decision mix (served, calibrated):", result["test_decision_mix"])
    print("test decision mix (uncalibrated):      ", result["test_decision_mix_uncalibrated"])
