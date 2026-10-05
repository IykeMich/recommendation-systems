"""Capstone experiments E01–E06 (guide section 24). E07 and E08 are API behaviours, tested in tests/test_api.py.

Run with: python -m src.experiments   (after python -m src.train)
Writes artifacts/experiment_log.csv; nothing else changes.
"""
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from .config import ARTIFACT_DIR, RANDOM_STATE, RAW_PATH
from .evaluate import decision_mix
from .features import CATEGORICAL, NOVELTY_FEATURES, NUMERIC, VELOCITY_FEATURES, CustomerProfiles, build_features
from .policy import Policy, load_policy
from .schema import load_clean_transactions
from .train import add_model_inputs, candidate_models, featurize, fit_pipeline, split

LOG_COLUMNS = ["experiment_id", "model", "features", "threshold", "pr_auc", "precision", "recall", "notes"]


def evaluate_model(model_name, train_features, test_features, numeric=NUMERIC, categorical=CATEGORICAL, class_weighted=True):
    """Fit one candidate model (optionally with a reduced feature set) and score the test features."""
    numeric_used, categorical_used, estimator = candidate_models(numeric, categorical, class_weighted)[model_name]
    pipeline = fit_pipeline(train_features, numeric_used, categorical_used, estimator)
    scores = pipeline.predict_proba(test_features[numeric_used + categorical_used])[:, 1]
    return pipeline, scores


def log_row(experiment_id, model, features, threshold, y_true, scores, notes):
    """One experiment_log.csv row: PR-AUC plus precision/recall at `threshold`."""
    predicted = (scores >= threshold).astype(int)
    return {
        "experiment_id": experiment_id, "model": model, "features": features, "threshold": threshold,
        "pr_auc": round(average_precision_score(y_true, scores), 4),
        "precision": round(precision_score(y_true, predicted, zero_division=0), 4),
        "recall": round(recall_score(y_true, predicted, zero_division=0), 4),
        "notes": notes,
    }


def main():
    """Run experiments E01-E06 on the standard split and write artifacts/experiment_log.csv."""
    transactions, _ = load_clean_transactions(RAW_PATH)
    train_rows, _, test_rows = split(transactions)
    # Validation is not needed here, so an empty frame (iloc[:0]) is passed in its place.
    _, train_features, _, test_features = featurize(train_rows, train_rows.iloc[:0], test_rows)
    y_test = test_features["is_fraud"]
    policy = load_policy()
    threshold = policy.review_threshold
    log = []

    print("E01 — remove velocity features")
    for label, numeric in [("baseline", NUMERIC), ("no_velocity", [f for f in NUMERIC if f not in VELOCITY_FEATURES])]:
        _, scores = evaluate_model("random_forest", train_features, test_features, numeric)
        log.append(log_row("E01", "random_forest", label, threshold, y_test, scores, "velocity ablation"))
        print(f"  {label:<14} PR-AUC {log[-1]['pr_auc']:.3f}  recall {log[-1]['recall']:.3f}")

    print("E02 — remove device/country novelty")
    # Test rows whose device type was never seen for that customer in training.
    new_device_rows = test_features["customer_device_share"] == 0
    for label, numeric in [("baseline", NUMERIC), ("no_novelty", [f for f in NUMERIC if f not in NOVELTY_FEATURES])]:
        _, scores = evaluate_model("random_forest", train_features, test_features, numeric)
        log.append(log_row("E02", "random_forest", label, threshold, y_test, scores, "novelty ablation"))
        on_new_device = average_precision_score(y_test[new_device_rows], scores[new_device_rows.to_numpy()])
        print(f"  {label:<14} PR-AUC {log[-1]['pr_auc']:.3f}  PR-AUC on new-device transactions {on_new_device:.3f}")

    print("E03 — change the review threshold, model fixed (policy only)")
    from .predict import FraudScorer
    scorer = FraudScorer.load()
    # Scores come from the saved calibrated model; only the policy threshold changes.
    served_scores = scorer.score_frame(test_rows)
    for review_threshold in [0.3, 0.55, 0.7, 0.75]:
        alternative = Policy(review_threshold=review_threshold, block_threshold=max(review_threshold, policy.block_threshold), version="experiment")
        log.append(log_row("E03", "served", "baseline", review_threshold, y_test, served_scores, "threshold change only"))
        print(f"  review ≥ {review_threshold:.2f}: precision {log[-1]['precision']:.3f} recall {log[-1]['recall']:.3f} "
              f"mix {decision_mix(served_scores, alternative)}")

    print("E04 — split design (the dataset has no timestamps, so no temporal split)")
    # Compare three ways of holding out 15%: stratified random, by row order, and by customer
    # (GroupShuffleSplit puts each customer wholly in train or test, testing unseen customers).
    order_cut = int(len(transactions) * 0.85)
    for label, (train_part, test_part) in {
        "stratified_random": (train_rows, test_rows),
        "transaction_id_order": (transactions.iloc[:order_cut], transactions.iloc[order_cut:]),
        "unseen_customers": tuple(transactions.iloc[index] for index in next(
            GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE).split(transactions, groups=transactions["customer_id"]))),
    }.items():
        profiles = CustomerProfiles().fit(train_part)
        _, scores = evaluate_model(
            "random_forest",
            add_model_inputs(build_features(train_part.reset_index(drop=True), profiles, leave_one_out=True)),
            add_model_inputs(build_features(test_part.reset_index(drop=True), profiles)),
        )
        log.append(log_row("E04", "random_forest", label, threshold, test_part["is_fraud"], scores, "split design"))
        print(f"  {label:<22} PR-AUC {log[-1]['pr_auc']:.3f}")

    print("E05 — balanced vs unbalanced class weights")
    for class_weighted in [True, False]:
        _, scores = evaluate_model("random_forest", train_features, test_features, class_weighted=class_weighted)
        label = "balanced" if class_weighted else "unbalanced"
        log.append(log_row("E05", "random_forest", label, threshold, y_test, scores, "class weighting"))
        print(f"  {label:<11} PR-AUC {log[-1]['pr_auc']:.3f}  mean score on fraud {scores[y_test.to_numpy() == 1].mean():.3f}  "
              f"mix {decision_mix(scores, policy)}")

    print("E06 — logistic regression vs random forest (same test set)")
    for model_name in ["logistic_regression", "logistic_regression_hour_categorical", "random_forest"]:
        _, scores = evaluate_model(model_name, train_features, test_features)
        log.append(log_row("E06", model_name, "baseline", threshold, y_test, scores, "model comparison"))
        print(f"  {model_name:<38} PR-AUC {log[-1]['pr_auc']:.3f}")

    print("E07 / E08 — retries and malformed records: covered by tests/test_api.py "
          "(test_duplicate_transaction_is_idempotent, test_malformed_records_are_rejected)")

    pd.DataFrame(log, columns=LOG_COLUMNS).to_csv(ARTIFACT_DIR / "experiment_log.csv", index=False)
    print(f"\nwrote {ARTIFACT_DIR / 'experiment_log.csv'}")


if __name__ == "__main__":
    main()
