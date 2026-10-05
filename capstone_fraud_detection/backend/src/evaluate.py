"""Evaluation helpers: ranking metrics, metrics at a threshold, threshold tables, decision mix."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .policy import Policy


def classification_metrics(y_true, scores, threshold: float) -> dict:
    """Precision, recall, F1, flag rate and confusion matrix when flagging scores >= threshold."""
    predicted = (np.asarray(scores) >= threshold).astype(int)
    # labels=[0, 1] keeps a 2x2 matrix even if one class is absent; ravel() gives tn, fp, fn, tp.
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "flag_rate": round(float(predicted.mean()), 4),
        "confusion_matrix": {"true_negative": int(true_negative), "false_positive": int(false_positive),
                             "false_negative": int(false_negative), "true_positive": int(true_positive)},
    }


def score_metrics(y_true, scores, policy: Policy) -> dict:
    """Threshold-free metrics (ROC-AUC, PR-AUC, Brier) plus metrics at the policy's two thresholds.

    PR-AUC is the headline metric because fraud is rare (~8%) and ROC-AUC flatters imbalanced data.
    """
    return {
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        # Lower is better; measures whether scores behave like probabilities.
        "brier_score": round(float(brier_score_loss(y_true, scores)), 4),
        "at_review_threshold": classification_metrics(y_true, scores, policy.review_threshold),
        "at_block_threshold": classification_metrics(y_true, scores, policy.block_threshold),
        "mean_score_fraud": round(float(np.mean(np.asarray(scores)[np.asarray(y_true) == 1])), 4),
        "mean_score_legitimate": round(float(np.mean(np.asarray(scores)[np.asarray(y_true) == 0])), 4),
    }


def threshold_table(y_true, scores) -> pd.DataFrame:
    """Metrics at thresholds 0.05, 0.10, ..., 0.95 to help choose policy thresholds."""
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        metrics = classification_metrics(y_true, scores, float(threshold))
        rows.append({key: metrics[key] for key in ["threshold", "precision", "recall", "f1", "flag_rate"]}
                    | metrics["confusion_matrix"])
    return pd.DataFrame(rows)


def decision_mix(scores, policy: Policy) -> dict:
    """Share of scores that the policy would ALLOW, REVIEW and BLOCK."""
    decisions = pd.Series([policy.decide(score) for score in scores])
    return {decision: round(float((decisions == decision).mean()), 4) for decision in ("ALLOW", "REVIEW", "BLOCK")}
