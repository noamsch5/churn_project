"""
Evaluation metrics for churn classification and business performance.
Includes standard ML metrics, ROC/PR curve points, lift, and threshold selection.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Computes complete classification metrics at a given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Business top-decile metrics (Top 10% highest predicted risk)
    n_top_10_pct = max(1, int(len(y_true) * 0.10))
    top_10_indices = np.argsort(y_prob)[::-1][:n_top_10_pct]
    top_10_actual = y_true[top_10_indices]
    
    overall_churn_rate = float(np.mean(y_true))
    precision_at_10 = float(np.mean(top_10_actual))
    recall_at_10 = float(np.sum(top_10_actual) / max(1, np.sum(y_true)))
    lift_at_10 = float(precision_at_10 / overall_churn_rate) if overall_churn_rate > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "business_metrics": {
            "precision_at_10_pct": round(precision_at_10, 4),
            "recall_at_10_pct": round(recall_at_10, 4),
            "lift_at_10_pct": round(lift_at_10, 2),
        }
    }


def compute_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 100,
) -> dict[str, list[dict[str, float]]]:
    """
    Computes smoothed ROC and PR curve points for frontend chart rendering.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # Subsample points uniformly if too many, to keep JSON lightweight
    roc_step = max(1, len(fpr) // n_points)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), roc_step)
    ]
    if roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    pr_step = max(1, len(recall) // n_points)
    pr_points = [
        {"recall": round(float(recall[i]), 4), "precision": round(float(precision[i]), 4)}
        for i in range(0, len(recall), pr_step)
    ]

    return {
        "roc_curve": roc_points,
        "pr_curve": pr_points,
    }


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, list[dict[str, float]]]:
    """
    Sweeps thresholds from 0.05 to 0.95 with step 0.01.
    Selects the threshold maximizing the specified metric. The caller is responsible
    for supplying validation or out-of-fold training predictions, never test data.
    Returns (best_threshold, sweep_data).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    sweep_results = []
    best_thresh = 0.5
    best_val = -1.0

    for t in thresholds:
        m = compute_classification_metrics(y_true, y_prob, threshold=t)
        score = m[metric]
        sweep_results.append({
            "threshold": round(float(t), 2),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
        })
        if score > best_val:
            best_val = score
            best_thresh = float(t)

    return round(best_thresh, 2), sweep_results
