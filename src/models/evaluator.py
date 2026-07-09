"""Model evaluation utilities for 30-day readmission prediction.

Provides metric computation, threshold optimisation, and diagnostic
visualisations (confusion matrix, calibration curve).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute a standard set of binary-classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_prob:
        Predicted probabilities for the positive class.
    y_pred:
        Hard predictions (0/1).

    Returns
    -------
    dict[str, float]
        Metric name to value mapping.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.70,
) -> float:
    """Find the classification threshold that maximises precision at a minimum recall.

    Scans thresholds from 0.01 to 0.99 (step 0.01). Among those
    achieving ``>= min_recall``, the one with the highest precision is
    returned.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_prob:
        Predicted probabilities for the positive class.
    min_recall:
        Minimum acceptable recall.

    Returns
    -------
    float
        Optimal decision threshold.
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_precision = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= min_recall:
            prec = precision_score(y_true, y_pred, zero_division=0)
            if prec > best_precision:
                best_precision = prec
                best_threshold = float(t)

    logger.info(
        "Optimal threshold: %.2f (precision=%.4f at min_recall=%.2f)",
        best_threshold,
        best_precision,
        min_recall,
    )
    if best_precision < 0:
        logger.warning(
            "No threshold achieved min_recall=%.2f; defaulting to %.2f",
            min_recall,
            best_threshold,
        )
    return best_threshold


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot a confusion matrix heatmap and optionally save to disk.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    save_path:
        If provided, the figure is saved to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Readmit", "Readmit"],
        yticklabels=["No Readmit", "Readmit"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved to %s", save_path)

    plt.close(fig)


def generate_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot a reliability diagram and optionally save to disk.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_prob:
        Predicted probabilities for the positive class.
    save_path:
        If provided, the figure is saved to this path.
    """
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted, fraction_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Calibration curve saved to %s", save_path)

    plt.close(fig)


def generate_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC curve and save to disk."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4361ee", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("ROC curve saved to %s", save_path)

    plt.close(fig)


def generate_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot Precision-Recall curve and save to disk."""
    from sklearn.metrics import precision_recall_curve

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="#e74c3c", lw=2,
            label=f"PR curve (AP = {pr_auc:.3f})")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, linestyle="--", color="gray", label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("PR curve saved to %s", save_path)

    plt.close(fig)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    save_dir: str | Path | None = None,
) -> dict[str, float]:
    """Run full evaluation: metrics, confusion matrix, and calibration curve.

    Parameters
    ----------
    model:
        Trained scikit-learn-compatible classifier.
    X_test:
        Test features.
    y_test:
        Test labels.
    threshold:
        Decision threshold for converting probabilities to hard labels.
    save_dir:
        If provided, diagnostic plots are saved to this directory.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_prob, y_pred)
    logger.info(
        "Evaluation — ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f | Brier: %.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
        metrics["brier_score"],
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        generate_confusion_matrix(y_test, y_pred, save_dir / "confusion_matrix.png")
        generate_calibration_curve(y_test, y_prob, save_dir / "calibration_curve.png")
        generate_roc_curve(y_test, y_prob, save_dir / "roc_curve.png")
        generate_pr_curve(y_test, y_prob, save_dir / "pr_curve.png")

    return metrics


def save_metrics_report(
    metrics: dict[str, float],
    threshold: float,
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist evaluation metrics to a machine-readable JSON file."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if extra:
        payload.update(extra)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("Metrics report saved to %s", output_path)
    return output_path


def check_quality_gates(
    metrics: dict[str, float],
    gates: dict[str, float],
) -> list[str]:
    """Return a list of quality-gate violations for aggregate metrics."""
    violations: list[str] = []
    if "min_pr_auc" in gates and metrics.get("pr_auc", 0.0) < gates["min_pr_auc"]:
        violations.append(
            f"PR-AUC {metrics.get('pr_auc', 0.0):.4f} below minimum {gates['min_pr_auc']:.4f}",
        )
    if "min_recall" in gates and metrics.get("recall", 0.0) < gates["min_recall"]:
        violations.append(
            f"Recall {metrics.get('recall', 0.0):.4f} below minimum {gates['min_recall']:.4f}",
        )
    if "max_brier_score" in gates and metrics.get("brier_score", 1.0) > gates["max_brier_score"]:
        violations.append(
            f"Brier score {metrics.get('brier_score', 1.0):.4f} above maximum {gates['max_brier_score']:.4f}",
        )
    return violations


def subgroup_recall_gaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_values: pd.Series,
) -> dict[str, float]:
    """Compute subgroup recall values and max-min recall gap."""
    recalls: dict[str, float] = {}
    for value in group_values.dropna().unique():
        mask = (group_values == value).to_numpy()
        yt = y_true[mask]
        yp = y_pred[mask]
        recalls[str(value)] = float(recall_score(yt, yp, zero_division=0))
    if not recalls:
        return {"max_recall_gap": 0.0}
    max_gap = max(recalls.values()) - min(recalls.values())
    recalls["max_recall_gap"] = float(max_gap)
    return recalls


def compute_multi_subgroup_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups_df: pd.DataFrame,
    group_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Compute subgroup recall gaps across multiple demographic dimensions.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_pred:
        Hard predictions (0/1).
    groups_df:
        DataFrame whose columns contain the grouping variables.
    group_columns:
        Column names in *groups_df* to evaluate. Columns not present in
        *groups_df* are silently skipped.

    Returns
    -------
    dict[str, dict[str, float]]
        Outer key: group column name.
        Inner dict: recall per group value plus ``"max_recall_gap"``.
    """
    results: dict[str, dict[str, float]] = {}
    for col in group_columns:
        if col not in groups_df.columns:
            logger.debug("Skipping fairness column '%s' — not in dataframe", col)
            continue
        results[col] = subgroup_recall_gaps(y_true, y_pred, groups_df[col].astype(str))
        logger.info(
            "Fairness [%s] — max recall gap: %.4f",
            col,
            results[col].get("max_recall_gap", 0.0),
        )
    return results
