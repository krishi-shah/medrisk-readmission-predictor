"""Model evaluation utilities for 30-day readmission prediction.

Provides metric computation, threshold optimisation, and diagnostic
visualisations (confusion matrix, calibration curve).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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

    return metrics
