"""SHAP-based model explainability for 30-day readmission predictions.

Supports both tree-based (XGBoost, LightGBM, Random Forest) and linear
(Logistic Regression) models via the appropriate SHAP explainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Compute SHAP values for the given model and data.

    Uses :class:`shap.TreeExplainer` for tree-based models and
    :class:`shap.LinearExplainer` for logistic regression.

    Parameters
    ----------
    model:
        Trained scikit-learn-compatible classifier.
    X:
        Feature matrix (2-D array).
    feature_names:
        Optional list of feature names matching columns of *X*.

    Returns
    -------
    shap.Explanation
        SHAP explanation object containing values, base values, and data.
    """
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    if feature_names is not None:
        shap_values.feature_names = feature_names

    logger.info("Computed SHAP values for %d samples", X.shape[0])
    return shap_values


def generate_summary_plot(
    shap_values: shap.Explanation,
    X: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Create a SHAP beeswarm summary plot and optionally save to disk.

    Parameters
    ----------
    shap_values:
        SHAP explanation object.
    X:
        Feature matrix used for colour-coding feature values.
    save_path:
        If provided, the figure is saved to this path.
    """
    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, show=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to %s", save_path)

    plt.close(fig)


def generate_waterfall_plot(
    shap_values: shap.Explanation,
    index: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Create a SHAP waterfall plot for a single observation.

    Parameters
    ----------
    shap_values:
        SHAP explanation object.
    index:
        Row index of the observation to explain.
    save_path:
        If provided, the figure is saved to this path.
    """
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], show=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP waterfall plot saved to %s", save_path)

    plt.close(fig)


def get_top_reasons(
    shap_values: shap.Explanation,
    feature_names: list[str],
    index: int = 0,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Return the top contributing features for a single prediction.

    Parameters
    ----------
    shap_values:
        SHAP explanation object.
    feature_names:
        List of feature names matching the columns of the dataset.
    index:
        Row index of the observation.
    top_n:
        Number of top features to return.

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains ``feature``, ``impact`` (absolute SHAP value),
        and ``direction`` (``"increased risk"`` or ``"decreased risk"``).
    """
    values = shap_values[index].values
    if values.ndim > 1:
        values = values[:, 1]

    abs_values = np.abs(values)
    top_indices = np.argsort(abs_values)[::-1][:top_n]

    reasons: list[dict[str, Any]] = []
    for i in top_indices:
        reasons.append({
            "feature": feature_names[i],
            "impact": float(abs_values[i]),
            "direction": "increased risk" if values[i] > 0 else "decreased risk",
        })

    return reasons


def explain_prediction(
    model: Any,
    X_single: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Generate a full SHAP explanation for a single patient.

    Parameters
    ----------
    model:
        Trained classifier.
    X_single:
        Feature array for one patient (1-D or 2-D with one row).
    feature_names:
        List of feature names.

    Returns
    -------
    dict[str, Any]
        ``shap_values`` (:class:`shap.Explanation`) and ``top_reasons``
        (list of contributing-feature dicts).
    """
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    shap_values = compute_shap_values(model, X_single, feature_names)
    top_reasons = get_top_reasons(shap_values, feature_names, index=0)

    logger.info(
        "Top reason for prediction: %s (%s)",
        top_reasons[0]["feature"],
        top_reasons[0]["direction"],
    )

    return {
        "shap_values": shap_values,
        "top_reasons": top_reasons,
    }
