"""Feature drift detection using Population Stability Index (PSI) and KS tests.

PSI interpretation:
  < 0.1  — no significant change
  0.1–0.2 — moderate change (monitor)
  > 0.2  — significant change (consider retraining)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PSI_EPSILON = 1e-4


def _psi_single(expected: np.ndarray, actual: np.ndarray, bins: int) -> float:
    combined_min = min(expected.min(), actual.min())
    combined_max = max(expected.max(), actual.max())
    if combined_min == combined_max:
        return 0.0
    breakpoints = np.linspace(combined_min, combined_max, bins + 1)
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, _PSI_EPSILON, exp_pct)
    act_pct = np.where(act_pct == 0, _PSI_EPSILON, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    bins: int = 10,
) -> dict[str, float]:
    """Compute Population Stability Index for each numeric feature."""
    results: dict[str, float] = {}
    for feature in numeric_features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        ref_vals = reference[feature].dropna().to_numpy(dtype=float)
        cur_vals = current[feature].dropna().to_numpy(dtype=float)
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        results[feature] = _psi_single(ref_vals, cur_vals, bins)
    return results


def compute_ks_tests(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, dict[str, float]]:
    """Run two-sample Kolmogorov-Smirnov test for each numeric feature."""
    results: dict[str, dict[str, float]] = {}
    for feature in numeric_features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        ref_vals = reference[feature].dropna().to_numpy(dtype=float)
        cur_vals = current[feature].dropna().to_numpy(dtype=float)
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        results[feature] = {"statistic": float(stat), "p_value": float(p_value)}
    return results


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> dict[str, Any]:
    """Run comprehensive drift detection and return a summary report.

    Parameters
    ----------
    reference:
        Training-time baseline DataFrame.
    current:
        Current serving DataFrame to compare against baseline.
    numeric_features:
        List of numeric column names to evaluate.
    psi_threshold:
        PSI value above which a feature is flagged.
    ks_alpha:
        Significance level for KS test; features with p-value below this
        are flagged.

    Returns
    -------
    dict
        Report with ``psi_scores``, ``ks_tests``, ``flagged_by_psi``,
        ``flagged_by_ks``, ``drift_detected``, and ``n_features_drifted``.
    """
    psi_scores = compute_psi(reference, current, numeric_features)
    ks_results = compute_ks_tests(reference, current, numeric_features)

    flagged_psi = {f: v for f, v in psi_scores.items() if v > psi_threshold}
    flagged_ks = {f: v for f, v in ks_results.items() if v["p_value"] < ks_alpha}
    drifted = set(flagged_psi) | set(flagged_ks)

    report: dict[str, Any] = {
        "psi_scores": psi_scores,
        "ks_tests": ks_results,
        "flagged_by_psi": flagged_psi,
        "flagged_by_ks": flagged_ks,
        "drift_detected": len(drifted) > 0,
        "n_features_drifted": len(drifted),
    }

    if report["drift_detected"]:
        logger.warning(
            "Drift detected in %d features — PSI flags: %s | KS flags: %s",
            report["n_features_drifted"],
            list(flagged_psi),
            list(flagged_ks),
        )
    else:
        logger.info("No significant drift detected across %d features", len(numeric_features))

    return report
