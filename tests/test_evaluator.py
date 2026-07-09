import numpy as np
import pandas as pd

from src.models.evaluator import (
    check_quality_gates,
    compute_metrics,
    subgroup_recall_gaps,
)


def test_quality_gates_pass_for_good_metrics():
    metrics = {"pr_auc": 0.4, "recall": 0.8, "brier_score": 0.1}
    gates = {"min_pr_auc": 0.2, "min_recall": 0.7, "max_brier_score": 0.2}
    assert check_quality_gates(metrics, gates) == []


def test_quality_gates_detect_violations():
    metrics = {"pr_auc": 0.05, "recall": 0.6, "brier_score": 0.4}
    gates = {"min_pr_auc": 0.2, "min_recall": 0.7, "max_brier_score": 0.2}
    violations = check_quality_gates(metrics, gates)
    assert len(violations) == 3


def test_subgroup_recall_gap():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 1])
    groups = pd.Series(["A", "A", "B", "B"])
    recalls = subgroup_recall_gaps(y_true, y_pred, groups)
    assert "max_recall_gap" in recalls
    assert recalls["max_recall_gap"] == 0.5


def test_compute_metrics_includes_brier_score():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_prob, y_pred)
    assert "brier_score" in metrics
