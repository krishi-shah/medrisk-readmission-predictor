"""Tests for SHAP-based explainability."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.models.explainer import compute_shap_values, explain_prediction, get_top_reasons


@pytest.fixture(scope="module")
def trained_lr_and_data():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model, X


def test_compute_shap_values_correct_row_count(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    shap_vals = compute_shap_values(model, X[:5], feature_names)
    assert shap_vals.values.shape[0] == 5


def test_compute_shap_values_sets_feature_names(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    shap_vals = compute_shap_values(model, X[:3], feature_names)
    assert shap_vals.feature_names == feature_names


def test_get_top_reasons_returns_requested_count(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    shap_vals = compute_shap_values(model, X[:1], feature_names)
    for top_n in (1, 3, 5):
        reasons = get_top_reasons(shap_vals, feature_names, index=0, top_n=top_n)
        assert len(reasons) == top_n


def test_get_top_reasons_structure(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    shap_vals = compute_shap_values(model, X[:1], feature_names)
    reasons = get_top_reasons(shap_vals, feature_names, index=0, top_n=3)
    for r in reasons:
        assert "feature" in r
        assert "impact" in r
        assert "direction" in r
        assert isinstance(r["feature"], str)
        assert isinstance(r["impact"], float)
        assert r["direction"] in ("increased risk", "decreased risk")


def test_get_top_reasons_sorted_descending_by_impact(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    shap_vals = compute_shap_values(model, X[:1], feature_names)
    reasons = get_top_reasons(shap_vals, feature_names, index=0, top_n=5)
    impacts = [r["impact"] for r in reasons]
    assert impacts == sorted(impacts, reverse=True)


def test_explain_prediction_returns_shap_and_reasons(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    result = explain_prediction(model, X[0], feature_names)
    assert "shap_values" in result
    assert "top_reasons" in result
    assert len(result["top_reasons"]) == 3


def test_explain_prediction_accepts_2d_input(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    result = explain_prediction(model, X[0:1], feature_names)
    assert len(result["top_reasons"]) == 3


def test_explain_prediction_impact_values_nonnegative(trained_lr_and_data):
    model, X = trained_lr_and_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    result = explain_prediction(model, X[0], feature_names)
    for reason in result["top_reasons"]:
        assert reason["impact"] >= 0.0
