"""Tests for model training utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification


from src.models.trainer import get_base_models


def test_get_base_models_returns_all_four():
    models = get_base_models(random_state=42)
    assert set(models.keys()) == {"logistic_regression", "random_forest", "xgboost", "lightgbm"}


def test_get_base_models_have_fit_and_predict_proba():
    models = get_base_models(random_state=42)
    for name, model in models.items():
        assert hasattr(model, "fit"), f"{name} missing fit()"
        assert hasattr(model, "predict_proba"), f"{name} missing predict_proba()"


def test_get_base_models_different_seeds_share_same_keys():
    models_a = get_base_models(random_state=0)
    models_b = get_base_models(random_state=99)
    assert set(models_a.keys()) == set(models_b.keys())


def test_base_models_fit_and_produce_valid_probabilities():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    models = get_base_models(random_state=42)
    for name, model in models.items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2), f"{name}: unexpected proba shape {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0), f"{name}: probabilities do not sum to 1"
        assert np.all(proba >= 0) and np.all(proba <= 1), f"{name}: probabilities out of [0, 1]"


def test_base_models_predict_binary_classes():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    models = get_base_models(random_state=42)
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1}), f"{name}: predictions contain values outside {{0, 1}}"
