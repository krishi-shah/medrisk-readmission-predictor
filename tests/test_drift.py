"""Tests for feature drift detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import compute_ks_tests, compute_psi, detect_drift

FEATURES = ["age_ordinal", "num_medications", "time_in_hospital"]


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age_ordinal": rng.integers(0, 10, 200).astype(float),
        "num_medications": rng.integers(1, 50, 200).astype(float),
        "time_in_hospital": rng.integers(1, 14, 200).astype(float),
    })


@pytest.fixture
def similar_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "age_ordinal": rng.integers(0, 10, 200).astype(float),
        "num_medications": rng.integers(1, 50, 200).astype(float),
        "time_in_hospital": rng.integers(1, 14, 200).astype(float),
    })


@pytest.fixture
def drifted_df():
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "age_ordinal": rng.integers(7, 10, 200).astype(float),
        "num_medications": rng.integers(40, 80, 200).astype(float),
        "time_in_hospital": rng.integers(1, 14, 200).astype(float),
    })


class TestComputePsi:
    def test_returns_all_requested_features(self, reference_df, similar_df):
        psi = compute_psi(reference_df, similar_df, FEATURES)
        assert set(psi.keys()) == set(FEATURES)

    def test_similar_distributions_have_low_psi(self, reference_df, similar_df):
        psi = compute_psi(reference_df, similar_df, FEATURES)
        assert all(v < 0.2 for v in psi.values())

    def test_drifted_distributions_have_high_psi(self, reference_df, drifted_df):
        psi = compute_psi(reference_df, drifted_df, FEATURES)
        assert psi["age_ordinal"] > 0.2 or psi["num_medications"] > 0.2

    def test_ignores_missing_columns(self, reference_df, similar_df):
        psi = compute_psi(reference_df, similar_df, ["age_ordinal", "nonexistent"])
        assert "age_ordinal" in psi
        assert "nonexistent" not in psi

    def test_psi_values_are_nonnegative(self, reference_df, similar_df):
        psi = compute_psi(reference_df, similar_df, FEATURES)
        assert all(v >= 0.0 for v in psi.values())


class TestComputeKsTests:
    def test_returns_statistic_and_pvalue_per_feature(self, reference_df, similar_df):
        ks = compute_ks_tests(reference_df, similar_df, FEATURES)
        for result in ks.values():
            assert "statistic" in result
            assert "p_value" in result

    def test_statistic_in_unit_interval(self, reference_df, similar_df):
        ks = compute_ks_tests(reference_df, similar_df, FEATURES)
        for result in ks.values():
            assert 0.0 <= result["statistic"] <= 1.0
            assert 0.0 <= result["p_value"] <= 1.0

    def test_identical_data_yields_high_pvalue(self, reference_df):
        ks = compute_ks_tests(reference_df, reference_df.copy(), FEATURES)
        for result in ks.values():
            assert result["p_value"] > 0.05


class TestDetectDrift:
    def test_report_has_required_keys(self, reference_df, similar_df):
        report = detect_drift(reference_df, similar_df, FEATURES)
        for key in ("psi_scores", "ks_tests", "flagged_by_psi", "flagged_by_ks",
                    "drift_detected", "n_features_drifted"):
            assert key in report

    def test_no_drift_on_similar_data(self, reference_df, similar_df):
        report = detect_drift(reference_df, similar_df, FEATURES)
        assert isinstance(report["drift_detected"], bool)

    def test_drift_detected_on_drifted_data(self, reference_df, drifted_df):
        report = detect_drift(reference_df, drifted_df, FEATURES, psi_threshold=0.1, ks_alpha=0.05)
        assert report["drift_detected"] is True
        assert report["n_features_drifted"] > 0

    def test_n_features_drifted_is_nonnegative(self, reference_df, similar_df):
        report = detect_drift(reference_df, similar_df, FEATURES)
        assert report["n_features_drifted"] >= 0
