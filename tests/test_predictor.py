import pytest
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path

from src.serving.predictor import ModelPredictor


@pytest.fixture
def mock_model_artifacts(tmp_path):
    """Create minimal mock model artifacts for testing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    numeric_features = ["age_ordinal", "time_in_hospital", "num_lab_procedures",
                        "num_procedures", "num_medications", "number_diagnoses",
                        "num_med_changes", "num_meds_active", "total_prior_visits"]
    categorical_features = ["race", "gender", "diag1_category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ]
    )

    train_data = pd.DataFrame({
        "age_ordinal": [5, 6, 7, 4, 5, 6, 7, 8],
        "time_in_hospital": [3, 5, 7, 2, 4, 6, 8, 3],
        "num_lab_procedures": [40, 44, 52, 28, 60, 35, 50, 42],
        "num_procedures": [1, 1, 3, 0, 2, 0, 2, 1],
        "num_medications": [15, 18, 22, 10, 16, 12, 20, 14],
        "number_diagnoses": [7, 9, 8, 5, 9, 7, 10, 6],
        "num_med_changes": [1, 3, 2, 0, 2, 1, 3, 1],
        "num_meds_active": [5, 8, 6, 3, 7, 4, 8, 5],
        "total_prior_visits": [1, 4, 3, 0, 6, 1, 5, 2],
        "race": ["Caucasian", "AfricanAmerican", "Caucasian", "Hispanic",
                 "Caucasian", "Asian", "Caucasian", "AfricanAmerican"],
        "gender": ["Female", "Male", "Female", "Male",
                   "Female", "Male", "Female", "Male"],
        "diag1_category": ["Circulatory", "Diabetes", "Respiratory", "Other",
                           "Circulatory", "Diabetes", "Injury", "Digestive"],
    })
    y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])

    X_processed = preprocessor.fit_transform(train_data)
    model = LogisticRegression(random_state=42)
    model.fit(X_processed, y_train)

    model_path = tmp_path / "best_model.joblib"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    threshold_path = tmp_path / "threshold.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    with open(threshold_path, "w") as f:
        json.dump({"threshold": 0.3, "min_recall": 0.7}, f)

    return {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "threshold_path": str(threshold_path),
        "model_version": "test-v1.0",
    }


@pytest.fixture
def predictor(mock_model_artifacts):
    return ModelPredictor(**mock_model_artifacts)


@pytest.fixture
def sample_features():
    return {
        "age_ordinal": 7,
        "time_in_hospital": 5,
        "num_lab_procedures": 44,
        "num_procedures": 1,
        "num_medications": 18,
        "number_diagnoses": 9,
        "num_med_changes": 3,
        "num_meds_active": 8,
        "total_prior_visits": 4,
        "insulin_changed": 1,
        "high_utilizer": 0,
        "medical_specialty_missing": 1,
        "race": "Caucasian",
        "gender": "Female",
        "diag1_category": "Circulatory",
        "diag2_category": "Diabetes",
        "diag3_category": "Other",
        "admission_type_id": "1",
    }


class TestModelPredictor:
    def test_risk_score_is_float(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert isinstance(result["risk_score"], float)

    def test_risk_score_between_0_and_1(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_tier_valid(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert result["risk_tier"] in ("Low", "Medium", "High")

    def test_top_reasons_count(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert len(result["top_reasons"]) == 3

    def test_top_reasons_structure(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        for reason in result["top_reasons"]:
            assert "feature" in reason
            assert "impact" in reason
            assert "direction" in reason
            assert isinstance(reason["feature"], str)
            assert isinstance(reason["impact"], float)
            assert reason["direction"] in ("increased risk", "decreased risk")

    def test_model_version_present(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert result["model_version"] == "test-v1.0"
        assert len(result["model_version"]) > 0

    def test_batch_prediction(self, predictor, sample_features):
        results = predictor.predict_batch([sample_features, sample_features])
        assert len(results) == 2
        for result in results:
            assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_tier_low(self, predictor):
        assert predictor._get_risk_tier(0.1) == "Low"
        assert predictor._get_risk_tier(0.29) == "Low"

    def test_risk_tier_medium(self, predictor):
        assert predictor._get_risk_tier(0.3) == "Medium"
        assert predictor._get_risk_tier(0.59) == "Medium"

    def test_risk_tier_high(self, predictor):
        assert predictor._get_risk_tier(0.6) == "High"
        assert predictor._get_risk_tier(0.99) == "High"
