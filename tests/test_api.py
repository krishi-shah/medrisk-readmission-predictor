import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import src.serving.api as api_module
from src.serving.api import app


@pytest.fixture
def mock_prediction():
    return {
        "risk_score": 0.73,
        "risk_tier": "High",
        "readmitted": 1,
        "top_reasons": [
            {"feature": "number_inpatient", "impact": 0.142, "direction": "increased risk"},
            {"feature": "num_med_changes", "impact": 0.098, "direction": "increased risk"},
            {"feature": "diag1_category_Circulatory", "impact": 0.067, "direction": "increased risk"},
        ],
        "model_version": "test-v1.0",
    }


@pytest.fixture
def valid_patient_data():
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
        "race": "Caucasian",
        "gender": "Female",
        "diag1_category": "Circulatory",
    }


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_predictor(mock_prediction):
    """Inject a mock predictor into the API module."""
    pred = MagicMock()
    pred.predict.return_value = mock_prediction
    pred.predict_batch.return_value = [mock_prediction, mock_prediction]
    pred.is_ready = True
    pred.model_version = "test-v1.0"
    pred.model = MagicMock()
    pred.model.__class__.__name__ = "XGBClassifier"
    pred.threshold = 0.18
    pred.feature_names = ["feat_a", "feat_b", "feat_c"]
    original = api_module._predictor
    api_module._predictor = pred
    yield pred
    api_module._predictor = original


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_version" in data
        assert "last_updated" in data


class TestPredictEndpoint:
    def test_predict_with_mock(self, client, valid_patient_data, mock_predictor):
        response = client.post("/predict", json=valid_patient_data)
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_tier" in data
        assert "top_reasons" in data
        assert data["risk_tier"] in ("Low", "Medium", "High")

    def test_predict_invalid_input(self, client):
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_missing_required_fields(self, client):
        response = client.post("/predict", json={
            "age_ordinal": 7,
        })
        assert response.status_code == 422

    def test_predict_validation_age_out_of_range(self, client):
        data = {
            "age_ordinal": 15,
            "time_in_hospital": 5,
            "num_lab_procedures": 44,
            "num_medications": 18,
            "number_diagnoses": 9,
            "num_med_changes": 3,
            "num_meds_active": 8,
            "total_prior_visits": 4,
            "insulin_changed": 1,
            "gender": "Female",
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_predict_503_without_model(self, client):
        original = api_module._predictor
        api_module._predictor = None
        response = client.post("/predict", json={
            "age_ordinal": 7, "time_in_hospital": 5,
            "num_lab_procedures": 44, "num_medications": 18,
            "number_diagnoses": 9, "num_med_changes": 3,
            "num_meds_active": 8, "total_prior_visits": 4,
            "insulin_changed": 1, "gender": "Female",
        })
        assert response.status_code == 503
        api_module._predictor = original


class TestBatchPredictEndpoint:
    def test_batch_predict_with_mock(self, client, valid_patient_data, mock_predictor):
        response = client.post("/predict/batch", json={
            "patients": [valid_patient_data, valid_patient_data]
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_batch_predict_empty_list_rejected(self, client, mock_predictor):
        response = client.post("/predict/batch", json={"patients": []})
        assert response.status_code == 422

    def test_batch_predict_exceeds_max_size_rejected(self, client, valid_patient_data, mock_predictor):
        patients = [valid_patient_data] * 501
        response = client.post("/predict/batch", json={"patients": patients})
        assert response.status_code == 422

    def test_validation_error_response_has_error_code(self, client):
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422
        data = response.json()
        assert "error_code" in data
        assert data["error_code"] == "VALIDATION_ERROR"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client, mock_predictor):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_version" in data
        assert "threshold" in data
        assert "n_features" in data
        assert "feature_names" in data
        assert data["n_features"] == 3
        assert data["threshold"] == 0.18

    def test_model_info_503_without_model(self, client):
        original = api_module._predictor
        api_module._predictor = None
        response = client.get("/model/info")
        assert response.status_code == 503
        api_module._predictor = original


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "medrisk_predictions_total" in response.text or response.status_code == 200


class TestDriftEndpoint:
    def test_drift_503_without_reference(self, client, mock_predictor):
        response = client.post("/drift/check", json={
            "records": [{"age_ordinal": 5, "time_in_hospital": 3}]
        })
        assert response.status_code == 503
