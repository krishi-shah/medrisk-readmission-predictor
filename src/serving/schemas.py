"""Pydantic v2 request / response schemas for the MedRisk prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 500


class PatientFeatures(BaseModel):
    """Input features for a single patient encounter."""

    age_ordinal: int = Field(..., ge=0, le=9)
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(..., ge=0, le=132)
    num_procedures: int = Field(0, ge=0, le=6)
    num_medications: int = Field(..., ge=0, le=81)
    number_diagnoses: int = Field(..., ge=1, le=16)
    num_med_changes: int = Field(..., ge=0, le=23)
    num_meds_active: int = Field(..., ge=0, le=23)
    total_prior_visits: int = Field(..., ge=0, le=100)
    insulin_changed: int = Field(..., ge=0, le=1)
    high_utilizer: int = Field(0, ge=0, le=1)
    medical_specialty_missing: int = Field(1, ge=0, le=1)
    race: str = Field("Unknown")
    gender: str
    diag1_category: str = Field("Other")
    diag2_category: str = Field("Other")
    diag3_category: str = Field("Other")
    admission_type_id: str = Field("1")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "age_ordinal": 6,
                "time_in_hospital": 3,
                "num_lab_procedures": 44,
                "num_procedures": 1,
                "num_medications": 13,
                "number_diagnoses": 9,
                "num_med_changes": 2,
                "num_meds_active": 7,
                "total_prior_visits": 1,
                "insulin_changed": 0,
                "high_utilizer": 0,
                "medical_specialty_missing": 1,
                "race": "Caucasian",
                "gender": "Female",
                "diag1_category": "Circulatory",
                "diag2_category": "Diabetes",
                "diag3_category": "Other",
                "admission_type_id": "1",
            }
        ]
    }}


class ReasonResponse(BaseModel):
    """A single SHAP-based reason driving the prediction."""

    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    """Result of a readmission-risk prediction for one patient."""

    risk_score: float
    risk_tier: Literal["Low", "Medium", "High"]
    readmitted: int = Field(0, ge=0, le=1, description="Binary prediction using tuned threshold (1 = readmitted)")
    top_reasons: list[ReasonResponse]
    model_version: str


class HealthResponse(BaseModel):
    """API health-check payload."""

    status: str
    model_version: str
    last_updated: str


class BatchPredictionRequest(BaseModel):
    """Wrapper for multiple patient predictions in one call."""

    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    """Wrapper for multiple prediction results."""

    predictions: list[PredictionResponse]


class ErrorResponse(BaseModel):
    """Structured error payload returned on 4xx/5xx responses."""

    error_code: str
    detail: str
