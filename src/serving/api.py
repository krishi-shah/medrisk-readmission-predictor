"""FastAPI application for the MedRisk readmission-risk prediction service."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.utils.config import get_serving_config, get_config
from src.utils.logger import get_logger

from .predictor import ModelPredictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    DriftFeatureReport,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)

logger = get_logger(__name__)

PREDICTION_COUNTER = Counter("medrisk_predictions_total", "Total predictions", ["risk_tier"])
PREDICTION_LATENCY = Histogram("medrisk_prediction_latency_seconds", "Prediction latency")
DRIFT_GAUGE = Gauge("medrisk_drift_detected", "Whether drift has been detected")

_predictor: ModelPredictor | None = None
_startup_time: str = ""


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Load the model on startup and clean up on shutdown."""
    global _predictor, _startup_time

    cfg = get_serving_config()
    _startup_time = datetime.now(timezone.utc).isoformat()

    predictor = ModelPredictor(
        model_path=cfg["model_path"],
        preprocessor_path=cfg["preprocessor_path"],
        threshold_path=cfg["threshold_path"],
        model_version=cfg["model_version"],
    )

    if predictor.is_ready:
        _predictor = predictor
        logger.info("ModelPredictor ready (version %s)", predictor.model_version)
    else:
        logger.warning("ModelPredictor not ready – API running in degraded mode")

    yield

    _predictor = None
    logger.info("Application shutdown complete")


app = FastAPI(
    title="MedRisk",
    description="Patient Readmission Risk Prediction API",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            detail=str(exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def _http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Log every incoming request with its latency."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _require_predictor() -> ModelPredictor:
    """Return the loaded predictor or raise 503."""
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The service is running in degraded mode.",
        )
    return _predictor


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and model metadata."""
    cfg = get_serving_config()
    return HealthResponse(
        status="healthy" if _predictor is not None else "degraded",
        model_version=cfg["model_version"],
        last_updated=_startup_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientFeatures) -> PredictionResponse:
    """Return a readmission-risk prediction for a single patient."""
    predictor = _require_predictor()
    try:
        with PREDICTION_LATENCY.time():
            result = predictor.predict(patient.model_dump())
        PREDICTION_COUNTER.labels(risk_tier=result["risk_tier"]).inc()
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.") from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(body: BatchPredictionRequest) -> BatchPredictionResponse:
    """Return readmission-risk predictions for a batch of patients."""
    predictor = _require_predictor()
    try:
        results = predictor.predict_batch([p.model_dump() for p in body.patients])
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed due to an internal error.") from exc
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results],
    )


@app.get("/metrics")
async def metrics():
    """Return Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    predictor = _require_predictor()
    cfg = get_serving_config()

    training_metrics = None
    try:
        metrics_path = Path("reports/metrics/latest_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as fh:
                training_metrics = json.load(fh)
    except Exception:
        pass

    return ModelInfoResponse(
        model_type=type(predictor.model).__name__,
        model_version=cfg["model_version"],
        threshold=predictor.threshold,
        n_features=len(predictor.feature_names),
        feature_names=predictor.feature_names,
        training_metrics=training_metrics,
    )


@app.post("/drift/check", response_model=DriftCheckResponse)
async def drift_check(body: DriftCheckRequest) -> DriftCheckResponse:
    """Run drift detection on incoming data against reference distribution."""
    import joblib

    try:
        reference_stats = joblib.load("models/reference_stats.joblib")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Reference stats not found. Run training pipeline first.",
        )

    cfg = get_config()
    numeric_features = cfg["features"]["numeric"]

    current_df = pd.DataFrame(body.records)

    from src.monitoring.drift import detect_drift, compute_psi, compute_ks_tests

    drift_report = detect_drift(reference_stats, current_df, numeric_features)
    psi_results = compute_psi(reference_stats, current_df, numeric_features)
    ks_results = compute_ks_tests(reference_stats, current_df, numeric_features)

    DRIFT_GAUGE.set(1.0 if drift_report["drift_detected"] else 0.0)

    feature_reports = []
    for feat in numeric_features:
        if feat not in current_df.columns:
            continue
        psi_score = psi_results.get(feat, 0.0)
        ks_stat = ks_results.get(feat, {}).get("statistic", 0.0)
        ks_p = ks_results.get(feat, {}).get("p_value", 1.0)
        feature_reports.append(DriftFeatureReport(
            feature=feat,
            psi_score=psi_score,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            drifted=psi_score > 0.2 or ks_p < 0.05,
        ))

    n_drifted = sum(1 for f in feature_reports if f.drifted)

    return DriftCheckResponse(
        drift_detected=drift_report["drift_detected"],
        n_features_drifted=n_drifted,
        features=feature_reports,
    )
