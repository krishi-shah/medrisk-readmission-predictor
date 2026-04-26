"""FastAPI application for the MedRisk readmission-risk prediction service."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import get_serving_config
from src.utils.logger import get_logger

from .predictor import ModelPredictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)

logger = get_logger(__name__)

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
        result = predictor.predict(patient.model_dump())
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
