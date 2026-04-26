"""Model loading and inference logic for MedRisk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.explainer import explain_prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Loads a trained pipeline and exposes single / batch prediction."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
        threshold_path: str | Path,
        model_version: str,
    ) -> None:
        self.model_version = model_version
        self.model = None
        self.preprocessor = None
        self.threshold: float = 0.5
        self.feature_names: list[str] = []

        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
        except FileNotFoundError:
            logger.warning("Model file not found at %s – running in degraded mode", model_path)

        try:
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_names = list(self.preprocessor.get_feature_names_out())
            logger.info("Preprocessor loaded from %s (%d features)", preprocessor_path, len(self.feature_names))
        except FileNotFoundError:
            logger.warning("Preprocessor file not found at %s – running in degraded mode", preprocessor_path)

        try:
            with open(threshold_path, "r", encoding="utf-8") as fh:
                self.threshold = float(json.load(fh).get("threshold", 0.5))
            logger.info("Decision threshold loaded: %.4f", self.threshold)
        except FileNotFoundError:
            logger.warning("Threshold file not found at %s – using default 0.5", threshold_path)

    @property
    def is_ready(self) -> bool:
        """Return *True* when both model and preprocessor are loaded."""
        return self.model is not None and self.preprocessor is not None

    @staticmethod
    def _get_risk_tier(score: float) -> str:
        """Map a probability score to a human-readable risk tier."""
        if score < 0.3:
            return "Low"
        if score < 0.6:
            return "Medium"
        return "High"

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference for a single patient.

        Parameters
        ----------
        features:
            Dictionary of patient features matching :class:`PatientFeatures`.

        Returns
        -------
        dict
            Keys compatible with :class:`PredictionResponse`.
        """
        df = pd.DataFrame([features])
        X_transformed = self.preprocessor.transform(df)
        proba: float = float(self.model.predict_proba(X_transformed)[0, 1])

        readmitted: int = int(proba >= self.threshold)

        explanation = explain_prediction(
            self.model,
            X_transformed[0] if isinstance(X_transformed, np.ndarray) else X_transformed.iloc[0],
            self.feature_names,
        )

        return {
            "risk_score": round(proba, 4),
            "risk_tier": self._get_risk_tier(proba),
            "readmitted": readmitted,
            "top_reasons": explanation["top_reasons"],
            "model_version": self.model_version,
        }

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run inference for multiple patients.

        Parameters
        ----------
        features_list:
            List of patient-feature dictionaries.

        Returns
        -------
        list[dict]
            One :class:`PredictionResponse`-compatible dict per patient.
        """
        return [self.predict(f) for f in features_list]
