"""Evaluation entrypoint with reproducible artifacts and quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.trainer import run_training_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation(train_if_missing: bool = False, fail_on_gates: bool = False) -> int:
    """Run training/evaluation and optionally fail on quality-gate violations."""
    models_dir = Path("models")
    required = [
        models_dir / "best_model.joblib",
        models_dir / "preprocessor.joblib",
        models_dir / "threshold.json",
    ]
    if train_if_missing and not all(p.exists() for p in required):
        logger.info("Missing model artifacts detected. Running training pipeline first.")

    result = run_training_pipeline()
    logger.info("Evaluation artifacts written to %s", result["metrics_path"])
    if result["quality_gate_violations"]:
        logger.warning("Quality gates failed: %s", result["quality_gate_violations"])
        if fail_on_gates:
            return 1
    print(json.dumps(result, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedRisk evaluation pipeline.")
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train if model artifacts are missing.",
    )
    parser.add_argument(
        "--fail-on-gates",
        action="store_true",
        help="Exit non-zero when quality-gate checks fail.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run_evaluation(args.train_if_missing, args.fail_on_gates))
