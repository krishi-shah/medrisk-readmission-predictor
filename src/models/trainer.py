"""Model training pipeline for 30-day diabetic readmission prediction.

Trains logistic regression, random forest, XGBoost, and LightGBM models
with Optuna-based hyperparameter tuning for the boosted models. All
experiments are tracked via MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_base_models(random_state: int) -> dict[str, Any]:
    """Return a mapping of model name to an untrained estimator.

    Parameters
    ----------
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        ``{model_name: estimator}`` for the four candidate algorithms.
    """
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            verbose=-1,
            random_state=random_state,
        ),
    }


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    random_state: int,
) -> dict[str, Any]:
    """Optimize XGBoost hyperparameters with Optuna (PR-AUC objective).

    Parameters
    ----------
    X_train, y_train:
        Training features and labels.
    X_val, y_val:
        Validation features and labels.
    n_trials:
        Number of Optuna trials.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Best hyperparameter dictionary.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
            "eval_metric": "logloss",
            "random_state": random_state,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    logger.info("XGBoost best PR-AUC: %.4f", study.best_value)
    return study.best_params


def optimize_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    random_state: int,
) -> dict[str, Any]:
    """Optimize LightGBM hyperparameters with Optuna (PR-AUC objective).

    Parameters
    ----------
    X_train, y_train:
        Training features and labels.
    X_val, y_val:
        Validation features and labels.
    n_trials:
        Number of Optuna trials.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Best hyperparameter dictionary.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
            "verbose": -1,
            "random_state": random_state,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    logger.info("LightGBM best PR-AUC: %.4f", study.best_value)
    return study.best_params


def train_and_log_model(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Train a model and log metrics / artifacts to MLflow.

    Parameters
    ----------
    model:
        Scikit-learn-compatible estimator.
    model_name:
        Human-readable model identifier (used as MLflow run name).
    X_train, y_train:
        Training features and labels.
    X_val, y_val:
        Validation features and labels.
    params:
        Optional parameter dict to log alongside auto-detected params.

    Returns
    -------
    dict[str, float]
        Validation metrics keyed by metric name.
    """
    with mlflow.start_run(run_name=model_name, nested=True):
        logger.info("Training %s …", model_name)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_val, y_prob),
            "pr_auc": average_precision_score(y_val, y_prob),
            "f1": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
        }

        if params:
            mlflow.log_params(params)
        mlflow.log_params({"model_name": model_name})
        mlflow.log_metrics(metrics)

        artifact_path = _PROJECT_ROOT / "models" / f"{model_name}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_path)
        mlflow.log_artifact(str(artifact_path))

        logger.info(
            "%s — ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f",
            model_name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
        )

    return metrics


def run_training_pipeline() -> None:
    """Execute the full training pipeline.

    1. Load and validate raw data.
    2. Engineer features, clean, and split.
    3. Build and fit preprocessing pipeline.
    4. Train all candidate models (Optuna for boosted models).
    5. Persist best model and preprocessor.
    """
    from src.data.loader import load_raw_data, validate_data, deduplicate_patients
    from src.data.preprocessor import clean_data, build_preprocessing_pipeline, split_data
    from src.features.engineer import run_feature_engineering

    cfg = get_config()["training"]
    random_state: int = cfg["random_state"]
    n_trials: int = cfg["n_optuna_trials"]

    logger.info("Loading raw data …")
    df = load_raw_data()
    validate_data(df)
    df = deduplicate_patients(df)

    logger.info("Feature engineering …")
    df = run_feature_engineering(df)
    df = clean_data(df)

    logger.info("Splitting data …")
    train_df, val_df, test_df = split_data(
        df,
        target_col="readmitted",
        test_size=cfg["test_size"],
        val_size=cfg["val_size"],
        random_state=random_state,
    )

    y_train = train_df["readmitted"].values
    y_val = val_df["readmitted"].values
    y_test = test_df["readmitted"].values

    feature_cfg = get_config()["features"]
    numeric_features = feature_cfg["numeric"]
    categorical_features = feature_cfg["categorical"]

    logger.info("Building preprocessing pipeline …")
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(train_df[numeric_features + categorical_features])
    X_val = preprocessor.transform(val_df[numeric_features + categorical_features])
    X_test = preprocessor.transform(test_df[numeric_features + categorical_features])

    models = get_base_models(random_state)
    results: dict[str, dict[str, float]] = {}

    mlflow_cfg = get_config().get("mlflow", {})
    experiment_name = mlflow_cfg.get("experiment_name", "medrisk-readmission")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="training_pipeline"):
        for name, model in models.items():
            if name not in cfg["models"]:
                logger.info("Skipping %s (not in config)", name)
                continue

            if name == "xgboost":
                best_params = optimize_xgboost(
                    X_train, y_train, X_val, y_val, n_trials, random_state,
                )
                model = XGBClassifier(
                    **best_params,
                    eval_metric="logloss",
                    random_state=random_state,
                )
                results[name] = train_and_log_model(
                    model, name, X_train, y_train, X_val, y_val, params=best_params,
                )
            elif name == "lightgbm":
                best_params = optimize_lightgbm(
                    X_train, y_train, X_val, y_val, n_trials, random_state,
                )
                model = LGBMClassifier(
                    **best_params,
                    verbose=-1,
                    random_state=random_state,
                )
                results[name] = train_and_log_model(
                    model, name, X_train, y_train, X_val, y_val, params=best_params,
                )
            else:
                results[name] = train_and_log_model(
                    model, name, X_train, y_train, X_val, y_val,
                )

    if not results:
        raise RuntimeError(
            "No models were trained. Check that 'training.models' in config.yaml "
            "contains at least one valid model name."
        )

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    logger.info("Best model by PR-AUC: %s (%.4f)", best_name, results[best_name]["pr_auc"])

    output_dir = _PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model = joblib.load(output_dir / f"{best_name}.joblib")
    joblib.dump(best_model, output_dir / "best_model.joblib")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    from src.models.evaluator import find_optimal_threshold, evaluate_model
    import json

    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, y_val_prob, min_recall=cfg["threshold_min_recall"])
    with open(output_dir / "threshold.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": threshold, "min_recall": cfg["threshold_min_recall"]}, fh)

    paths_cfg = get_config().get("paths", {})
    reports_dir = _PROJECT_ROOT / paths_cfg.get("reports_dir", "reports/figures")
    test_metrics = evaluate_model(best_model, X_test, y_test, threshold, save_dir=reports_dir)
    logger.info(
        "Test-set evaluation — ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f",
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["f1"],
    )

    logger.info("Saved best_model.joblib, preprocessor.joblib, and threshold.json to %s", output_dir)


if __name__ == "__main__":
    run_training_pipeline()
