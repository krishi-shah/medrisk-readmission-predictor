"""Data cleaning, preprocessing pipeline, and train/val/test splitting."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier / low-signal columns and create the binary target.

    Steps
    -----
    1. Drop ``encounter_id``, ``patient_nbr``, ``weight``, ``payer_code``.
    2. Create ``medical_specialty_missing`` indicator, then drop
       ``medical_specialty``.
    3. Map ``readmitted`` to binary (1 if ``"<30"``, else 0).

    Parameters
    ----------
    df:
        Validated, deduplicated dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with binary ``readmitted`` target.
    """
    df = df.copy()

    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code"],
                 errors="ignore")

    df["medical_specialty_missing"] = df["medical_specialty"].isna().astype(int)
    df = df.drop(columns=["medical_specialty"])

    df["readmitted"] = (df["readmitted"] == "<30").astype(int)

    logger.info(
        "Cleaned data — shape: %s, target rate: %.3f",
        df.shape,
        df["readmitted"].mean(),
    )
    return df


def build_preprocessing_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Build a scikit-learn ``ColumnTransformer`` for numeric + categorical features.

    Parameters
    ----------
    numeric_features:
        Column names to process with median imputation + standard scaling.
    categorical_features:
        Column names to process with most-frequent imputation + one-hot encoding.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready for ``.fit()`` / ``.fit_transform()``.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
    )
    logger.info(
        "Built preprocessing pipeline — %d numeric, %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str = "readmitted",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split (default 70/15/15).

    Parameters
    ----------
    df:
        Full dataframe including the target column.
    target_col:
        Name of the binary target column.
    test_size:
        Fraction held out for the test set.
    val_size:
        Fraction of the *original* data held out for validation.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` dataframes.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[target_col],
        random_state=random_state,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test
