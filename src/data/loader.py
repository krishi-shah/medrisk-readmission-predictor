"""Data loading and validation utilities for the UCI Diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import get_data_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the UCI Diabetes 130-Hospitals CSV.

    Parameters
    ----------
    path:
        Explicit CSV path.  Falls back to ``data.raw_path`` in config.yaml.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with ``"?"`` values replaced by ``NaN``.
    """
    if path is None:
        path = get_data_config()["raw_path"]
    df = pd.read_csv(path, na_values=["?"])
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Assert that expected columns are present.

    Parameters
    ----------
    df:
        Raw dataframe to validate.

    Returns
    -------
    pd.DataFrame
        The same dataframe (pass-through), after validation.

    Raises
    ------
    AssertionError
        If any required column is missing.
    """
    required = {
        "encounter_id",
        "patient_nbr",
        "race",
        "gender",
        "age",
        "admission_type_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "diag_1",
        "diag_2",
        "diag_3",
        "insulin",
        "readmitted",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    logger.info("Validation passed — shape: %s", df.shape)
    return df


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first encounter per patient to prevent data leakage.

    Parameters
    ----------
    df:
        Dataframe that may contain multiple encounters per patient.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    before = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(
        subset=["patient_nbr"], keep="first"
    )
    logger.info("Deduplicated patients: %d -> %d rows", before, len(df))
    return df.reset_index(drop=True)
