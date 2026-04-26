"""Feature engineering for the UCI Diabetes 130-Hospitals dataset."""

from __future__ import annotations

import math

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MEDICATION_COLUMNS: list[str] = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

_AGE_BRACKET_MAP: dict[str, int] = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
    "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
    "[80-90)": 8, "[90-100)": 9,
}


# ------------------------------------------------------------------
# ICD-9 mapping
# ------------------------------------------------------------------

def map_icd9_to_category(code: str | float) -> str:
    """Map a single ICD-9 diagnosis code to a clinical category.

    Parameters
    ----------
    code:
        Raw ICD-9 code string (e.g. ``"250.13"``, ``"V45"``, ``"E819"``).
        May be ``NaN``.

    Returns
    -------
    str
        One of *Circulatory*, *Respiratory*, *Digestive*, *Diabetes*,
        *Injury*, *Musculoskeletal*, *Genitourinary*, *Neoplasms*,
        *External*, or *Other*.
    """
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "Other"

    code = str(code).strip()
    if not code:
        return "Other"

    first = code[0].upper()

    if first in ("V", "E"):
        return "External"

    try:
        numeric = float(code)
    except ValueError:
        return "Other"

    if numeric == 785:
        return "Circulatory"
    if numeric == 786:
        return "Respiratory"
    if numeric == 787:
        return "Digestive"
    if numeric == 788:
        return "Genitourinary"

    integer_part = int(numeric)

    if 249 <= integer_part <= 259:
        return "Diabetes"
    if 390 <= integer_part <= 459:
        return "Circulatory"
    if 460 <= integer_part <= 519:
        return "Respiratory"
    if 520 <= integer_part <= 579:
        return "Digestive"
    if 580 <= integer_part <= 629:
        return "Genitourinary"
    if 710 <= integer_part <= 739:
        return "Musculoskeletal"
    if 800 <= integer_part <= 999:
        return "Injury"
    if 140 <= integer_part <= 239:
        return "Neoplasms"

    return "Other"


# ------------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------------

def engineer_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``diag_1/2/3`` ICD-9 codes to clinical categories and drop originals.

    Parameters
    ----------
    df:
        Dataframe containing ``diag_1``, ``diag_2``, ``diag_3``.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``diag1_category``, ``diag2_category``,
        ``diag3_category`` and without the original diagnosis columns.
    """
    df = df.copy()
    for i in range(1, 4):
        src_col = f"diag_{i}"
        dst_col = f"diag{i}_category"
        if src_col in df.columns:
            df[dst_col] = df[src_col].apply(map_icd9_to_category)
        else:
            df[dst_col] = "Other"
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"], errors="ignore")
    logger.info("Engineered diagnosis category features")
    return df


def engineer_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 23 individual medication columns into summary features.

    Creates ``num_med_changes``, ``num_meds_active``, and
    ``insulin_changed``, then drops the original medication columns.

    Parameters
    ----------
    df:
        Dataframe with the 23 medication columns present.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    present_cols = [c for c in MEDICATION_COLUMNS if c in df.columns]

    change_values = {"Up", "Down"}
    active_values = {"Up", "Down", "Steady"}

    if present_cols:
        df["num_med_changes"] = df[present_cols].apply(
            lambda row: sum(v in change_values for v in row), axis=1,
        )
        df["num_meds_active"] = df[present_cols].apply(
            lambda row: sum(v in active_values for v in row), axis=1,
        )
    else:
        df["num_med_changes"] = 0
        df["num_meds_active"] = 0
    df["insulin_changed"] = df["insulin"].isin(change_values).astype(int) if "insulin" in df.columns else 0

    df = df.drop(columns=present_cols, errors="ignore")
    logger.info("Engineered medication summary features")
    return df


def engineer_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``total_prior_visits`` and ``high_utilizer`` flag.

    Parameters
    ----------
    df:
        Dataframe with ``number_inpatient``, ``number_outpatient``,
        ``number_emergency``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in ("number_inpatient", "number_outpatient", "number_emergency"):
        if col not in df.columns:
            df[col] = 0
    df["total_prior_visits"] = (
        df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    )
    threshold = get_config()["features"].get("high_utilizer_threshold", 5)
    df["high_utilizer"] = (df["total_prior_visits"] >= threshold).astype(int)
    logger.info("Engineered utilization features")
    return df


def engineer_age_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age bracket strings to ordinal integers.

    Parameters
    ----------
    df:
        Dataframe with an ``age`` column containing bracket strings
        like ``"[70-80)"``.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``age_ordinal`` and without the original ``age``
        column.
    """
    df = df.copy()
    if "age" in df.columns:
        df["age_ordinal"] = df["age"].map(_AGE_BRACKET_MAP).fillna(-1).astype(int)
        df = df.drop(columns=["age"], errors="ignore")
    else:
        df["age_ordinal"] = -1
    logger.info("Engineered age ordinal feature")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature-engineering pipeline.

    Applies, in order: diagnosis mapping, medication aggregation,
    utilization features, and age ordinal encoding.

    Parameters
    ----------
    df:
        Cleaned dataframe (output of :func:`src.data.preprocessor.clean_data`).

    Returns
    -------
    pd.DataFrame
        Fully engineered dataframe.
    """
    df = engineer_diagnosis_features(df)
    df = engineer_medication_features(df)
    df = engineer_utilization_features(df)
    df = engineer_age_ordinal(df)
    logger.info("Feature engineering complete — final shape: %s", df.shape)
    return df
