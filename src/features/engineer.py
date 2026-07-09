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


def engineer_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ratio features showing clinical intensity per hospital day.

    Must be called after :func:`engineer_medication_features` because it
    depends on ``num_med_changes`` and ``num_meds_active``.

    Parameters
    ----------
    df:
        Dataframe with ``num_lab_procedures``, ``num_procedures``,
        ``num_medications``, ``number_diagnoses``, ``time_in_hospital``,
        ``num_med_changes``, and ``num_meds_active``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    tih = df["time_in_hospital"].clip(lower=1) if "time_in_hospital" in df.columns else 1

    if "num_lab_procedures" in df.columns:
        df["lab_procedure_intensity"] = df["num_lab_procedures"] / tih
    else:
        df["lab_procedure_intensity"] = 0.0

    if "num_procedures" in df.columns:
        df["procedure_intensity"] = df["num_procedures"] / tih
    else:
        df["procedure_intensity"] = 0.0

    if "num_medications" in df.columns:
        df["medication_intensity"] = df["num_medications"] / tih
    else:
        df["medication_intensity"] = 0.0

    if "number_diagnoses" in df.columns:
        df["diagnosis_complexity"] = df["number_diagnoses"] / tih
    else:
        df["diagnosis_complexity"] = 0.0

    num_med_changes = df["num_med_changes"] if "num_med_changes" in df.columns else 0
    num_meds_active = df["num_meds_active"] if "num_meds_active" in df.columns else 0
    df["med_change_ratio"] = num_med_changes / (num_meds_active + 1)

    logger.info("Engineered intensity ratio features")
    return df


def engineer_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary clinical indicator flags from categorical columns.

    Parameters
    ----------
    df:
        Dataframe potentially containing ``A1Cresult``, ``diabetesMed``,
        and ``change``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "A1Cresult" in df.columns:
        df["A1Cresult_abnormal"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    else:
        df["A1Cresult_abnormal"] = 0

    if "diabetesMed" in df.columns:
        df["diabetesMed_flag"] = (df["diabetesMed"] == "Yes").astype(int)
    else:
        df["diabetesMed_flag"] = 0

    if "change" in df.columns:
        df["change_flag"] = (df["change"] == "Ch").astype(int)
    else:
        df["change_flag"] = 0

    df = df.drop(columns=["A1Cresult", "diabetesMed", "change"], errors="ignore")
    logger.info("Engineered clinical flag features")
    return df


def engineer_disposition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group high-cardinality discharge and admission source codes.

    Parameters
    ----------
    df:
        Dataframe potentially containing ``discharge_disposition_id`` and
        ``admission_source_id``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    discharge_map: dict[int, str] = {
        1: "Home",
        **{k: "Transfer" for k in (2, 3, 4, 5)},
        **{k: "Home_Health" for k in (6, 8)},
        **{k: "AMA_Left" for k in (7, 9, 10, 12)},
        **{k: "SNF_Rehab" for k in (11, 13, 14, 15, 22, 23, 24)},
    }

    admission_source_map: dict[int, str] = {
        **{k: "Referral" for k in (1, 2, 3)},
        **{k: "Transfer" for k in (4, 5, 6, 10, 22)},
        7: "Emergency",
        8: "Court_Law",
    }

    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_grouped"] = (
            df["discharge_disposition_id"].map(discharge_map).fillna("Other")
        )
    else:
        df["discharge_disposition_grouped"] = "Other"

    if "admission_source_id" in df.columns:
        df["admission_source_grouped"] = (
            df["admission_source_id"].map(admission_source_map).fillna("Other")
        )
    else:
        df["admission_source_grouped"] = "Other"

    df = df.drop(columns=["discharge_disposition_id", "admission_source_id"], errors="ignore")
    logger.info("Engineered disposition group features")
    return df


def engineer_visit_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve individual visit type counts before they are aggregated.

    Must be called **before** :func:`engineer_utilization_features`.

    Parameters
    ----------
    df:
        Dataframe with ``number_emergency``, ``number_inpatient``,
        ``number_outpatient``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "number_emergency" in df.columns:
        df["num_emergency_visits"] = df["number_emergency"]
    else:
        df["num_emergency_visits"] = 0

    if "number_inpatient" in df.columns:
        df["num_inpatient_visits"] = df["number_inpatient"]
    else:
        df["num_inpatient_visits"] = 0

    if "number_outpatient" in df.columns:
        df["num_outpatient_visits"] = df["number_outpatient"]
    else:
        df["num_outpatient_visits"] = 0

    logger.info("Engineered visit decomposition features")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature-engineering pipeline.

    Applies, in order: diagnosis mapping, medication aggregation,
    visit decomposition, utilization features, age ordinal encoding,
    intensity ratios, clinical flags, and disposition grouping.

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
    df = engineer_visit_decomposition(df)
    df = engineer_utilization_features(df)
    df = engineer_age_ordinal(df)
    df = engineer_intensity_features(df)
    df = engineer_clinical_flags(df)
    df = engineer_disposition_features(df)
    logger.info("Feature engineering complete — final shape: %s", df.shape)
    return df
