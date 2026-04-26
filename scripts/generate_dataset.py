"""Generate a realistic 2000-row synthetic dataset matching the UCI Diabetes
130-Hospitals schema. Designed to reproduce the statistical properties of the
real dataset: ~11% <30-day readmission rate, correlated clinical risk factors,
realistic ICD-9 distributions, and proper class imbalance.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N = 2000

# ------------------------------------------------------------------
# Reference distributions from the real UCI dataset
# ------------------------------------------------------------------

RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
_r = [0.742, 0.188, 0.027, 0.009, 0.034]
RACE_PROBS = [v / sum(_r) for v in _r]

GENDERS = ["Male", "Female"]
GENDER_PROBS = [0.454, 0.546]

AGE_BRACKETS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
_a = [0.002, 0.005, 0.015, 0.040, 0.115, 0.175, 0.245, 0.255, 0.135, 0.013]
AGE_PROBS = [v / sum(_a) for v in _a]

ADMISSION_TYPES = [1, 2, 3, 4, 5, 6, 7, 8]
_at = [0.45, 0.10, 0.08, 0.05, 0.03, 0.15, 0.12, 0.02]
ADMISSION_TYPE_PROBS = [v / sum(_at) for v in _at]

DISCHARGE_DISPOSITIONS = [1, 2, 3, 4, 6, 7, 11, 13, 14, 18, 19, 20, 22, 23, 25, 28]
_dd = [0.54, 0.06, 0.04, 0.03, 0.04, 0.04, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
DISCHARGE_DISP_PROBS = [v / sum(_dd) for v in _dd]

ADMISSION_SOURCES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 20, 22, 25]
_as = [0.30, 0.08, 0.04, 0.02, 0.01, 0.01, 0.26, 0.05, 0.01, 0.01, 0.08, 0.05, 0.03, 0.03, 0.02]
ADMISSION_SOURCE_PROBS = [v / sum(_as) for v in _as]

MEDICAL_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
    "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
    "Psychiatry", "Surgery-Cardiovascular", "PhysicalMedicineandRehabilitation",
    "Podiatry", "Urology", None,  # None represents missing (?)
]
_sp = [0.135, 0.087, 0.073, 0.062, 0.054, 0.040, 0.035, 0.025, 0.020, 0.020,
       0.018, 0.015, 0.012, 0.010, 0.010, 0.384]
SPECIALTY_PROBS = [v / sum(_sp) for v in _sp]

GLU_SERUM_VALUES = ["None", "Norm", ">200", ">300"]
_gs = [0.947, 0.020, 0.019, 0.014]
GLU_SERUM_PROBS = [v / sum(_gs) for v in _gs]

A1C_VALUES = ["None", "Norm", ">7", ">8"]
_ac = [0.832, 0.059, 0.049, 0.060]
A1C_PROBS = [v / sum(_ac) for v in _ac]

MED_STATES = ["No", "Steady", "Up", "Down"]

CHANGE_VALUES = ["Ch", "No"]
CHANGE_PROBS = [0.539, 0.461]

DIABETES_MED_VALUES = ["Yes", "No"]
DIABETES_MED_PROBS = [0.769, 0.231]

# ICD-9 code pools by clinical category
ICD9_POOLS = {
    "Diabetes":       ["250", "250.01", "250.02", "250.1", "250.13", "250.2",
                       "250.5", "250.6", "250.7", "250.8", "250.9", "251", "252"],
    "Circulatory":    ["401", "402", "410", "411", "414", "427", "428", "429",
                       "430", "431", "433", "434", "436", "440", "443", "444",
                       "453", "458", "785"],
    "Respiratory":    ["466", "480", "486", "491", "492", "496", "507", "511",
                       "514", "518", "519", "786"],
    "Digestive":      ["520", "531", "535", "540", "553", "560", "562", "564",
                       "571", "574", "575", "787"],
    "Genitourinary":  ["580", "584", "585", "590", "592", "595", "599", "600",
                       "614", "626", "788"],
    "Musculoskeletal":["710", "714", "715", "716", "717", "720", "722", "724",
                       "726", "728", "730"],
    "Neoplasms":      ["140", "150", "153", "162", "174", "185", "188", "199",
                       "200", "202", "210", "220", "230"],
    "Injury":         ["800", "812", "820", "823", "825", "830", "840", "850",
                       "860", "880", "900", "940", "996", "998", "999"],
    "External":       ["V10", "V12", "V15", "V45", "V53", "V58", "V65",
                       "E819", "E826", "E878", "E933"],
    "Other":          ["276", "277", "285", "287", "288", "290", "296", "300",
                       "305", "311", "346", "349", "356", "362", "366", "375",
                       "380", "389"],
}

# Primary diagnosis weights (diabetes + circulatory dominate)
DIAG1_CATEGORY_PROBS = {
    "Diabetes": 0.23, "Circulatory": 0.26, "Respiratory": 0.07,
    "Digestive": 0.05, "Genitourinary": 0.04, "Musculoskeletal": 0.04,
    "Neoplasms": 0.03, "Injury": 0.06, "External": 0.04, "Other": 0.18,
}

# Medication columns (23 total)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Per-medication probability of being active (Steady/Up/Down) when prescribed
MED_ACTIVE_PROBS = {
    "metformin": 0.28, "repaglinide": 0.04, "nateglinide": 0.02,
    "chlorpropamide": 0.01, "glimepiride": 0.07, "acetohexamide": 0.00,
    "glipizide": 0.08, "glyburide": 0.08, "tolbutamide": 0.00,
    "pioglitazone": 0.06, "rosiglitazone": 0.03, "acarbose": 0.01,
    "miglitol": 0.00, "troglitazone": 0.00, "tolazamide": 0.00,
    "examide": 0.00, "citoglipton": 0.00, "insulin": 0.45,
    "glyburide-metformin": 0.02, "glipizide-metformin": 0.01,
    "glimepiride-pioglitazone": 0.00, "metformin-rosiglitazone": 0.00,
    "metformin-pioglitazone": 0.00,
}


def _sample_icd9(category: str) -> str:
    codes = ICD9_POOLS.get(category, ICD9_POOLS["Other"])
    return random.choice(codes)


def _medication_state(prob_active: float, high_risk: bool) -> str:
    """Sample a medication state; high-risk patients have more changes."""
    if rng.random() > prob_active:
        return "No"
    if high_risk:
        return rng.choice(["Steady", "Up", "Down"], p=[0.40, 0.40, 0.20])
    return rng.choice(["Steady", "Up", "Down"], p=[0.60, 0.25, 0.15])


def generate_row(encounter_id: int, patient_nbr: int) -> dict:
    # ---------- demographics ----------
    race = rng.choice(RACES, p=RACE_PROBS)
    gender = rng.choice(GENDERS, p=GENDER_PROBS)
    age_idx = rng.choice(len(AGE_BRACKETS), p=AGE_PROBS)
    age = AGE_BRACKETS[age_idx]

    # ---------- admission ----------
    admission_type_id = int(rng.choice(ADMISSION_TYPES, p=ADMISSION_TYPE_PROBS))
    discharge_disp_id = int(rng.choice(DISCHARGE_DISPOSITIONS, p=DISCHARGE_DISP_PROBS))
    admission_source_id = int(rng.choice(ADMISSION_SOURCES, p=ADMISSION_SOURCE_PROBS))

    # ---------- stay features ----------
    time_in_hospital = int(rng.integers(1, 14))
    num_lab_procedures = int(rng.integers(1, 132))
    num_procedures = int(rng.integers(0, 7))
    num_medications = int(rng.integers(1, 81))
    number_diagnoses = int(rng.integers(1, 16))

    # ---------- prior utilization (key readmission driver) ----------
    # Use a zero-inflated distribution to match real data
    _op = [0.61, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
    _op = [v / sum(_op) for v in _op]
    number_outpatient = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 40], p=_op))

    _em = [0.607, 0.155, 0.083, 0.047, 0.031, 0.019, 0.014, 0.010, 0.007, 0.012, 0.008, 0.007]
    _em = [v / sum(_em) for v in _em]
    number_emergency = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15], p=_em))

    _ip = [0.523, 0.196, 0.115, 0.066, 0.040, 0.021, 0.013, 0.008, 0.005, 0.004, 0.004, 0.003, 0.002]
    _ip = [v / sum(_ip) for v in _ip]
    number_inpatient = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], p=_ip))

    total_prior = number_inpatient + number_outpatient + number_emergency
    high_risk = (number_inpatient >= 2) or (total_prior >= 5) or (time_in_hospital >= 8)

    # ---------- diagnosis codes ----------
    _diag_cats = list(DIAG1_CATEGORY_PROBS.keys())
    _diag_p = list(DIAG1_CATEGORY_PROBS.values())
    _diag_p = [v / sum(_diag_p) for v in _diag_p]

    cat1_name = rng.choice(_diag_cats, p=_diag_p)
    diag_1 = _sample_icd9(cat1_name)

    # diag_2 and diag_3 often related or blank
    cat2_name = rng.choice(_diag_cats, p=_diag_p)
    diag_2 = _sample_icd9(cat2_name) if rng.random() < 0.90 else float("nan")

    cat3_name = rng.choice(_diag_cats, p=_diag_p)
    diag_3 = _sample_icd9(cat3_name) if rng.random() < 0.80 else float("nan")

    # ---------- lab results ----------
    max_glu_serum = rng.choice(GLU_SERUM_VALUES, p=GLU_SERUM_PROBS)
    A1Cresult = rng.choice(A1C_VALUES, p=A1C_PROBS)
    a1c_elevated = A1Cresult in (">7", ">8")

    # ---------- medications ----------
    meds = {}
    for med in MED_COLS:
        meds[med] = _medication_state(MED_ACTIVE_PROBS[med], high_risk)

    num_active = sum(1 for v in meds.values() if v != "No")
    change = "Ch" if any(v in ("Up", "Down") for v in meds.values()) else "No"
    diabetes_med = "Yes" if num_active > 0 else "No"

    # ---------- readmission label (clinical logic) ----------
    # Base readmission probability ~11% matching real dataset
    logit = (
        -4.5
        + 0.70 * number_inpatient
        + 0.35 * number_emergency
        + 0.10 * time_in_hospital
        + 0.30 * a1c_elevated
        + 0.20 * (cat1_name == "Circulatory")
        + 0.15 * (cat1_name == "Respiratory")
        + 0.10 * (num_medications > 15)
        + 0.20 * (age_idx >= 7)           # age >= 70
        + 0.30 * (number_inpatient >= 3)
        - 0.15 * (discharge_disp_id == 1) # routine discharge slightly protective
        + rng.normal(0, 0.3)              # residual noise
    )
    prob_readmit = 1.0 / (1.0 + np.exp(-logit))
    readmitted_flag = rng.random() < prob_readmit

    if readmitted_flag:
        readmitted = "<30"
    else:
        # ~15% of non-<30 are actually >30
        readmitted = ">30" if rng.random() < 0.15 else "NO"

    # ---------- specialty (missing ~38% of the time) ----------
    specialty = rng.choice(MEDICAL_SPECIALTIES, p=SPECIALTY_PROBS)
    specialty_out = "?" if specialty is None else specialty

    row = {
        "encounter_id": encounter_id,
        "patient_nbr": patient_nbr,
        "race": race,
        "gender": gender,
        "age": age,
        "weight": "?",
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disp_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "payer_code": "?",
        "medical_specialty": specialty_out,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2 if not (isinstance(diag_2, float) and np.isnan(diag_2)) else "?",
        "diag_3": diag_3 if not (isinstance(diag_3, float) and np.isnan(diag_3)) else "?",
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        **meds,
        "change": change,
        "diabetesMed": diabetes_med,
        "readmitted": readmitted,
    }
    return row


def main() -> None:
    rows = []
    for i in range(N):
        rows.append(generate_row(encounter_id=i + 1, patient_nbr=i + 1001))

    df = pd.DataFrame(rows)

    readmit_rate = (df["readmitted"] == "<30").mean()
    print(f"Generated {N} rows | <30-day readmission rate: {readmit_rate:.3f}")
    print(f"Class distribution:\n{df['readmitted'].value_counts()}")

    root = Path(__file__).resolve().parents[1]

    sample_path = root / "data" / "sample" / "sample_data.csv"
    raw_path = root / "data" / "raw" / "diabetic_data.csv"

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample: first 200 rows for quick tests / dashboard demos
    df.head(200).to_csv(sample_path, index=False)
    print(f"Saved sample (200 rows) -> {sample_path}")

    # Full synthetic dataset used for training
    df.to_csv(raw_path, index=False)
    print(f"Saved full dataset ({N} rows) -> {raw_path}")


if __name__ == "__main__":
    main()
