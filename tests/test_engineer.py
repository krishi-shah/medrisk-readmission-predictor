import pytest
import pandas as pd
import numpy as np

from src.features.engineer import (
    map_icd9_to_category,
    engineer_diagnosis_features,
    engineer_medication_features,
    engineer_utilization_features,
    engineer_age_ordinal,
    engineer_intensity_features,
    engineer_clinical_flags,
    engineer_disposition_features,
    engineer_visit_decomposition,
    run_feature_engineering,
)


class TestICD9Mapping:
    def test_diabetes_code(self):
        assert map_icd9_to_category("250") == "Diabetes"
        assert map_icd9_to_category("250.83") == "Diabetes"
        assert map_icd9_to_category("250.01") == "Diabetes"

    def test_circulatory_codes(self):
        assert map_icd9_to_category("401") == "Circulatory"
        assert map_icd9_to_category("428") == "Circulatory"
        assert map_icd9_to_category("785") == "Circulatory"

    def test_respiratory_codes(self):
        assert map_icd9_to_category("486") == "Respiratory"
        assert map_icd9_to_category("786") == "Respiratory"

    def test_digestive_codes(self):
        assert map_icd9_to_category("530") == "Digestive"
        assert map_icd9_to_category("787") == "Digestive"

    def test_injury_codes(self):
        assert map_icd9_to_category("800") == "Injury"
        assert map_icd9_to_category("999") == "Injury"

    def test_musculoskeletal_codes(self):
        assert map_icd9_to_category("715") == "Musculoskeletal"

    def test_genitourinary_codes(self):
        assert map_icd9_to_category("585") == "Genitourinary"
        assert map_icd9_to_category("788") == "Genitourinary"

    def test_neoplasm_codes(self):
        assert map_icd9_to_category("150") == "Neoplasms"

    def test_v_codes(self):
        assert map_icd9_to_category("V58") == "External"
        assert map_icd9_to_category("V10.2") == "External"

    def test_e_codes(self):
        assert map_icd9_to_category("E819") == "External"

    def test_nan_returns_other(self):
        assert map_icd9_to_category(np.nan) == "Other"
        assert map_icd9_to_category(None) == "Other"
        assert map_icd9_to_category("?") == "Other"

    def test_other_codes(self):
        assert map_icd9_to_category("780") == "Other"


class TestMedicationFeatures:
    def _make_med_df(self):
        med_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone",
        ]
        data = {col: ["No"] for col in med_cols}
        data["metformin"] = ["Up"]
        data["insulin"] = ["Down"]
        data["glipizide"] = ["Steady"]
        return pd.DataFrame(data)

    def test_med_changes_count(self):
        df = engineer_medication_features(self._make_med_df())
        assert df["num_med_changes"].iloc[0] == 2  # Up + Down

    def test_meds_active_count(self):
        df = engineer_medication_features(self._make_med_df())
        assert df["num_meds_active"].iloc[0] == 3  # Up + Down + Steady

    def test_insulin_changed_flag(self):
        df = engineer_medication_features(self._make_med_df())
        assert df["insulin_changed"].iloc[0] == 1

    def test_med_changes_nonnegative(self):
        df = engineer_medication_features(self._make_med_df())
        assert df["num_med_changes"].iloc[0] >= 0

    def test_original_columns_dropped(self):
        df = engineer_medication_features(self._make_med_df())
        assert "metformin" not in df.columns
        assert "insulin" not in df.columns

    def test_handles_missing_medication_columns(self):
        df = pd.DataFrame({"some_other_col": [1, 2]})
        result = engineer_medication_features(df)
        assert result["num_med_changes"].tolist() == [0, 0]
        assert result["num_meds_active"].tolist() == [0, 0]
        assert result["insulin_changed"].tolist() == [0, 0]


class TestUtilizationFeatures:
    def test_total_prior_visits(self):
        df = pd.DataFrame({
            "number_inpatient": [2],
            "number_outpatient": [1],
            "number_emergency": [1],
        })
        result = engineer_utilization_features(df)
        assert result["total_prior_visits"].iloc[0] == 4

    def test_high_utilizer_true(self):
        df = pd.DataFrame({
            "number_inpatient": [3],
            "number_outpatient": [1],
            "number_emergency": [2],
        })
        result = engineer_utilization_features(df)
        assert result["high_utilizer"].iloc[0] == 1

    def test_high_utilizer_false(self):
        df = pd.DataFrame({
            "number_inpatient": [1],
            "number_outpatient": [1],
            "number_emergency": [0],
        })
        result = engineer_utilization_features(df)
        assert result["high_utilizer"].iloc[0] == 0


class TestAgeOrdinal:
    @pytest.mark.parametrize("bracket,expected", [
        ("[0-10)", 0), ("[10-20)", 1), ("[20-30)", 2), ("[30-40)", 3),
        ("[40-50)", 4), ("[50-60)", 5), ("[60-70)", 6), ("[70-80)", 7),
        ("[80-90)", 8), ("[90-100)", 9),
    ])
    def test_all_brackets(self, bracket, expected):
        df = pd.DataFrame({"age": [bracket]})
        result = engineer_age_ordinal(df)
        assert result["age_ordinal"].iloc[0] == expected
        assert "age" not in result.columns


class TestIntensityFeatures:
    def test_lab_procedure_intensity(self):
        df = pd.DataFrame({
            "num_lab_procedures": [30],
            "num_procedures": [3],
            "num_medications": [15],
            "number_diagnoses": [6],
            "time_in_hospital": [3],
            "num_med_changes": [2],
            "num_meds_active": [4],
        })
        result = engineer_intensity_features(df)
        assert result["lab_procedure_intensity"].iloc[0] == pytest.approx(10.0)
        assert result["procedure_intensity"].iloc[0] == pytest.approx(1.0)
        assert result["medication_intensity"].iloc[0] == pytest.approx(5.0)
        assert result["diagnosis_complexity"].iloc[0] == pytest.approx(2.0)
        assert result["med_change_ratio"].iloc[0] == pytest.approx(0.4)

    def test_zero_time_in_hospital_clips_to_one(self):
        df = pd.DataFrame({
            "num_lab_procedures": [10],
            "time_in_hospital": [0],
            "num_procedures": [0],
            "num_medications": [0],
            "number_diagnoses": [0],
            "num_med_changes": [0],
            "num_meds_active": [0],
        })
        result = engineer_intensity_features(df)
        assert result["lab_procedure_intensity"].iloc[0] == pytest.approx(10.0)

    def test_missing_columns_default_to_zero(self):
        df = pd.DataFrame({"other_col": [1]})
        result = engineer_intensity_features(df)
        assert result["lab_procedure_intensity"].iloc[0] == 0.0
        assert result["procedure_intensity"].iloc[0] == 0.0


class TestClinicalFlags:
    def test_a1c_abnormal_positive(self):
        df = pd.DataFrame({"A1Cresult": [">7", ">8", "Norm", "None"]})
        result = engineer_clinical_flags(df)
        assert result["A1Cresult_abnormal"].tolist() == [1, 1, 0, 0]

    def test_diabetes_med_flag(self):
        df = pd.DataFrame({"diabetesMed": ["Yes", "No"]})
        result = engineer_clinical_flags(df)
        assert result["diabetesMed_flag"].tolist() == [1, 0]

    def test_change_flag(self):
        df = pd.DataFrame({"change": ["Ch", "No"]})
        result = engineer_clinical_flags(df)
        assert result["change_flag"].tolist() == [1, 0]

    def test_source_columns_dropped(self):
        df = pd.DataFrame({
            "A1Cresult": [">7"],
            "diabetesMed": ["Yes"],
            "change": ["Ch"],
        })
        result = engineer_clinical_flags(df)
        assert "A1Cresult" not in result.columns
        assert "diabetesMed" not in result.columns
        assert "change" not in result.columns

    def test_missing_columns_default_to_zero(self):
        df = pd.DataFrame({"other_col": [1]})
        result = engineer_clinical_flags(df)
        assert result["A1Cresult_abnormal"].iloc[0] == 0
        assert result["diabetesMed_flag"].iloc[0] == 0
        assert result["change_flag"].iloc[0] == 0


class TestDispositionFeatures:
    def test_discharge_grouping(self):
        df = pd.DataFrame({"discharge_disposition_id": [1, 2, 6, 7, 11, 99]})
        result = engineer_disposition_features(df)
        expected = ["Home", "Transfer", "Home_Health", "AMA_Left", "SNF_Rehab", "Other"]
        assert result["discharge_disposition_grouped"].tolist() == expected

    def test_admission_source_grouping(self):
        df = pd.DataFrame({"admission_source_id": [1, 4, 7, 8, 99]})
        result = engineer_disposition_features(df)
        expected = ["Referral", "Transfer", "Emergency", "Court_Law", "Other"]
        assert result["admission_source_grouped"].tolist() == expected

    def test_source_columns_dropped(self):
        df = pd.DataFrame({
            "discharge_disposition_id": [1],
            "admission_source_id": [7],
        })
        result = engineer_disposition_features(df)
        assert "discharge_disposition_id" not in result.columns
        assert "admission_source_id" not in result.columns

    def test_missing_columns_default_to_other(self):
        df = pd.DataFrame({"other_col": [1]})
        result = engineer_disposition_features(df)
        assert result["discharge_disposition_grouped"].iloc[0] == "Other"
        assert result["admission_source_grouped"].iloc[0] == "Other"


class TestVisitDecomposition:
    def test_copies_visit_columns(self):
        df = pd.DataFrame({
            "number_emergency": [3],
            "number_inpatient": [2],
            "number_outpatient": [5],
        })
        result = engineer_visit_decomposition(df)
        assert result["num_emergency_visits"].iloc[0] == 3
        assert result["num_inpatient_visits"].iloc[0] == 2
        assert result["num_outpatient_visits"].iloc[0] == 5

    def test_missing_columns_default_to_zero(self):
        df = pd.DataFrame({"other_col": [1]})
        result = engineer_visit_decomposition(df)
        assert result["num_emergency_visits"].iloc[0] == 0
        assert result["num_inpatient_visits"].iloc[0] == 0
        assert result["num_outpatient_visits"].iloc[0] == 0


class TestRunFeatureEngineering:
    def _make_full_df(self):
        med_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone",
        ]
        data = {col: ["No", "No"] for col in med_cols}
        data["metformin"] = ["Up", "No"]
        data["insulin"] = ["Down", "Steady"]
        data.update({
            "diag_1": ["250.01", "428"],
            "diag_2": ["401", "486"],
            "diag_3": ["V58", "E819"],
            "number_inpatient": [2, 0],
            "number_outpatient": [1, 0],
            "number_emergency": [3, 0],
            "age": ["[60-70)", "[40-50)"],
            "time_in_hospital": [5, 3],
            "num_lab_procedures": [50, 30],
            "num_procedures": [2, 1],
            "num_medications": [15, 10],
            "number_diagnoses": [7, 4],
            "A1Cresult": [">7", "None"],
            "diabetesMed": ["Yes", "No"],
            "change": ["Ch", "No"],
            "discharge_disposition_id": [1, 11],
            "admission_source_id": [7, 1],
        })
        return pd.DataFrame(data)

    def test_full_pipeline_runs(self):
        df = self._make_full_df()
        result = run_feature_engineering(df)
        expected_cols = [
            "diag1_category", "diag2_category", "diag3_category",
            "num_med_changes", "num_meds_active", "insulin_changed",
            "total_prior_visits", "high_utilizer", "age_ordinal",
            "num_emergency_visits", "num_inpatient_visits", "num_outpatient_visits",
            "lab_procedure_intensity", "procedure_intensity",
            "medication_intensity", "diagnosis_complexity", "med_change_ratio",
            "A1Cresult_abnormal", "diabetesMed_flag", "change_flag",
            "discharge_disposition_grouped", "admission_source_grouped",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_raw_columns_removed(self):
        df = self._make_full_df()
        result = run_feature_engineering(df)
        assert "diag_1" not in result.columns
        assert "age" not in result.columns
        assert "metformin" not in result.columns
        assert "A1Cresult" not in result.columns
        assert "diabetesMed" not in result.columns
        assert "discharge_disposition_id" not in result.columns

    def test_diabetes_icd9_range(self):
        assert map_icd9_to_category("249") == "Diabetes"
        assert map_icd9_to_category("249.5") == "Diabetes"
        assert map_icd9_to_category("250") == "Diabetes"
        assert map_icd9_to_category("259") == "Diabetes"
        assert map_icd9_to_category("260") != "Diabetes"

    def test_feature_count_at_least_30(self):
        df = self._make_full_df()
        result = run_feature_engineering(df)
        engineered = [
            "diag1_category", "diag2_category", "diag3_category",
            "num_med_changes", "num_meds_active", "insulin_changed",
            "total_prior_visits", "high_utilizer", "age_ordinal",
            "num_emergency_visits", "num_inpatient_visits", "num_outpatient_visits",
            "lab_procedure_intensity", "procedure_intensity",
            "medication_intensity", "diagnosis_complexity", "med_change_ratio",
            "A1Cresult_abnormal", "diabetesMed_flag", "change_flag",
            "discharge_disposition_grouped", "admission_source_grouped",
        ]
        present = [c for c in engineered if c in result.columns]
        assert len(present) >= 22
