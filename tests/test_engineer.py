import pytest
import pandas as pd
import numpy as np

from src.features.engineer import (
    map_icd9_to_category,
    engineer_diagnosis_features,
    engineer_medication_features,
    engineer_utilization_features,
    engineer_age_ordinal,
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
        })
        return pd.DataFrame(data)

    def test_full_pipeline_runs(self):
        df = self._make_full_df()
        result = run_feature_engineering(df)
        expected_cols = [
            "diag1_category", "diag2_category", "diag3_category",
            "num_med_changes", "num_meds_active", "insulin_changed",
            "total_prior_visits", "high_utilizer", "age_ordinal",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_raw_columns_removed(self):
        df = self._make_full_df()
        result = run_feature_engineering(df)
        assert "diag_1" not in result.columns
        assert "age" not in result.columns
        assert "metformin" not in result.columns

    def test_diabetes_icd9_range(self):
        assert map_icd9_to_category("249") == "Diabetes"
        assert map_icd9_to_category("249.5") == "Diabetes"
        assert map_icd9_to_category("250") == "Diabetes"
        assert map_icd9_to_category("259") == "Diabetes"
        assert map_icd9_to_category("260") != "Diabetes"
