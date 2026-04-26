import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
from pathlib import Path

from src.data.preprocessor import clean_data, build_preprocessing_pipeline, split_data


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4, 5],
        "patient_nbr": [101, 102, 103, 104, 105],
        "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Caucasian"],
        "gender": ["Female", "Male", "Female", "Male", "Female"],
        "age": ["[60-70)", "[40-50)", "[70-80)", "[50-60)", "[80-90)"],
        "weight": ["?", "?", "?", "?", "?"],
        "payer_code": ["MC", "?", "SP", "?", "MC"],
        "medical_specialty": ["InternalMedicine", np.nan, "Surgery", np.nan, "Cardiology"],
        "admission_type_id": [1, 1, 2, 1, 3],
        "discharge_disposition_id": [1, 1, 3, 1, 1],
        "admission_source_id": [7, 7, 2, 7, 1],
        "time_in_hospital": [5, 3, 7, 2, 4],
        "num_lab_procedures": [44, 35, 52, 28, 60],
        "num_procedures": [1, 0, 3, 0, 2],
        "num_medications": [18, 12, 22, 10, 16],
        "number_diagnoses": [9, 7, 8, 5, 9],
        "readmitted": ["<30", ">30", "<30", "NO", "<30"],
    })


class TestCleanData:
    def test_drops_id_columns(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "encounter_id" not in cleaned.columns
        assert "patient_nbr" not in cleaned.columns

    def test_drops_weight(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "weight" not in cleaned.columns

    def test_drops_payer_code(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "payer_code" not in cleaned.columns

    def test_creates_medical_specialty_missing(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "medical_specialty_missing" in cleaned.columns
        assert cleaned["medical_specialty_missing"].iloc[1] == 1
        assert cleaned["medical_specialty_missing"].iloc[0] == 0

    def test_drops_medical_specialty(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "medical_specialty" not in cleaned.columns

    def test_binary_target(self, raw_df):
        cleaned = clean_data(raw_df)
        assert set(cleaned["readmitted"].unique()).issubset({0, 1})
        assert cleaned["readmitted"].iloc[0] == 1  # <30 -> 1
        assert cleaned["readmitted"].iloc[3] == 0  # NO -> 0

    def test_missing_medical_specialty_column(self, raw_df):
        df = raw_df.drop(columns=["medical_specialty"])
        cleaned = clean_data(df)
        assert "medical_specialty_missing" in cleaned.columns
        assert cleaned["medical_specialty_missing"].eq(1).all()


class TestPreprocessingPipeline:
    def test_output_has_no_nans(self):
        numeric = ["num1", "num2"]
        categorical = ["cat1"]
        pipeline = build_preprocessing_pipeline(numeric, categorical)

        df = pd.DataFrame({
            "num1": [1.0, np.nan, 3.0, 4.0],
            "num2": [10.0, 20.0, np.nan, 40.0],
            "cat1": ["A", "B", None, "A"],
        })
        result = pipeline.fit_transform(df)
        assert not np.any(np.isnan(result))

    def test_output_shape(self):
        numeric = ["num1"]
        categorical = ["cat1"]
        pipeline = build_preprocessing_pipeline(numeric, categorical)

        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["A", "B", "A"],
        })
        result = pipeline.fit_transform(df)
        assert result.shape[0] == 3
        assert result.shape[1] >= 2  # 1 numeric + at least 2 one-hot

    def test_handles_unseen_categories(self):
        numeric = ["num1"]
        categorical = ["cat1"]
        pipeline = build_preprocessing_pipeline(numeric, categorical)

        train_df = pd.DataFrame({"num1": [1.0, 2.0], "cat1": ["A", "B"]})
        test_df = pd.DataFrame({"num1": [3.0], "cat1": ["C"]})

        pipeline.fit(train_df)
        result = pipeline.transform(test_df)
        assert result.shape[0] == 1

    def test_pipeline_serializable(self):
        numeric = ["num1"]
        categorical = ["cat1"]
        pipeline = build_preprocessing_pipeline(numeric, categorical)

        df = pd.DataFrame({"num1": [1.0, 2.0], "cat1": ["A", "B"]})
        pipeline.fit(df)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(pipeline, f.name)
            loaded = joblib.load(f.name)
            result = loaded.transform(df)
            assert result.shape[0] == 2


class TestSplitData:
    def test_split_sizes(self):
        df = pd.DataFrame({
            "feature": range(100),
            "target": [1] * 15 + [0] * 85,
        })
        train, val, test = split_data(df, "target", test_size=0.15, val_size=0.15, random_state=42)
        total = len(train) + len(val) + len(test)
        assert total == 100
        assert len(test) == pytest.approx(15, abs=3)
        assert len(val) == pytest.approx(15, abs=3)

    def test_stratification(self):
        df = pd.DataFrame({
            "feature": range(200),
            "target": [1] * 22 + [0] * 178,
        })
        train, val, test = split_data(df, "target", test_size=0.15, val_size=0.15, random_state=42)
        train_rate = train["target"].mean()
        test_rate = test["target"].mean()
        assert abs(train_rate - test_rate) < 0.05
