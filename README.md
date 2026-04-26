# MedRisk: Patient Readmission Risk Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![MLflow](https://img.shields.io/badge/MLflow-2.12-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://img.shields.io/github/actions/workflow/status/krishi-shah/medrisk-readmission-predictor/ci.yml?label=CI)

An end to end clinical machine learning system that predicts 30 day hospital readmission risk for diabetic patients. The pipeline covers data ingestion, clinical feature engineering, model training with experiment tracking, SHAP explainability, and a production ready FastAPI inference service with a clinician facing Streamlit dashboard.

Hospital readmissions within 30 days are a federally tracked quality metric. In the United States, CMS penalizes hospitals with excess readmission rates through the Hospital Readmissions Reduction Program (HRRP). In Canada, CIHI tracks readmission as a health system performance indicator. This project demonstrates how machine learning can identify high risk patients at discharge time, enabling targeted interventions that improve patient outcomes and reduce institutional costs.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Explainability](#explainability)
- [API Reference](#api-reference)
- [Clinical Dashboard](#clinical-dashboard)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Testing](#testing)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

MedRisk is built around four core capabilities.

**Clinical Feature Engineering.** Raw electronic health record data including ICD-9 diagnosis codes, medication records, and prior utilization history is transformed into clinically meaningful features. Diagnosis codes are grouped into 9 clinical categories (Circulatory, Respiratory, Diabetes, etc.), medication change patterns are quantified, and prior visit frequency is engineered into utilization risk indicators.

**Multi-Model Training with Experiment Tracking.** Four model architectures (Logistic Regression, Random Forest, XGBoost, LightGBM) are trained and compared using stratified cross-validation for model ranking. All experiments are logged to MLflow with full hyperparameter, metric, and artifact tracking. Optuna handles Bayesian hyperparameter optimization with PR-AUC as the objective.

**SHAP Explainability.** Every prediction comes with a human readable explanation. Global SHAP summary plots reveal which features drive readmission risk across the population, while per-patient waterfall plots explain individual predictions. This transparency is essential for clinical adoption since clinicians need to understand why a patient is flagged, not just that they are.

**Production Ready Serving.** The best model is served through a FastAPI REST API with Pydantic input validation, risk tier classification (Low / Medium / High), and per-prediction SHAP explanations. The entire stack is containerized with Docker Compose and tested through GitHub Actions CI on every commit.

---

## How It Works

### Pipeline Steps

1. **Ingest** — The UCI Diabetes 130-US Hospitals dataset (101,766 encounters) is loaded and validated. Missing value markers (`?`) are converted to proper NaN representations.

2. **Clean** — Clinically irrelevant features are dropped (patient IDs, encounter IDs, payer codes). Weight is removed due to 97% missingness. Duplicate patient encounters are deduplicated to prevent data leakage.

3. **Engineer** — ICD-9 diagnosis codes are mapped to 9 clinical categories. Medication columns are aggregated into change counts and active medication counts. Prior utilization features (inpatient, outpatient, emergency visits) are combined into a total utilization score and a high utilizer flag.

4. **Encode** — Categorical features are one-hot encoded and numeric features are standardized through a scikit-learn ColumnTransformer pipeline that handles imputation, scaling, and encoding in a single reproducible step.

5. **Train** — Four models are trained on stratified train/validation splits for tuning, then ranked with stratified cross-validation PR-AUC. XGBoost and LightGBM use `scale_pos_weight` to handle class imbalance. All runs are logged to MLflow.

6. **Evaluate** — Models are compared on ROC-AUC, PR-AUC, F1 score, precision, and recall. Threshold tuning optimizes for clinical utility (target: 70%+ recall to minimize missed high risk patients). Calibration curves verify that predicted probabilities are reliable.

7. **Explain** — SHAP TreeExplainer generates global feature importance plots and per-patient waterfall explanations. Calibration quality is monitored via calibration curves and Brier score.

8. **Serve** — The best model is deployed as a FastAPI service with /predict and /predict/batch endpoints. A Streamlit dashboard provides a clinician facing interface for interactive risk assessment.

### Design Decisions

**Why binarize readmission at 30 days?** The 30 day window aligns with CMS HRRP penalty criteria and is the standard clinical benchmark. Predicting `<30 days` vs `>30 days or not readmitted` creates a clinically actionable binary classification.

**Why PR-AUC over ROC-AUC as the primary optimization metric?** With only 11% positive class prevalence, ROC-AUC can be misleadingly optimistic. PR-AUC directly measures performance on the minority class (readmitted patients), which is what clinicians care about.

**Why threshold tuning?** The default 0.5 classification threshold is arbitrary and suboptimal for imbalanced clinical data. Tuning the threshold to achieve at least 70% recall ensures that most high risk patients are identified, even at the cost of some false positives. In clinical settings, missing a high risk patient (false negative) is far more costly than an unnecessary follow-up call (false positive).

**Why SHAP over feature importance?** Built-in feature importance from tree models tells you which features the model uses most, but not how they affect individual predictions or in which direction. SHAP values provide additive, per-feature explanations for every single prediction, enabling clinicians to understand the reasoning behind each risk score.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│                                                                  │
│  UCI Diabetes 130-Hospitals Dataset (101,766 encounters)         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │ Demographics│ │ Diagnoses    │  │ Medications   │              │
│  │ Age, Race,  │ │ ICD-9 Codes  │  │ 23 Drug Cols  │              │
│  │ Gender      │ │ diag_1/2/3   │  │ Dosage Changes│              │
│  └──────┬─────┘  └──────┬───────┘  └──────┬────────┘              │
│         │               │                 │                      │
│         └───────────────┼─────────────────┘                      │
│                         ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│                   FEATURE ENGINEERING                            │
│                                                                  │
│  ICD-9 → 9 Clinical Categories    Medication Change Counts       │
│  Age Bracket → Ordinal Encoding   Prior Utilization Score        │
│  Missingness Flags                High Utilizer Flag             │
│                         │                                        │
│                         ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│                    MODEL TRAINING                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Logistic Reg │  │ XGBoost  │  │ LightGBM │  │ Random      │ │
│  │ (Baseline)   │  │          │  │          │  │ Forest      │ │
│  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│         └───────────────┼─────────────┼────────────────┘        │
│                         ▼                                        │
│              ┌──────────────────┐                                │
│              │   Optuna HPO     │                                │
│              │   50 Trials      │                                │
│              │   PR-AUC Target  │                                │
│              └────────┬─────────┘                                │
│                       ▼                                          │
│              ┌──────────────────┐                                │
│              │  MLflow Tracking │                                │
│              │  Params/Metrics  │                                │
│              │  Model Registry  │                                │
│              └────────┬─────────┘                                │
│                       ▼                                          │
├──────────────────────────────────────────────────────────────────┤
│                   EVALUATION                                     │
│                                                                  │
│  ROC-AUC │ PR-AUC │ F1 │ Precision │ Recall │ Calibration      │
│  Threshold Tuning (Target: ≥70% Recall)                         │
│                       │                                          │
│                       ▼                                          │
├──────────────────────────────────────────────────────────────────┤
│                 EXPLAINABILITY                                   │
│                                                                  │
│  SHAP TreeExplainer                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ Summary Plot   │  │ Waterfall Plot │  │ Force Plot        │  │
│  │ (Global)       │  │ (Per-Patient)  │  │ (Individual)      │  │
│  └────────────────┘  └────────────────┘  └───────────────────┘  │
│                       │                                          │
│                       ▼                                          │
├──────────────────────────────────────────────────────────────────┤
│                   SERVING LAYER                                  │
│                                                                  │
│  ┌────────────────────────┐    ┌──────────────────────────────┐ │
│  │  FastAPI REST API      │    │  Streamlit Clinical Dashboard│ │
│  │  /predict              │◄───│  Risk Gauge Visualization    │ │
│  │  /predict/batch        │    │  SHAP Waterfall Display      │ │
│  │  /health               │    │  Batch CSV Upload            │ │
│  │  Pydantic Validation   │    │  Per-Patient Explanation     │ │
│  └────────────────────────┘    └──────────────────────────────┘ │
│                                                                  │
│  Docker Compose │ GitHub Actions CI │ pytest                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

**Citation:** Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, Volume 2014.

| Property | Value |
|---|---|
| Total Encounters | 101,766 |
| Features | 50 (demographics, diagnoses, medications, utilization) |
| Target Variable | Readmitted within 30 days (binary) |
| Positive Class Rate | ~11.2% |
| Time Period | 1999 to 2008 |
| Hospitals | 130 US hospitals |
| Patient Type | Diabetic inpatients |

### Key Feature Groups

**Demographics:** Age (10 year brackets), race, gender

**Clinical:** Primary, secondary, and tertiary ICD-9 diagnosis codes; number of lab procedures; number of diagnoses

**Medications:** 23 medication columns (metformin, insulin, glipizide, etc.) each recording whether dosage was changed (Up, Down, Steady, No)

**Utilization:** Number of prior inpatient, outpatient, and emergency visits; time in hospital; number of procedures

**Administrative:** Admission type, discharge disposition, admission source

### Data Quality Notes

- The `weight` feature is missing in 97% of records and is dropped
- `payer_code` is missing in 40% and is dropped
- `medical_specialty` is missing in 49%; missingness is encoded as a binary indicator feature
- The `?` character represents missing values throughout the dataset and is converted to NaN during ingestion
- Duplicate patient encounters (same `patient_nbr`) are deduplicated to prevent data leakage between train/test splits

---

## Feature Engineering

Feature engineering is the most impactful component of this pipeline. The raw dataset contains high cardinality categorical codes and sparse medication records that require domain aware transformation.

### ICD-9 Diagnosis Code Grouping

The three diagnosis columns (`diag_1`, `diag_2`, `diag_3`) contain raw ICD-9 codes with thousands of unique values. These are mapped to 9 clinically meaningful categories.

| ICD-9 Range | Clinical Category |
|---|---|
| 390 to 459, 785 | Circulatory |
| 460 to 519, 786 | Respiratory |
| 520 to 579, 787 | Digestive |
| 250 | Diabetes |
| 800 to 999 | Injury |
| 710 to 739 | Musculoskeletal |
| 580 to 629, 788 | Genitourinary |
| 140 to 239 | Neoplasms |
| V and E codes | External |
| All others | Other |

### Medication Change Features

The 23 medication columns are aggregated into clinically interpretable features.

| Feature | Description |
|---|---|
| `num_med_changes` | Count of medications with dosage changed (Up or Down) |
| `num_meds_active` | Count of medications currently prescribed (Up, Down, or Steady) |
| `insulin_changed` | Binary flag indicating whether insulin dosage was modified |

### Utilization Features

| Feature | Description |
|---|---|
| `total_prior_visits` | Sum of inpatient + outpatient + emergency visits in the prior year |
| `high_utilizer` | Binary flag: 1 if total prior visits >= 5 |

### Missingness Indicators

| Feature | Description |
|---|---|
| `medical_specialty_missing` | Binary flag: 1 if medical specialty is not recorded |

### Encoding Pipeline

Categorical features are one-hot encoded and numeric features are standardized using a scikit-learn `ColumnTransformer` with `SimpleImputer` for handling remaining missing values.

**Numeric features:** age_ordinal, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_diagnoses, num_med_changes, num_meds_active, total_prior_visits

**Categorical features:** race, gender, diag1_category, diag2_category, diag3_category, admission_type_id

---

## Model Training and Evaluation

### Models Compared

| Model | Purpose | Imbalance Strategy |
|---|---|---|
| Logistic Regression | Interpretable linear baseline | `class_weight='balanced'` |
| Random Forest | Non-linear ensemble baseline | `class_weight='balanced'` |
| XGBoost | Primary gradient boosted candidate | `scale_pos_weight` tuned via Optuna |
| LightGBM | Primary gradient boosted candidate | `scale_pos_weight` tuned via Optuna |

### Experiment Tracking

All training runs are logged to MLflow.

| Logged Artifact | Description |
|---|---|
| Hyperparameters | All model parameters including imbalance strategy |
| Metrics | ROC-AUC, PR-AUC, F1, Precision, Recall on validation set |
| Model Artifacts | Serialized model objects (.joblib) |
| Feature Importance | SHAP-based global feature rankings |
| Confusion Matrices | Per-threshold confusion matrices |
| Calibration Curves | Reliability diagrams for probability calibration |

### Hyperparameter Optimization

Optuna runs 50 Bayesian optimization trials with PR-AUC as the objective. The search space includes:

| Parameter | Range |
|---|---|
| n_estimators | 100 to 1000 |
| max_depth | 3 to 10 |
| learning_rate | 0.01 to 0.3 (log scale) |
| subsample | 0.6 to 1.0 |
| colsample_bytree | 0.6 to 1.0 |
| scale_pos_weight | 5 to 15 |

### Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| ROC-AUC | Overall discriminative performance across all thresholds |
| PR-AUC | Performance on the minority class (readmitted patients), robust to class imbalance |
| Recall | Proportion of actual readmissions correctly identified (clinical priority) |
| Precision | Proportion of flagged patients who are actually readmitted (resource efficiency) |
| F1 Score | Harmonic mean of precision and recall |
| Calibration | Reliability of predicted probabilities (essential for clinical trust) |

### Threshold Tuning

The default 0.5 threshold is replaced with a clinically optimized threshold that maximizes precision while maintaining at least 70% recall. This prioritizes catching high risk patients while keeping false alarm rates manageable.

---

## Explainability

### Global Feature Importance

SHAP summary plots reveal which features contribute most to readmission predictions across the entire test set. Typical top features include:

- Number of prior inpatient visits
- Number of diagnoses
- Time in hospital
- Medication change counts
- Primary diagnosis category

### Per-Patient Explanations

Every prediction includes the top 3 SHAP-based reasons for the risk score. Example output:

```json
{
  "risk_score": 0.73,
  "risk_tier": "High",
  "top_reasons": [
    {
      "feature": "number_inpatient",
      "impact": 0.142,
      "direction": "increased risk",
      "explanation": "4 prior inpatient visits in the last year"
    },
    {
      "feature": "num_med_changes",
      "impact": 0.098,
      "direction": "increased risk",
      "explanation": "3 medication dosage changes during this encounter"
    },
    {
      "feature": "diag1_category_Circulatory",
      "impact": 0.067,
      "direction": "increased risk",
      "explanation": "Primary diagnosis in circulatory disease category"
    }
  ],
  "model_version": "xgboost-v1.2-optuna"
}
```

### Probability Calibration

Calibration curves are generated to verify that predicted probabilities match observed readmission rates. If the model shows poor calibration (common with tree-based models), isotonic regression calibration is applied post-hoc using `CalibratedClassifierCV` fitted on the validation set.

---

## API Reference

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_version": "xgboost-v1.2-optuna",
  "last_updated": "2026-04-20T10:00:00Z"
}
```

### Single Prediction

```
POST /predict
```

**Request Body:**
```json
{
  "age_ordinal": 7,
  "time_in_hospital": 5,
  "num_lab_procedures": 44,
  "num_medications": 18,
  "number_diagnoses": 9,
  "num_med_changes": 3,
  "num_meds_active": 8,
  "total_prior_visits": 4,
  "insulin_changed": 1,
  "race": "Caucasian",
  "gender": "Female",
  "diag1_category": "Circulatory"
}
```

**Response:**
```json
{
  "risk_score": 0.73,
  "risk_tier": "High",
  "top_reasons": [
    {"feature": "number_inpatient", "impact": 0.142},
    {"feature": "num_med_changes", "impact": 0.098},
    {"feature": "diag1_category_Circulatory", "impact": 0.067}
  ],
  "model_version": "xgboost-v1.2-optuna"
}
```

### Batch Prediction

```
POST /predict/batch
```

Accepts a JSON array of patient feature objects. Returns an array of prediction responses.

### API Documentation

FastAPI auto-generates interactive API documentation available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Clinical Dashboard

The Streamlit dashboard provides a clinician facing interface for interactive risk assessment.

### Features

- **Patient Input Panel:** Sidebar with sliders, dropdowns, and numeric inputs for entering patient features
- **Risk Gauge:** Visual risk score display with color-coded tiers (Green for Low, Yellow for Medium, Red for High)
- **SHAP Waterfall:** Interactive per-patient SHAP waterfall plot showing the top contributing features
- **Top Reasons Cards:** Plain English explanation cards (for example: "4 prior inpatient visits in the last year increased readmission risk")
- **Batch Upload:** CSV upload tab for scoring multiple patients at once with downloadable results
- **Model Info:** Display of current model version, training metrics, and last update timestamp

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/krishi-shah/medrisk-readmission-predictor.git
cd medrisk-readmission-predictor
pip install -r requirements.txt
```

### Download the Dataset

Download the UCI Diabetes 130-Hospitals dataset and place it in the `data/raw/` directory.

```bash
mkdir -p data/raw
# Download from: https://archive.ics.uci.edu/dataset/296
# Place diabetic_data.csv and IDs_mapping.csv in data/raw/
```

### Train a Model

```bash
make train
```

This runs the full pipeline: data loading, feature engineering, model training with Optuna, MLflow logging, and model artifact saving.

### Start the API

```bash
make run-api
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Start the Dashboard

```bash
make run-ui
# Dashboard available at http://localhost:8501
```

### Run with Docker

```bash
make docker-up
# API at http://localhost:8000
# Dashboard at http://localhost:8501
```

### View Experiment History

```bash
make mlflow-ui
# MLflow UI at http://localhost:5000
```

### Makefile Reference

| Command | Description |
|---|---|
| `make install` | Install all Python dependencies |
| `make train` | Run full training pipeline with MLflow tracking |
| `make run-api` | Start FastAPI server on port 8000 |
| `make run-ui` | Start Streamlit dashboard on port 8501 |
| `make test` | Run pytest test suite |
| `make evaluate` | Run full evaluation pipeline and quality-gate checks |
| `make docker-up` | Build and start all services with Docker Compose |
| `make mlflow-ui` | Launch MLflow experiment tracking UI |

---

## Project Structure

```
medrisk-readmission-predictor/
├── data/
│   ├── raw/                           # Original UCI CSV files
│   ├── processed/                     # Cleaned and feature-engineered datasets
│   └── sample/                        # Small sample for testing and CI
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb   # Feature engineering development
│   └── 03_model_experiments.ipynb     # Model comparison and tuning
├── src/
│   ├── data/
│   │   ├── loader.py                  # Dataset loading and validation
│   │   └── preprocessor.py            # Cleaning, encoding, and pipeline
│   ├── features/
│   │   └── engineer.py                # ICD-9 mapping, medication features, utilization
│   ├── models/
│   │   ├── trainer.py                 # Training loop, CV ranking, Optuna, artifact outputs
│   │   ├── evaluator.py              # Metrics, calibration, threshold tuning
│   │   └── explainer.py              # SHAP analysis and plot generation
│   ├── serving/
│   │   ├── api.py                     # FastAPI application
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── predictor.py              # Model loading and inference logic
│   └── utils/
│       ├── config.py                  # Configuration and environment loader
│       └── logger.py                  # Structured logging setup
├── tests/
│   ├── test_preprocessor.py           # Data cleaning and encoding tests
│   ├── test_engineer.py               # Feature engineering edge case tests
│   ├── test_predictor.py              # Inference output validation tests
│   └── test_api.py                    # API endpoint and schema tests
├── mlruns/                            # MLflow experiment tracking data
├── models/                            # Saved model artifacts (.joblib)
├── reports/
│   └── figures/                       # SHAP plots, confusion matrices, calibration curves
├── ui/
│   └── app.py                         # Streamlit clinical dashboard
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI pipeline
├── Dockerfile                         # API container definition
├── Dockerfile.ui                      # Dashboard container definition
├── docker-compose.yml                 # Multi-service orchestration
├── requirements.txt                   # Pinned Python dependencies
├── Makefile                           # Developer task shortcuts
├── config.yaml                        # Centralized hyperparameters and paths
├── .env.example                       # Environment variable template
├── LICENSE                            # MIT License
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Feature Engineering | scikit-learn Pipelines, ColumnTransformer | Reproducible data preprocessing and encoding |
| Model Training | XGBoost, LightGBM, scikit-learn | Gradient boosted and linear classifiers |
| Hyperparameter Tuning | Optuna | Bayesian hyperparameter optimization |
| Experiment Tracking | MLflow | Parameter, metric, and artifact logging with model registry |
| Explainability | SHAP | Global and per-patient feature contribution analysis |
| Class Imbalance | imbalanced-learn (SMOTE), class weighting | Oversampling and cost-sensitive learning comparison |
| Calibration | scikit-learn CalibratedClassifierCV | Post-hoc probability calibration for clinical reliability |
| API Serving | FastAPI, Pydantic, Uvicorn | REST API with automatic validation and documentation |
| Dashboard | Streamlit | Clinician facing interactive risk assessment interface |
| Containerization | Docker, Docker Compose | Reproducible deployment of API and dashboard |
| CI/CD | GitHub Actions, pytest | Automated testing on every commit |
| Data Processing | Pandas, NumPy | Data manipulation and numerical computation |
| Visualization | Matplotlib, Seaborn | EDA plots, calibration curves, confusion matrices |

---

## Testing

### Run the Full Test Suite

```bash
make test
```

### Run Individual Test Files

```bash
pytest tests/test_preprocessor.py -v
pytest tests/test_engineer.py -v
pytest tests/test_predictor.py -v
pytest tests/test_api.py -v
```

### Test Coverage

**test_preprocessor.py**
- Output shape matches expected feature count after encoding
- No NaN values remain in the processed output
- One-hot encoder handles previously unseen categories without errors
- Pipeline is serializable and reproducible

**test_engineer.py**
- ICD-9 code mapping handles all edge cases: V codes, E codes, decimal codes, and missing values
- Medication change count is always non-negative
- Age ordinal encoding correctly maps all 10 brackets
- High utilizer flag triggers at the correct threshold

**test_predictor.py**
- Output risk score is a float between 0.0 and 1.0
- Risk tier is exactly one of Low, Medium, or High
- Top reasons list contains exactly 3 items with valid feature names
- Model version string is present and non-empty

**test_api.py**
- `/health` endpoint returns HTTP 200 with correct schema
- `/predict` endpoint returns a valid `PredictionResponse` for well-formed input
- `/predict` endpoint returns HTTP 422 for invalid input with descriptive error messages
- `/predict/batch` endpoint correctly handles a list of patient feature objects
- API starts and shuts down cleanly

---

## Future Work

- **MIMIC-IV Integration:** Extend the pipeline to ingest MIMIC-IV clinical data for broader patient populations beyond diabetes
- **Temporal Feature Engineering:** Incorporate time series features from sequential lab measurements and vital signs
- **Deep Learning Baseline:** Add a TabNet or FT-Transformer model for comparison against gradient boosted baselines
- **Fairness Auditing:** Evaluate model performance across demographic subgroups (race, gender, age) to identify and mitigate bias
- **FHIR Integration:** Build a FHIR-compliant data ingestion module for compatibility with modern EHR systems
- **Real-Time Scoring:** Integrate with hospital event streams for discharge-time risk assessment
- **Model Monitoring:** Add drift detection and performance monitoring for production deployment

---

## References

1. Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, Volume 2014.

2. Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30 (NIPS 2017). [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)

3. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)

4. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30 (NIPS 2017).

5. Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

6. Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
