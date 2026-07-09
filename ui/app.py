import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

API_BASE = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

RACE_OPTIONS = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]
GENDER_OPTIONS = ["Female", "Male"]
DIAG_CATEGORIES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "External", "Other",
]
DISPOSITION_GROUPS = ["Home", "Transfer", "Home_Health", "AMA_Left", "SNF_Rehab", "Other"]
ADMISSION_SOURCE_GROUPS = ["Referral", "Transfer", "Emergency", "Court_Law", "Other"]
AGE_BRACKETS = {
    0: "[0-10)", 1: "[10-20)", 2: "[20-30)", 3: "[30-40)", 4: "[40-50)",
    5: "[50-60)", 6: "[60-70)", 7: "[70-80)", 8: "[80-90)", 9: "[90-100)",
}

TIER_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedRisk - Readmission Risk",
    page_icon="➕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.6rem;
        border-radius: 2rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        text-align: center;
    }
    .factor-card {
        background: #f8f9fa;
        border-radius: 0.75rem;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4361ee;
    }
    .factor-name {font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem;}
    .factor-bar-bg {
        background: #e0e0e0;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin-top: 0.25rem;
    }
    .factor-bar {
        height: 8px;
        border-radius: 4px;
    }
    .direction-up {color: #e74c3c;}
    .direction-down {color: #2ecc71;}
    .metric-card {
        background: #f8f9fa;
        border-radius: 0.75rem;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {font-size: 1.8rem; font-weight: 700; color: #4361ee;}
    .metric-label {font-size: 0.85rem; color: #666; margin-top: 0.3rem;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def check_health():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=120)
def get_model_info():
    try:
        resp = requests.get(f"{API_BASE}/model/info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def predict_single(features: dict):
    resp = requests.post(f"{API_BASE}/predict", json=features, timeout=30)
    if resp.status_code == 503:
        st.warning("Model is not loaded on the server. Please try again later.")
        return None
    resp.raise_for_status()
    return resp.json()


def predict_batch(patients: list[dict]):
    resp = requests.post(f"{API_BASE}/predict/batch", json={"patients": patients}, timeout=120)
    if resp.status_code == 503:
        st.warning("Model is not loaded on the server. Please try again later.")
        return None
    resp.raise_for_status()
    return resp.json()


def build_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48}, "valueformat": ".1%"},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".0%", "dtick": 0.2},
            "bar": {"color": "#4361ee", "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 0.3], "color": "#d4edda"},
                {"range": [0.3, 0.6], "color": "#fff3cd"},
                {"range": [0.6, 1.0], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
        title={"text": "Readmission Risk", "font": {"size": 20}},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=40, r=40))
    return fig


def render_factor_card(feature: str, impact: float, direction: str):
    is_risk_increase = direction == "increased risk"
    arrow = "↑" if is_risk_increase else "↓"
    dir_class = "direction-up" if is_risk_increase else "direction-down"
    bar_color = "#e74c3c" if is_risk_increase else "#2ecc71"
    bar_pct = min(abs(impact) * 100, 100)
    st.markdown(f"""
    <div class="factor-card">
        <div class="factor-name">{feature}</div>
        <div class="factor-bar-bg">
            <div class="factor-bar" style="width:{bar_pct}%; background:{bar_color};"></div>
        </div>
        <small class="{dir_class}"><b>{arrow} {direction}</b> &nbsp;|&nbsp; impact: {abs(impact):.3f}</small>
    </div>
    """, unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("# ➕ MedRisk")
    st.caption("30-Day Hospital Readmission Risk Prediction")

with col_status:
    health = check_health()
    if health:
        st.success("API Connected", icon="🟢")
        st.caption(f"Model v{health.get('model_version', '?')}")
    else:
        st.error("API Offline", icon="🔴")

st.divider()

# ── Sidebar – Patient Input Panel ────────────────────────────────────────────

with st.sidebar:
    st.header("Patient Input Panel")
    st.caption("Enter clinical features below.")

    age_ordinal = st.number_input(
        "Age bracket",
        min_value=0, max_value=9, value=5,
        help="0 = [0-10), 1 = [10-20), … 5 = [50-60), … 9 = [90-100)",
    )
    time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 3)
    num_lab_procedures = st.number_input("Lab procedures", 0, 132, 40)
    num_procedures = st.number_input("Procedures", 0, 6, 1)
    num_medications = st.number_input("Medications", 0, 81, 15)
    number_diagnoses = st.number_input("Number of diagnoses", 1, 16, 7)
    num_med_changes = st.number_input("Medication changes", 0, 23, 1)
    num_meds_active = st.number_input("Active medications", 0, 23, 5)
    total_prior_visits = st.number_input("Total prior visits", 0, 40, 1)
    insulin_changed = st.selectbox("Insulin changed?", ["No", "Yes"])
    medical_specialty_missing = st.selectbox(
        "Medical specialty recorded?", ["No", "Yes"],
        help="Whether the attending physician's specialty was recorded",
    )
    race = st.selectbox("Race", RACE_OPTIONS)
    gender = st.selectbox("Gender", GENDER_OPTIONS)
    admission_type = st.selectbox(
        "Admission type",
        ["1 — Emergency", "2 — Urgent", "3 — Elective", "4 — Newborn", "5 — Not available"],
        help="1=Emergency, 2=Urgent, 3=Elective",
    )
    diag1_category = st.selectbox("Primary diagnosis", DIAG_CATEGORIES, key="d1")
    diag2_category = st.selectbox("Secondary diagnosis", DIAG_CATEGORIES, index=3, key="d2")
    diag3_category = st.selectbox("Tertiary diagnosis", DIAG_CATEGORIES, index=2, key="d3")

    st.divider()
    st.subheader("v2 Features")

    num_emergency_visits = st.number_input("Emergency visits (prior year)", 0, 40, 0)
    num_inpatient_visits = st.number_input("Inpatient visits (prior year)", 0, 40, 0)
    num_outpatient_visits = st.number_input("Outpatient visits (prior year)", 0, 40, 0)
    a1c_abnormal = st.selectbox("A1C result abnormal?", ["No", "Yes"])
    diabetes_med = st.selectbox("On diabetes medication?", ["No", "Yes"])
    med_changed = st.selectbox("Medication changed this encounter?", ["No", "Yes"])
    discharge_group = st.selectbox("Discharge disposition", DISPOSITION_GROUPS)
    admission_source = st.selectbox("Admission source", ADMISSION_SOURCE_GROUPS)

    st.divider()
    predict_clicked = st.button("🔍  Predict", use_container_width=False, type="primary")


def _build_payload() -> dict:
    tih = max(1, time_in_hospital)
    return {
        "age_ordinal": age_ordinal,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "num_med_changes": num_med_changes,
        "num_meds_active": num_meds_active,
        "total_prior_visits": total_prior_visits,
        "insulin_changed": 1 if insulin_changed == "Yes" else 0,
        "high_utilizer": 1 if total_prior_visits >= 5 else 0,
        "medical_specialty_missing": 0 if medical_specialty_missing == "Yes" else 1,
        "num_emergency_visits": num_emergency_visits,
        "num_inpatient_visits": num_inpatient_visits,
        "num_outpatient_visits": num_outpatient_visits,
        "lab_procedure_intensity": round(num_lab_procedures / tih, 4),
        "procedure_intensity": round(num_procedures / tih, 4),
        "medication_intensity": round(num_medications / tih, 4),
        "diagnosis_complexity": round(number_diagnoses / tih, 4),
        "med_change_ratio": round(num_med_changes / (num_meds_active + 1), 4),
        "A1Cresult_abnormal": 1 if a1c_abnormal == "Yes" else 0,
        "diabetesMed_flag": 1 if diabetes_med == "Yes" else 0,
        "change_flag": 1 if med_changed == "Yes" else 0,
        "race": race,
        "gender": gender,
        "admission_type_id": admission_type.split(" — ")[0],
        "diag1_category": diag1_category,
        "diag2_category": diag2_category,
        "diag3_category": diag3_category,
        "discharge_disposition_grouped": discharge_group,
        "admission_source_grouped": admission_source,
    }


# ── Main Area Tabs ───────────────────────────────────────────────────────────

tab_single, tab_batch, tab_model = st.tabs([
    "🧑‍⚕️ Single Patient", "📂 Batch Upload", "📊 Model Performance",
])

# ── Single Patient Tab ───────────────────────────────────────────────────────

with tab_single:
    if predict_clicked:
        payload = _build_payload()
        try:
            with st.spinner("Running prediction…"):
                result = predict_single(payload)
            if result is None:
                st.stop()

            score = result["risk_score"]
            tier = result["risk_tier"]
            color = TIER_COLORS.get(tier, "#888")

            col_gauge, col_info = st.columns([2, 1])
            with col_gauge:
                st.plotly_chart(build_gauge(score), use_container_width=False)
            with col_info:
                st.markdown(f"""
                <div style="text-align:center; margin-top:2rem;">
                    <span class="risk-badge" style="background:{color};">{tier} Risk</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Risk Score", f"{score:.1%}")
                st.metric("Age Bracket", AGE_BRACKETS.get(age_ordinal, "?"))
                st.metric("Hospital Days", time_in_hospital)

            st.divider()
            st.subheader("Top Risk Factors")

            reasons = result.get("top_reasons", [])
            if reasons:
                cols = st.columns(min(len(reasons), 3))
                for i, reason in enumerate(reasons):
                    with cols[i % len(cols)]:
                        render_factor_card(
                            reason["feature"],
                            reason["impact"],
                            reason["direction"],
                        )

            st.divider()
            st.caption(f"Model version: {result.get('model_version', 'N/A')}")

        except requests.ConnectionError:
            st.error(f"Could not connect to the MedRisk API. Is the server running at {API_BASE}?")
        except requests.HTTPError as exc:
            st.error(f"API returned an error: {exc.response.status_code} – {exc.response.text}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
    else:
        st.info("Configure patient features in the sidebar and click **Predict** to see results.")

# ── Batch Upload Tab ─────────────────────────────────────────────────────────

with tab_batch:
    uploaded = st.file_uploader("Upload a CSV of patient records", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(f"**{len(df)} patients loaded**")
            with st.expander("Preview uploaded data"):
                st.dataframe(df.head(10), use_container_width=False)

            if st.button("🚀  Run Batch Prediction", type="primary"):
                patients = df.to_dict(orient="records")
                try:
                    with st.spinner(f"Predicting for {len(patients)} patients…"):
                        batch_result = predict_batch(patients)
                    if batch_result is None:
                        st.stop()

                    preds = batch_result["predictions"]
                    result_df = df.copy()
                    result_df["risk_score"] = [p["risk_score"] for p in preds]
                    result_df["risk_tier"] = [p["risk_tier"] for p in preds]

                    st.divider()

                    tier_counts = result_df["risk_tier"].value_counts()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("High Risk", int(tier_counts.get("High", 0)))
                    m2.metric("Medium Risk", int(tier_counts.get("Medium", 0)))
                    m3.metric("Low Risk", int(tier_counts.get("Low", 0)))

                    tier_order = ["High", "Medium", "Low"]
                    tier_vals = [int(tier_counts.get(t, 0)) for t in tier_order]
                    tier_colors_list = [TIER_COLORS[t] for t in tier_order]
                    dist_fig = go.Figure(go.Bar(
                        x=tier_order,
                        y=tier_vals,
                        marker_color=tier_colors_list,
                        text=tier_vals,
                        textposition="outside",
                    ))
                    dist_fig.update_layout(
                        title="Risk Tier Distribution",
                        xaxis_title="Risk Tier",
                        yaxis_title="Patient Count",
                        height=300,
                        margin=dict(t=50, b=40, l=40, r=20),
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)

                    def _highlight_tier(val):
                        bg = TIER_COLORS.get(val, "")
                        if bg:
                            return f"background-color: {bg}; color: #fff; font-weight: 600; border-radius: 4px;"
                        return ""

                    styled = result_df.style.map(_highlight_tier, subset=["risk_tier"])
                    st.dataframe(styled, use_container_width=False, height=400)

                    csv_bytes = result_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️  Download Results CSV",
                        data=csv_bytes,
                        file_name="medrisk_predictions.csv",
                        mime="text/csv",
                    )

                except requests.ConnectionError:
                    st.error("Could not connect to the MedRisk API. Is the server running?")
                except requests.HTTPError as exc:
                    st.error(f"API error: {exc.response.status_code} – {exc.response.text}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")

# ── Model Performance Tab ────────────────────────────────────────────────────

with tab_model:
    st.subheader("Model Performance & Diagnostics")

    model_info = get_model_info()

    if model_info:
        col_a, col_b, col_c, col_d = st.columns(4)
        metrics = model_info.get("training_metrics", {}) or {}

        with col_a:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{metrics.get('roc_auc', 0):.3f}</div>
                <div class="metric-label">ROC-AUC</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{metrics.get('pr_auc', 0):.3f}</div>
                <div class="metric-label">PR-AUC</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{metrics.get('recall', 0):.1%}</div>
                <div class="metric-label">Recall</div>
            </div>""", unsafe_allow_html=True)
        with col_d:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{metrics.get('precision', 0):.1%}</div>
                <div class="metric-label">Precision</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        info_col, detail_col = st.columns(2)
        with info_col:
            st.markdown("**Model Details**")
            st.markdown(f"- **Type:** {model_info.get('model_type', 'N/A')}")
            st.markdown(f"- **Version:** {model_info.get('model_version', 'N/A')}")
            st.markdown(f"- **Threshold:** {model_info.get('threshold', 0.5):.4f}")
            st.markdown(f"- **Features:** {model_info.get('n_features', 0)}")

        with detail_col:
            st.markdown("**Additional Metrics**")
            st.markdown(f"- **F1 Score:** {metrics.get('f1', 0):.4f}")
            st.markdown(f"- **Brier Score:** {metrics.get('brier_score', 0):.4f}")

            violations = model_info.get("quality_gate_violations", [])
            if violations:
                st.warning(f"{len(violations)} quality gate violation(s)")
                for v in violations:
                    st.markdown(f"  - {v}")
            else:
                st.success("All quality gates passed")

        st.divider()
        st.subheader("Diagnostic Plots")

        reports_dir = Path("reports/figures")
        plot_files = {
            "SHAP Feature Importance": reports_dir / "shap_bar.png",
            "SHAP Beeswarm": reports_dir / "shap_beeswarm.png",
            "ROC Curve": reports_dir / "roc_curve.png",
            "Precision-Recall Curve": reports_dir / "pr_curve.png",
            "Confusion Matrix": reports_dir / "confusion_matrix.png",
            "Calibration Curve": reports_dir / "calibration_curve.png",
        }

        available_plots = {k: v for k, v in plot_files.items() if v.exists()}
        if available_plots:
            plot_cols = st.columns(2)
            for i, (title, path) in enumerate(available_plots.items()):
                with plot_cols[i % 2]:
                    st.markdown(f"**{title}**")
                    st.image(str(path), use_container_width=True)
        else:
            st.info(
                "No diagnostic plots available. Run `make train` to generate "
                "SHAP summaries, ROC/PR curves, and calibration plots."
            )
    else:
        st.info(
            "Could not load model info from the API. "
            "Make sure the API is running and a model has been trained."
        )
