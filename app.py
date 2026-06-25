"""
Step 9 — Streamlit App: Student Performance Predictor
File: app.py
Deploy to Streamlit Cloud with model artifacts in same repo folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ─── Load artifacts ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    preprocessor = joblib.load(os.path.join(base, 'preprocessor.pkl'))
    selector     = joblib.load(os.path.join(base, 'feature_selector.pkl'))
    ridge        = joblib.load(os.path.join(base, 'ridge_model.pkl'))
    rf           = joblib.load(os.path.join(base, 'rf_model.pkl'))
    xgb          = joblib.load(os.path.join(base, 'xgb_model.pkl'))
    with open(os.path.join(base, 'metadata.json')) as f:
        meta = json.load(f)
    with open(os.path.join(base, 'column_info.json')) as f:
        col_info = json.load(f)
    return preprocessor, selector, ridge, rf, xgb, meta, col_info

try:
    preprocessor, selector, ridge, rf, xgb, meta, col_info = load_artifacts()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ─── Header ──────────────────────────────────────────────────
st.title("🎓 Student Performance Predictor")
st.markdown("Predict a student's **average score** (Math + Reading + Writing) using demographic & socioeconomic features.")
st.markdown("---")

# ─── Sidebar — Model Info ────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"**Best Model:** {meta.get('best_model','XGBoost')}")
    st.markdown(f"**Task:** Regression")
    perf = meta.get('performance', {})
    best = meta.get('best_model','XGBoost')
    if best in perf:
        st.markdown(f"**Test R²:** {perf[best]['R2']:.4f}")
        st.markdown(f"**Test MAE:** {perf[best]['MAE']:.4f}")
        st.markdown(f"**Test RMSE:** {perf[best]['RMSE']:.4f}")
    st.markdown("---")
    st.markdown("**Models Used:**")
    st.markdown("- Ridge Regression")
    st.markdown("- Random Forest")
    st.markdown("- XGBoost ⭐")
    st.markdown("---")
    st.markdown("*Built by Shivakumar — ML Internship Project*")

# ─── Input form ──────────────────────────────────────────────
st.subheader("🔢 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])

    parental_education = st.selectbox(
        "Parental Level of Education",
        ["some high school", "high school", "some college",
         "associate's degree", "bachelor's degree", "master's degree"]
    )

    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

with col2:
    race_ethnicity = st.selectbox(
        "Race / Ethnicity",
        ["group a", "group b", "group c", "group d", "group e"]
    )

    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

with col3:
    st.markdown("##### Derived Features (auto-calculated)")
    edu_rank_map = {
        "some high school": 0, "high school": 1,
        "some college": 2, "associate's degree": 3,
        "bachelor's degree": 4, "master's degree": 5
    }
    parental_edu_rank = edu_rank_map[parental_education]
    prep_completed    = 1 if test_prep == "completed" else 0
    ses_index         = 1 if lunch == "standard" else 0
    gender_code       = 1 if gender == "female" else 0
    race_group_code   = {"group a":0,"group b":1,"group c":2,"group d":3,"group e":4}[race_ethnicity]
    advantage_score   = prep_completed*2 + ses_index*2 + parental_edu_rank

    st.metric("Parental Edu Rank", parental_edu_rank)
    st.metric("Advantage Score", advantage_score)
    st.metric("Prep Completed", prep_completed)
    st.metric("SES Index", ses_index)

st.markdown("---")

# ─── Predict ─────────────────────────────────────────────────
if st.button("🔮 Predict Average Score", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        'parental_edu_rank':   parental_edu_rank,
        'advantage_score':     advantage_score,
        'gender_code':         gender_code,
        'race_group_code':     race_group_code,
        'prep_completed':      prep_completed,
        'ses_index':           ses_index,
        'math_reading_gap':    0.0,
        'reading_writing_gap': 0.0,
        'score_std':           0.0
    }])

    # Ensure column order matches training
    all_cols = col_info.get('all_feature_columns', list(input_data.columns))
    for c in all_cols:
        if c not in input_data.columns:
            input_data[c] = 0
    # Only keep columns the model saw
    num_cols = col_info.get('numerical_columns', list(input_data.columns))
    cat_cols_info = col_info.get('categorical_columns', [])
    keep_cols = [c for c in num_cols + cat_cols_info if c in input_data.columns]
    if keep_cols:
        input_data = input_data[keep_cols]

    try:
        input_t = preprocessor.transform(input_data)
        input_s = selector.transform(input_t)

        pred_ridge = float(ridge.predict(input_s)[0])
        pred_rf    = float(rf.predict(input_s)[0])
        pred_xgb   = float(xgb.predict(input_s)[0])
        pred_avg   = (pred_ridge + pred_rf + pred_xgb) / 3

        # Clip to valid range
        pred_ridge = np.clip(pred_ridge, 0, 100)
        pred_rf    = np.clip(pred_rf,    0, 100)
        pred_xgb   = np.clip(pred_xgb,   0, 100)
        pred_avg   = np.clip(pred_avg,   0, 100)

        st.markdown("## 🎯 Prediction Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔵 Ridge",         f"{pred_ridge:.1f}/100")
        c2.metric("🟢 Random Forest", f"{pred_rf:.1f}/100")
        c3.metric("🟠 XGBoost",       f"{pred_xgb:.1f}/100")
        c4.metric("⭐ Ensemble Avg",  f"{pred_avg:.1f}/100", help="Average of all 3 models")

        # Grade bucket
        def grade(score):
            if score >= 90: return "A+ 🌟", "green"
            elif score >= 75: return "A 🟢", "green"
            elif score >= 60: return "B 🔵", "blue"
            elif score >= 45: return "C 🟡", "orange"
            else: return "D 🔴", "red"

        g, color = grade(pred_avg)
        st.markdown(f"### Grade Estimate: **:{color}[{g}]**")

        # Interpretation
        st.markdown("---")
        st.subheader("📋 Interpretation")
        if prep_completed:
            st.success("✅ Test preparation completed — this significantly boosts predicted score.")
        else:
            st.warning("⚠️ No test preparation — completing it can improve score by 5–10 points.")
        if ses_index:
            st.info("🥗 Standard lunch (indicator of stable nutrition/resources) — positive factor.")
        if parental_edu_rank >= 4:
            st.info("🎓 High parental education level — strong positive correlation with performance.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Check that all .pkl files and column_info.json are in the same folder as app.py.")

# ─── All models comparison table ─────────────────────────────
st.markdown("---")
st.subheader("📈 Model Performance Summary (Test Set)")
if perf:
    df_perf = pd.DataFrame(perf).T
    df_perf.index.name = 'Model'
    st.dataframe(df_perf.style.highlight_max(axis=0, color='#c8f0c8'), use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption("Student Performance Predictor | ML Internship Project | VEMU Institute of Technology")
