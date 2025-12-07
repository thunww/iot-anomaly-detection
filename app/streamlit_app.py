import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import torch
import plotly.express as px

# =====================================================
# FIX PYTHONPATH
# =====================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocessing.feature_engineering import build_features
from src.hybrid.hybrid_detector import HybridDetector


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_detector():
    return HybridDetector()

detector = load_detector()


# =====================================================
# PAGE CONFIG + HEADER
# =====================================================
st.set_page_config(page_title="IoT Zero-Day Detector", layout="wide")
st.title("⚡ IoT Zero-Day Attack Detection Dashboard")
st.caption("Hybrid Model = Autoencoder + XGBoost (TON_IoT Dataset)")


# =====================================================
# FILE UPLOADER
# =====================================================
uploaded = st.file_uploader("📁 Upload CSV (date,time,FC1,FC2,FC3,FC4)", type=["csv"])

if uploaded:

    # RAW DATA ======================================================
    st.header("📌 1. Raw Input Data")
    df_raw = pd.read_csv(uploaded)
    st.dataframe(df_raw.head(), use_container_width=True)


    # FEATURE ENGINEERING ==========================================
    st.header("✨ 2. Feature Engineering (27 Features)")
    df_feat = build_features(df_raw)

    # LIST FEATURES ONLY
    feature_cols = [
        c for c in df_feat.columns
        if c not in ["date", "time", "label", "type", "date_time"]
    ]
    X = df_feat[feature_cols].values

    st.dataframe(df_feat.head(), use_container_width=True)


    # PREDICTION ====================================================
    st.header("🚨 3. Prediction Results")

    preds = detector.predict(X).astype(int)
    df_feat["prediction"] = preds
    df_feat["result"] = df_feat["prediction"].map({0: "NORMAL", 1: "ATTACK"})


    # COLORED TABLE ================================================
    def highlight_row(row):
        if row["result"] == "ATTACK":
            return ['background-color: #ffcccc'] * len(row)
        else:
            return ['background-color: #ccffcc'] * len(row)

    st.subheader("🔍 Detailed Table (Colored Results)")
    st.dataframe(
        df_feat.style.apply(highlight_row, axis=1),
        use_container_width=True
    )


    # SUMMARY METRICS ===============================================
    st.header("📊 4. Summary")

    normal_count = (preds == 0).sum()
    attack_count = (preds == 1).sum()

    c1, c2 = st.columns(2)
    c1.metric("🟩 NORMAL", normal_count)
    c2.metric("🟥 ATTACK", attack_count)


    # PIE CHART =====================================================
    st.subheader("📈 Attack vs Normal Visualization")

    pie_df = pd.DataFrame({
        "Type": ["NORMAL", "ATTACK"],
        "Count": [normal_count, attack_count]
    })

    fig = px.pie(
        pie_df,
        names="Type",
        values="Count",
        color="Type",
        color_discrete_map={"NORMAL": "green", "ATTACK": "red"},
        title="Distribution of Predictions"
    )
    st.plotly_chart(fig, use_container_width=True)


    # EXPORT =========================================================
    st.subheader("💾 Download Result File")
    csv = df_feat.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Result", csv, "prediction_output.csv", "text/csv")

else:
    st.info("⬆️ Upload CSV để bắt đầu phân tích")
