import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import torch
import plotly.express as px

# =====================================================
# FIX PYTHONPATH: import src/*
# =====================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocessing.feature_engineering import build_features
from src.hybrid.hybrid_detector import HybridDetector


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="IoT Zero-Day Attack Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Tải file CSV gồm 6 cột:\n**date, time, FC1, FC2, FC3, FC4**")

uploaded = st.sidebar.file_uploader("📁 Upload CSV Input", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.write("**Model:** Hybrid Autoencoder + XGBoost")
st.sidebar.write("**Dataset:** TON_IoT (Telemetry/Modbus)")


# =====================================================
# LOAD MODEL (Cache)
# =====================================================
@st.cache_resource
def load_detector():
    return HybridDetector()

detector = load_detector()


# =====================================================
# MAIN TITLE
# =====================================================
st.title("⚡ IoT Zero-Day Attack Detection Dashboard")
st.caption("Hybrid Model = Autoencoder (Anomaly) + XGBoost (Classification)")


# =====================================================
# WHEN FILE IS UPLOADED
# =====================================================
if uploaded:

    # ---------------------------------------------
    # 1) RAW INPUT
    # ---------------------------------------------
    st.header("📌 1. Raw Input Data")
    df_raw = pd.read_csv(uploaded)
    st.dataframe(df_raw.head(10), use_container_width=True)


    # ---------------------------------------------
    # 2) FEATURE ENGINEERING
    # ---------------------------------------------
    st.header("✨ 2. Feature Engineering (Derived features)")

    df_feat = build_features(df_raw)

    feature_cols = [
        c for c in df_feat.columns
        if c not in ["date", "time", "label", "type", "date_time"]
    ]

    X = df_feat[feature_cols].values

    st.dataframe(df_feat.head(10), use_container_width=True)


    # ---------------------------------------------
    # 3) PREDICT
    # ---------------------------------------------
    st.header("🚨 3. Prediction Results")

    preds = detector.predict(X).astype(int)
    df_feat["prediction"] = preds
    df_feat["result"] = df_feat["prediction"].map({0: "NORMAL", 1: "ATTACK"})


    # ---------------------------------------------
    # 3A) Safe PREVIEW TABLE (colored)
    # ---------------------------------------------
    st.subheader("🔍 Preview (First 200 rows with color)")

    def highlight(row):
        color = "#ffcccc" if row["result"] == "ATTACK" else "#ccffcc"
        return [f"background-color: {color}"] * len(row)

    preview = df_feat.head(200)

    st.dataframe(
        preview.style.apply(highlight, axis=1),
        use_container_width=True
    )


    # ---------------------------------------------
    # 3B) FULL TABLE (no color)
    # ---------------------------------------------
    st.subheader("📄 Full Prediction Table (no color to avoid Streamlit limit)")
    st.dataframe(df_feat, use_container_width=True)


    # ---------------------------------------------
    # 4) SUMMARY STATISTICS
    # ---------------------------------------------
    st.header("📊 4. Summary Statistics")

    normal_count = int((preds == 0).sum())
    attack_count = int((preds == 1).sum())

    c1, c2 = st.columns(2)
    c1.metric("🟩 NORMAL", normal_count)
    c2.metric("🟥 ATTACK", attack_count)


    # ---------------------------------------------
    # 5) PIE CHART
    # ---------------------------------------------
    st.subheader("📈 Attack vs Normal Distribution")

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
        title="Prediction Distribution",
    )

    st.plotly_chart(fig, use_container_width=True)


    # ---------------------------------------------
    # 6) DOWNLOAD OUTPUT
    # ---------------------------------------------
    st.header("💾 5. Export Results")

    csv_out = df_feat.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Prediction CSV",
        csv_out,
        "iot_prediction_output.csv",
        "text/csv"
    )


else:
    st.info("⬆️ Hãy upload file CSV để hệ thống bắt đầu phân tích IoT anomaly.")
