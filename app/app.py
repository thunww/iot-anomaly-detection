import streamlit as st
import pandas as pd
from utils import load_models
from inference import predict_single, predict_batch
from preprocessing_features_advanced import create_features

st.set_page_config(page_title="IoT Anomaly Detection", layout="wide")

st.title("🔎 IoT Zero-Day Attack Detection (Autoencoder + XGBoost)")

scaler, ae, xgb_model = load_models()

tab1, tab2 = st.tabs(["📂 Upload CSV", "🧪 Test Single Input"])

# ============================================
# TAB 1 – UPLOAD CSV
# ============================================
with tab1:
    st.header("Upload a CSV file to detect anomalies")

    file = st.file_uploader("Choose CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        # Feature engineering
        df_feat = create_features(df.copy())
        df_feat = df_feat.drop(columns=["date","time","type","timestamp","delta","label"], errors="ignore")

        preds, probs = predict_batch(df_feat, scaler, ae, xgb_model)

        df["prediction"] = preds
        df["prob_attack"] = probs

        st.success("Done! Below is the prediction result:")
        st.dataframe(df)

# ============================================
# TAB 2 – SINGLE INPUT
# ============================================
with tab2:
    st.header("Test a single IoT packet")

    f1 = st.number_input("FC1_Read_Input_Register", value=50000)
    f2 = st.number_input("FC2_Read_Discrete_Value", value=52000)
    f3 = st.number_input("FC3_Read_Holding_Register", value=25000)
    f4 = st.number_input("FC4_Read_Coil", value=15000)

    if st.button("Detect"):
        row = [f1, f2, f3, f4]

        # Create feature dataframe
        df_tmp = pd.DataFrame([row], columns=[
            "FC1_Read_Input_Register",
            "FC2_Read_Discrete_Value",
            "FC3_Read_Holding_Register",
            "FC4_Read_Coil"
        ])

        df_feat = create_features(df_tmp.copy())
        df_feat = df_feat.drop(columns=["date","time","type","timestamp","delta","label"], errors="ignore")

        pred, prob = predict_single(df_feat.iloc[0].values, scaler, ae, xgb_model)

        if pred == 1:
            st.error(f"🚨 Attack detected! (prob = {prob:.4f})")
        else:
            st.success(f"✔ Normal traffic (prob attack = {prob:.4f})")
