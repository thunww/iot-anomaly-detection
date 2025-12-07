import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb

from model import DeepAE
from preprocessing_features_advanced import create_features

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_input(df):
    """
    Preprocess dữ liệu đầu vào giống hệt pipeline training.
    """

    # 1. Feature engineering FULL
    df_feat = create_features(df.copy())

    # 2. Drop đúng 5 cột như training
    df_feat = df_feat.drop(
        columns=["date", "time", "type", "timestamp", "delta"],
        errors="ignore"
    )

    # 3. Clean INF / NaN giống training
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Clip giá trị giống training
    df_feat = df_feat.clip(-1e6, 1e6)

    return df_feat


def load_models():
    """Load scaler, AE, XGBoost đúng arch training."""
    scaler = joblib.load("models/scaler.pkl")

    # input_dim phải = 67 (số features sau khi train preprocessing)
    ae = DeepAE(input_dim=67, latent_dim=64).to(device)
    ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
    ae.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost.json")

    return scaler, ae, xgb_model


def infer(df, scaler, ae, xgb_model):
    """
    Chạy full pipeline: preprocess → scale → AE → latent → XGB.
    """

    df_feat = preprocess_input(df)

    # Scale
    X_scaled = scaler.transform(df_feat)

    # AE latent
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = ae.encoder(X_tensor).cpu().numpy()

    # XGBoost predict
    pred = xgb_model.predict(z)
    prob = xgb_model.predict_proba(z)[:, 1]

    df_out = df.copy()
    df_out["prediction"] = pred
    df_out["prob_attack"] = prob

    return df_out
