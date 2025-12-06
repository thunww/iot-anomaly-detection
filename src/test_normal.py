import pandas as pd
import numpy as np
import joblib
import torch
from model import DeepAE
from preprocessing_features_advanced import create_features
import xgboost as xgb

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
df = df[df["label"] == 0]        # chỉ lấy NORMAL
df2 = create_features(df.copy())
features = df2.drop(columns=["date", "time", "type", "label"])

# Load scaler
scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

# AE load
input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=32).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

# Encode latent
X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

# XGBoost load
model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

pred = model.predict(z)

tp = np.sum(pred == 1)   # bị báo "ATTACK" nhầm
total = len(pred)

print(f"[NORMAL TEST] False positives = {tp}/{total}")
print(f"False Positive Rate = {tp/total:.4f}")
