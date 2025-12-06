import pandas as pd
import numpy as np
import joblib
import torch
from model import DeepAE
from preprocessing_features_advanced import create_features
import xgboost as xgb

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD DATA (ONLY ATTACK)
# =============================
df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
df = df[df["label"] == 1]                # chỉ ATTACK
df2 = create_features(df.copy())

features = df2.drop(columns=[
    "date", "time", "type", "timestamp", "delta", "label"
])

# =============================
# SCALING
# =============================
scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

# =============================
# AUTOENCODER
# =============================
input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=64).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

# =============================
# XGBOOST
# =============================
model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

pred = model.predict(z)

# ATTACK predicted as ATTACK = True Positive
tp = np.sum(pred == 1)
total = len(pred)

print(f"[ATTACK TEST] True Positive Rate = {tp/total:.4f}")
