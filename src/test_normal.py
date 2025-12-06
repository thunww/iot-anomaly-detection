import pandas as pd
import numpy as np
import joblib
import torch
from model import DeepAE
from preprocessing_features_advanced import create_features
import xgboost as xgb

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD DATA (ONLY NORMAL)
# =============================
df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
df = df[df["label"] == 0]               # chỉ NORMAL
df2 = create_features(df.copy())

# chọn đúng feature columns (không lấy timestamp/delta)
features = df2.drop(columns=[
    "date", "time", "type", "timestamp", "delta", "label"
])

# =============================
# LOAD SCALER
# =============================
scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

# =============================
# LOAD AUTOENCODER
# =============================
input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=64).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

# =============================
# LOAD XGBOOST
# =============================
model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

# =============================
# INFERENCE
# =============================
pred = model.predict(z)

# NORMAL predicted as ATTACK = False Positive
fp = np.sum(pred == 1)
total = len(pred)

print(f"[NORMAL TEST] False Positives = {fp}/{total}")
print(f"False Positive Rate = {fp/total:.4f}")
