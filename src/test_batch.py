import pandas as pd
import numpy as np
import joblib
import torch

from model import DeepAE
from preprocessing_features_advanced import create_features
import xgboost as xgb

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load raw test set
df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
df2 = create_features(df.copy())

# Only attack samples to evaluate
attacks = df2[df2["label"] == 1].copy()
attacks = attacks.drop(columns=["date", "time", "type", "label"])

print("[INFO] Total attack samples:", len(attacks))

# Load scaler (CORRECT PATH)
scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(attacks)

# Load AE
input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=32).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

# Get latent vector
X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

preds = model.predict(z)

# Report accuracy on attack-only set
correct = (preds == 1).sum()
total = len(preds)

print(f"\n[RESULT] Detected {correct}/{total} attacks")
print(f"Accuracy on attack samples = {correct/total:.4f}")
