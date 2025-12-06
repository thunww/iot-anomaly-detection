import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb
from model import DeepAE
from preprocessing_features_advanced import create_features

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
df_attack = df[df["label"] == 1]  # attack only

df2 = create_features(df_attack.copy())
features = df2.drop(columns=["date", "time", "type", "label"])

scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=64).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

pred = model.predict(z)

tpr = np.sum(pred == 1) / len(pred)
print("[ATTACK TEST] True Positive Rate =", tpr)
