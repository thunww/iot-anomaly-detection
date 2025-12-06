import pandas as pd
import numpy as np
import torch
import joblib
import xgboost as xgb

from model import DeepAE
from preprocessing_features_advanced import create_features

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
sample = df.iloc[[100]]

df2 = create_features(df.copy())
features = df2.drop(columns=["date", "time", "type", "label"]).iloc[[100]]

scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=128).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

pred = model.predict(z)[0]

print("Prediction:", "ATTACK ❌" if pred == 1 else "NORMAL ✅")
