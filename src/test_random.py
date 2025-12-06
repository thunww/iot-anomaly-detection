import pandas as pd
import numpy as np
import joblib
import torch
from model import DeepAE
from preprocessing_features_advanced import create_features
import xgboost as xgb

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/raw/Train_Test_IoT_Modbus.csv")
df_sample = df.sample(100)    # random
df2 = create_features(df_sample.copy())

labels = df_sample["label"].values
features = df2.drop(columns=["date", "time", "type", "label"])

scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

# AE
input_dim = scaled.shape[1]
ae = DeepAE(input_dim=input_dim, latent_dim=64).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

X = torch.tensor(scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    z = ae.encoder(X).cpu().numpy()

# XGB
model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

pred = model.predict(z)

for i in range(100):
    print(f"Row {i}:  True={labels[i]}  Pred={pred[i]}")

acc = np.sum(pred == labels) / 100
print("\nAccuracy on 100 random samples =", acc)
