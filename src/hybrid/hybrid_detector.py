import numpy as np
import torch
import joblib
import xgboost as xgb

from src.autoencoder.model import DeepAE
from src.utils.paths import (
    SCALER_PATH, AE_MODEL_PATH, THRESHOLD_PATH, XGB_MODEL_PATH
)

class HybridDetector:
    def __init__(self, input_dim, latent_dim=128):
        self.scaler = joblib.load(SCALER_PATH)

        # AE
        self.ae = DeepAE(input_dim=input_dim, latent_dim=latent_dim)
        self.ae.load_state_dict(torch.load(AE_MODEL_PATH))
        self.ae.eval()

        # threshold
        self.threshold = np.load(THRESHOLD_PATH)

        # XGB
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(XGB_MODEL_PATH)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            recon = self.ae(X_tensor).numpy()

        mse = np.mean((X_scaled - recon)**2, axis=1)
        ae_flag_strict = mse > self.threshold * 1.5

        # latent → XGB
        z = self.ae.encode(X_tensor).detach().numpy()
        xgb_flag = self.xgb.predict(z)

        # Hybrid rule:
        final_flag = np.where(
            (xgb_flag == 1) | (ae_flag_strict == 1),
            1,
            0
        )

        return final_flag
