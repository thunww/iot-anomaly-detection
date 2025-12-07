import numpy as np
import torch
import joblib
import xgboost as xgb

from src.autoencoder.model import DeepAE
from src.utils.paths import (
    SCALER_PATH, AE_MODEL_PATH, THRESHOLD_PATH, XGB_MODEL_PATH
)


class HybridDetector:
    def __init__(self, latent_dim=128):
        # Load scaler
        self.scaler = joblib.load(SCALER_PATH)

        # Detect input dim automatically
        dummy = np.zeros((1, 27), dtype=np.float32)   # số feature fixed
        input_dim = dummy.shape[1]

        # Load AE model
        self.ae = DeepAE(input_dim=input_dim, latent_dim=latent_dim)
        self.ae.load_state_dict(torch.load(AE_MODEL_PATH))
        self.ae.eval()

        # Load threshold
        self.threshold = np.load(THRESHOLD_PATH)

        # Load XGBoost
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(XGB_MODEL_PATH)

    def predict(self, X):
        # scale
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # autoencoder reconstruction
        with torch.no_grad():
            recon = self.ae(X_tensor).numpy()

        mse = np.mean((X_scaled - recon) ** 2, axis=1)
        ae_flag = mse > self.threshold * 1.5

        # latent vector -> xgb
        z = self.ae.encode(X_tensor).detach().numpy()
        xgb_pred = self.xgb.predict(z)

        final = np.where((ae_flag == 1) | (xgb_pred == 1), 1, 0)
        return final
