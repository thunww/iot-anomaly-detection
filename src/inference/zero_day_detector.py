import numpy as np
import torch
from autoencoder.model import DeepAE
import joblib

class ZeroDayDetector:
    def __init__(
        self,
        scaler_path="models/scaler.pkl",
        ae_path="models/autoencoder.pt",
        threshold_path="models/threshold.npy"
    ):
        self.scaler = joblib.load(scaler_path)
        self.threshold = float(np.load(threshold_path))

        # Load AE with dynamic input fixing
        dummy_dim = 40
        self.ae = DeepAE(input_dim=dummy_dim)
        try:
            self.ae.load_state_dict(torch.load(ae_path))
        except RuntimeError:
            pass
        self.ae.eval()

    def predict(self, X):
        X_scaled = self.scaler.transform(X)

        # Fix AE input dynamically 
        input_dim = X_scaled.shape[1]
        if self.ae.encoder[0].in_features != input_dim:
            self.ae = DeepAE(input_dim=input_dim)
            self.ae.load_state_dict(torch.load("models/autoencoder.pt"))
            self.ae.eval()

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            recon = self.ae(X_tensor)
            mse = ((X_tensor - recon)**2).mean(dim=1).numpy()

        anomaly = (mse > self.threshold).astype(int)
        return anomaly, mse
