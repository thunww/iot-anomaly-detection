import numpy as np
import torch
import joblib
from sklearn.metrics import roc_curve

from src.autoencoder.model import DeepAE
from src.utils.paths import (
    SCALER_PATH, AE_MODEL_PATH,
    TEST_PATH, TEST_LABEL_PATH, THRESHOLD_PATH
)

def tune_threshold():
    print("[INFO] Loading data...")

    scaler = joblib.load(SCALER_PATH)
    X = np.load(TEST_PATH)
    y = np.load(TEST_LABEL_PATH)

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = DeepAE(input_dim=X.shape[1], latent_dim=128)
    model.load_state_dict(torch.load(AE_MODEL_PATH))
    model.eval()

    with torch.no_grad():
        recon = model(X_tensor).numpy()

    mse = np.mean((X_scaled - recon)**2, axis=1)

    # ROC to find best threshold
    fpr, tpr, thresholds = roc_curve(y, mse)

    # best threshold = maximize tpr - fpr
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    np.save(THRESHOLD_PATH, best_threshold)

    print(f"[OK] Best Threshold: {best_threshold}")

if __name__ == "__main__":
    tune_threshold()
