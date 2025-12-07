import numpy as np
import torch
import joblib

from src.autoencoder.model import DeepAE
from src.utils.paths import (
    TRAIN_NORMAL_PATH, TRAIN_ATTACK_PATH, TEST_PATH,
    SCALER_PATH, AE_MODEL_PATH,
    Z_NORMAL, Z_ATTACK, Z_TEST
)

def extract_latent():

    print("[INFO] Loading scaler & AE model...")
    scaler = joblib.load(SCALER_PATH)

    # Determine feature dimension dynamically
    sample = np.load(TRAIN_NORMAL_PATH)
    input_dim = sample.shape[1]

    model = DeepAE(input_dim=input_dim, latent_dim=128)
    model.load_state_dict(torch.load(AE_MODEL_PATH))
    model.eval()

    def encode(path):
        X = np.load(path)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            Z = model.encode(X_tensor).numpy()

        return Z

    print("[INFO] Encoding latent vectors...")

    np.save(Z_NORMAL, encode(TRAIN_NORMAL_PATH))
    np.save(Z_ATTACK, encode(TRAIN_ATTACK_PATH))
    np.save(Z_TEST,   encode(TEST_PATH))

    print("[OK] Latent vectors saved!")

if __name__ == "__main__":
    extract_latent()
