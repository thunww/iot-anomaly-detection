import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from src.autoencoder.model import DeepAE
from src.utils.paths import TRAIN_NORMAL_PATH, SCALER_PATH, AE_MODEL_PATH

def train_autoencoder():
    print("[INFO] Loading scaler & normal training data...")
    scaler = joblib.load(SCALER_PATH)

    X = np.load(TRAIN_NORMAL_PATH)
    X = scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    input_dim = X.shape[1]
    latent_dim = 128   # CHUẨN THEO YÊU CẦU

    model = DeepAE(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("[INFO] Training Autoencoder...")
    epochs = 200

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss={loss.item():.6f}")

    torch.save(model.state_dict(), AE_MODEL_PATH)
    print("[OK] Autoencoder saved at:", AE_MODEL_PATH)

if __name__ == "__main__":
    train_autoencoder()
