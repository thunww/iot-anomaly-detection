import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
from src.autoencoder.model import DeepAE
from src.preprocessing.build_scaler import load_scaler
from src.utils.paths import AE_PATH

# Paths
DATA_NORMAL = "data/processed/train_normal.npy"
DATA_ATTACK = "data/processed/train_attack.npy"
THRESHOLD_PATH = "models/ae_threshold.npy"


def clean(x):
    """Safely remove NaN / Inf from arrays."""
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0)


def tune_threshold():
    print("[INFO] Loading data...")

    # Load data
    X_normal = np.load(DATA_NORMAL)
    X_attack = np.load(DATA_ATTACK)

    # Load scaler + AE
    scaler = load_scaler()
    ae = DeepAE(input_dim=X_normal.shape[1])
    ae.load_state_dict(torch.load(AE_PATH))
    ae.eval()

    # Combine data
    X = np.vstack([X_normal, X_attack])
    y = np.array([0]*len(X_normal) + [1]*len(X_attack))

    # Scale
    X_scaled = scaler.transform(X)
    X_scaled = clean(X_scaled)       # FIX 1

    # Torch
    X_torch = torch.tensor(X_scaled, dtype=torch.float32)

    # AE reconstruction
    with torch.no_grad():
        recon = ae(X_torch).numpy()

    recon = clean(recon)             # FIX 2

    # Reconstruction error
    mse = np.mean((X_scaled - recon)**2, axis=1)
    mse = clean(mse)                 # FIX 3

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y, mse)
    auc_score = auc(fpr, tpr)

    # Best threshold (Youden’s J)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]

    print(f"[OK] ROC-AUC = {auc_score:.4f}")
    print(f"[OK] Best threshold = {best_threshold}")

    # Save threshold
    np.save(THRESHOLD_PATH, best_threshold)
    print("[OK] Threshold saved!")


if __name__ == "__main__":
    tune_threshold()
