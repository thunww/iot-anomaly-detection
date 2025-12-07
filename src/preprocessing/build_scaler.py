import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils.paths import TRAIN_NORMAL_PATH, SCALER_PATH

def build_scaler():
    print("[INFO] Loading normal samples...")
    X = np.load(TRAIN_NORMAL_PATH)

    print("[INFO] Fitting scaler on NORMAL only...")
    scaler = StandardScaler()
    scaler.fit(X)

    joblib.dump(scaler, SCALER_PATH)

    print("[OK] Scaler saved!")

if __name__ == "__main__":
    build_scaler()
