import pandas as pd
import numpy as np
from src.preprocessing.feature_engineering import build_features
from src.utils.paths import DATA_RAW, DATA_PROCESSED
import os

def prepare_dataset():
    print("[INFO] Loading raw dataset...")

    df = pd.read_csv(DATA_RAW)
    df = build_features(df)

    feature_cols = [c for c in df.columns if c not in ["date","time","date_time","label","type"]]

    X = df[feature_cols].values
    y = df["label"].values

    X_normal = X[y==0]
    X_attack = X[y==1]

    np.save(os.path.join(DATA_PROCESSED, "train_normal.npy"), X_normal[:20000])
    np.save(os.path.join(DATA_PROCESSED, "train_attack.npy"), X_attack[:20000])

    np.save(os.path.join(DATA_PROCESSED, "test.npy"), X)
    np.save(os.path.join(DATA_PROCESSED, "test_labels.npy"), y)

    print("[OK] Dataset saved!")

if __name__ == "__main__":
    prepare_dataset()
