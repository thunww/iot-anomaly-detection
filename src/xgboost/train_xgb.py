import numpy as np
import xgboost as xgb
from src.utils.paths import (
    Z_NORMAL, Z_ATTACK, XGB_MODEL_PATH
)

def train_xgb():

    print("[INFO] Loading latent vectors...")

    z_normal = np.load(Z_NORMAL)
    z_attack = np.load(Z_ATTACK)

    X = np.concatenate([z_normal, z_attack], axis=0)
    y = np.concatenate([
        np.zeros(len(z_normal)),
        np.ones(len(z_attack))
    ])

    print("[INFO] Training XGBoost classifier...")

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss"
    )

    model.fit(X, y)
    model.save_model(XGB_MODEL_PATH)

    print("[OK] XGBoost saved at:", XGB_MODEL_PATH)


if __name__ == "__main__":
    train_xgb()
