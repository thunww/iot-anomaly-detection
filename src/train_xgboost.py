import numpy as np
import xgboost as xgb
import joblib

X = np.load("data/processed/z_train.npy")
y = np.load("data/processed/train_labels.npy")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss"
)

print("[INFO] Training XGBoost...")
model.fit(X, y)

model.save_model("models/xgboost.json")
print("[OK] Saved XGBoost model")
