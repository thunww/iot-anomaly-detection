import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.paths import (
    XGB_MODEL_PATH,
    Z_TEST, TEST_LABEL_PATH
)

def evaluate_xgb():

    print("[INFO] Loading model & test data...")

    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)

    Z = np.load(Z_TEST)
    y_true = np.load(TEST_LABEL_PATH)

    preds = model.predict(Z)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, preds))

    print("\nClassification Report:")
    print(classification_report(y_true, preds))


if __name__ == "__main__":
    evaluate_xgb()
