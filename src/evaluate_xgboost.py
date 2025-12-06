import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

X = np.load("data/processed/z_test.npy")
y = np.load("data/processed/test_labels.npy")

model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

pred = model.predict(X)

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y, pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y, pred))
