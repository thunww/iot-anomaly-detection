import numpy as np
import torch
import joblib
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from model import DeepAE

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 1) LOAD DATA
# =============================
X_test = np.load("data/processed/test_features.npy")
y_test = np.load("data/processed/test_labels.npy")

print(f"[INFO] Test shape = {X_test.shape}, Labels = {y_test.shape}")

# =============================
# 2) LOAD AUTOENCODER
# =============================
input_dim = X_test.shape[1]
latent_dim = 64

ae = DeepAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
ae.eval()

# =============================
# 3) ENCODE LATENT
# =============================
X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    z = ae.encoder(X_t).cpu().numpy()

print(f"[INFO] Latent shape = {z.shape}")

# =============================
# 4) LOAD XGBOOST CLASSIFIER
# =============================
model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

# Predict
y_pred = model.predict(z)

# =============================
# 5) METRICS
# =============================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC-AUC cần probability
try:
    y_prob = model.predict_proba(z)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
except:
    auc = -1

print("\n========== EVALUATION RESULT ==========")
print("Confusion Matrix:")
print(cm)
print("----------------------------------------")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print(f"F1-score = {f1:.4f}")
print(f"ROC-AUC = {auc:.4f}")
print("----------------------------------------")
print(f"False Positive Rate  = {fp / (fp + tn):.4f}")
print(f"False Negative Rate  = {fn / (fn + tp):.4f}")
print("========================================\n")
