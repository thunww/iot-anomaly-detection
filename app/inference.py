import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1) PREDICT CHO 1 DÒNG
# ============================================================
def predict_single(feature_row, scaler, ae, xgb_model):
    """
    feature_row: numpy array shape (N_features,)
    return: label (0/1) + prob attack
    """

    # Scale input
    scaled = scaler.transform([feature_row])

    # Encode latent
    X = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        z = ae.encoder(X).cpu().numpy()

    # Predict with XGBoost
    pred = xgb_model.predict(z)[0]
    prob_attack = xgb_model.predict_proba(z)[0][1]

    return pred, prob_attack



# ============================================================
# 2) PREDICT CHO NHIỀU DÒNG (CSV)
# ============================================================
def predict_batch(df_features, scaler, ae, xgb_model):
    """
    df_features: pandas DataFrame với đúng feature_cols đã scale
    return: preds (0/1) và probabilities attack
    """

    # Scale toàn bộ dataset
    scaled = scaler.transform(df_features)

    # Encode latent
    X = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        z = ae.encoder(X).cpu().numpy()

    # Dự đoán nhiều dòng
    preds = xgb_model.predict(z)
    probs = xgb_model.predict_proba(z)[:, 1]

    return preds, probs
