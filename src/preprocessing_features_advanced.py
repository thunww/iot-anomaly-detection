import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

RAW_PATH = "data/raw/Train_Test_IoT_Modbus.csv"


# ============================================================
# 1) CREATE TIMESTAMP
# ============================================================
def add_timestamp(df):
    df["time"] = df["time"].astype(str).str.strip()

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d-%b-%y %H:%M:%S",
        errors="coerce"
    )

    df["timestamp"] = df["timestamp"].ffill()
    return df


# ============================================================
# 2) REQUEST RATE
# ============================================================
def add_request_rate(df):
    df["delta"] = df["timestamp"].diff().dt.total_seconds()
    df["delta"] = df["delta"].replace(0, 1e-6).fillna(1e-6)

    df["req_rate"] = 1 / df["delta"]
    df["req_rate"] = df["req_rate"].replace([np.inf, -np.inf], 0)

    return df


# ============================================================
# 3) TEMPORAL FEATURES
# ============================================================
def add_temporal_features(df, cols):
    for c in cols:

        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_lag2"] = df[c].shift(2)
        df[f"{c}_lag3"] = df[c].shift(3)

        df[f"{c}_diff"] = df[c].diff()

        df[f"{c}_mean3"] = df[c].rolling(3).mean()
        df[f"{c}_std3"] = df[c].rolling(3).std()
        df[f"{c}_max3"] = df[c].rolling(3).max()
        df[f"{c}_min3"] = df[c].rolling(3).min()

        df[f"{c}_mean5"] = df[c].rolling(5).mean()
        df[f"{c}_std5"] = df[c].rolling(5).std()

    return df.fillna(0)        # <-- BẮT BUỘC PHẢI CÓ!


# ============================================================
# 4) MAIN FEATURE ENGINEERING
# ============================================================
def create_features(df):

    numeric_cols = [
        "FC1_Read_Input_Register",
        "FC2_Read_Discrete_Value",
        "FC3_Read_Holding_Register",
        "FC4_Read_Coil"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().fillna(0)

    df = add_timestamp(df)
    df = add_request_rate(df)

    # RATIOS
    for c1 in numeric_cols:
        for c2 in numeric_cols:
            if c1 != c2:
                df[f"{c1}_{c2}_ratio"] = df[c1] / (df[c2] + 1e-6)

    # TEMP FEATURES
    temporal_cols = numeric_cols + ["req_rate"]
    df = add_temporal_features(df, temporal_cols)

    return df


# ============================================================
# 5) LOAD
# ============================================================
df = pd.read_csv(RAW_PATH)
df = create_features(df)

print("DEBUG RAW SHAPE:", df.shape)
print("DEBUG COL COUNT:", len(df.columns))

labels = df["label"].values
np.save("data/processed/train_labels.npy", labels[:15000])
np.save("data/processed/test_labels.npy", labels[15000:])

df = df.drop(columns=["date", "time", "type", "timestamp", "delta"])

train_df = df.iloc[:15000].copy()
test_df = df.iloc[15000:].copy()

# CLEAN
train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

train_df = train_df.clip(-1e6, 1e6)
test_df = test_df.clip(-1e6, 1e6)

feature_cols = train_df.drop(columns=["label"]).columns

scaler = MinMaxScaler().fit(train_df[feature_cols])
joblib.dump(scaler, "models/scaler.pkl")

train_scaled = scaler.transform(train_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

np.save("data/processed/train_features.npy", train_scaled)
np.save("data/processed/test_features.npy", test_scaled)

print("[INFO] Total features:", len(feature_cols))
print("[OK] Preprocessing done!")
