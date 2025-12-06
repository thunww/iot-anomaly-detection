import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

RAW_PATH = "data/raw/Train_Test_IoT_Modbus.csv"

def create_features(df):
    numeric_cols = [
        "FC1_Read_Input_Register",
        "FC2_Read_Discrete_Value",
        "FC3_Read_Holding_Register",
        "FC4_Read_Coil"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().fillna(0)

    # 1. diff
    for col in numeric_cols:
        df[col + "_diff"] = df[col].diff().fillna(0)

    # 2. ratios
    for c1 in numeric_cols:
        for c2 in numeric_cols:
            if c1 != c2:
                df[c1 + "_" + c2 + "_ratio"] = df[c1] / (df[c2] + 1e-6)

    # 3. rolling stats
    windows = [3, 5, 10]
    for w in windows:
        for col in numeric_cols:
            df[f"{col}_mean_w{w}"] = df[col].rolling(w).mean().bfill()
            df[f"{col}_std_w{w}"] = df[col].rolling(w).std().bfill()
            df[f"{col}_max_w{w}"] = df[col].rolling(w).max().bfill()
            df[f"{col}_min_w{w}"] = df[col].rolling(w).min().bfill()

    return df


df = pd.read_csv(RAW_PATH)
df = create_features(df)

labels = df["label"].values
np.save("data/processed/train_labels.npy", labels[:15000])
np.save("data/processed/test_labels.npy", labels[15000:])

df = df.drop(columns=["date", "time", "type"])

train_df = df.iloc[:15000]
test_df  = df.iloc[15000:]

feature_cols = train_df.drop(columns=["label"]).columns

scaler = MinMaxScaler().fit(train_df[feature_cols])
joblib.dump(scaler, "models/scaler.pkl")

train_scaled = scaler.transform(train_df[feature_cols])
test_scaled  = scaler.transform(test_df[feature_cols])

np.save("data/processed/train_features.npy", train_scaled)
np.save("data/processed/test_features.npy", test_scaled)

print("[INFO] Total features:", len(feature_cols))
print("[OK] Preprocessing done!")
