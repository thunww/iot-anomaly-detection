import pandas as pd
import numpy as np

def build_features(df):

    # -----------------------------------
    # 0. Rename columns
    # -----------------------------------
    df = df.rename(columns={
        "FC1_Read_Input_Register": "FC1",
        "FC2_Read_Discrete_Value": "FC2",
        "FC3_Read_Holding_Register": "FC3",
        "FC4_Read_Coil": "FC4"
    })

    # Strip spaces in time column
    df["time"] = df["time"].astype(str).str.strip()

    # -----------------------------------
    # 1. Build datetime
    # -----------------------------------
    df["date_time"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d-%b-%y %H:%M:%S",
        errors="coerce"
    )

    df = df.sort_values("date_time").reset_index(drop=True)

    # -----------------------------------
    # 2. Time features
    # -----------------------------------
    df["hour"]   = df["date_time"].dt.hour
    df["minute"] = df["date_time"].dt.minute
    df["second"] = df["date_time"].dt.second
    df["time_index"] = df["hour"] * 3600 + df["minute"] * 60 + df["second"]
    df["is_peak"] = df["hour"].between(8, 18).astype(int)

    # -----------------------------------
    # 3. Basic FC statistics
    # -----------------------------------
    df["fc_mean"]  = df[["FC1","FC2","FC3","FC4"]].mean(axis=1)
    df["fc_std"]   = df[["FC1","FC2","FC3","FC4"]].std(axis=1)
    df["fc_min"]   = df[["FC1","FC2","FC3","FC4"]].min(axis=1)
    df["fc_max"]   = df[["FC1","FC2","FC3","FC4"]].max(axis=1)
    df["fc_range"] = df["fc_max"] - df["fc_min"]

    # -----------------------------------
    # 4. Ratios
    # -----------------------------------
    df["fc_ratio_12"] = df["FC1"] / (df["FC2"] + 1)
    df["fc_ratio_34"] = df["FC3"] / (df["FC4"] + 1)

    # -----------------------------------
    # 5. Aggregations
    # -----------------------------------
    df["fc_sum"] = df["FC1"] + df["FC2"] + df["FC3"] + df["FC4"]
    df["fc_energy"] = df["FC1"]**2 + df["FC2"]**2 + df["FC3"]**2 + df["FC4"]**2

    # -----------------------------------
    # 6. Delta features
    # -----------------------------------
    df["delta_fc1"] = df["FC1"].diff().fillna(0)
    df["delta_fc2"] = df["FC2"].diff().fillna(0)
    df["delta_fc3"] = df["FC3"].diff().fillna(0)
    df["delta_fc4"] = df["FC4"].diff().fillna(0)
    df["delta_fc_sum"] = df["fc_sum"].diff().fillna(0)

    # -----------------------------------
    # 7. Lag + Rolling
    # -----------------------------------
    df["Lag1"] = df["fc_mean"].shift(1).bfill()
    df["Lag2"] = df["fc_mean"].shift(2).bfill()
    df["roll_mean"] = df["fc_mean"].rolling(5).mean().bfill()
    df["roll_std"]  = df["fc_std"].rolling(5).std().bfill()

    # -----------------------------------
    # 8. CLEAN NaN + INF
    # -----------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df
