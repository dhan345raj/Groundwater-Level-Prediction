import pandas as pd
import numpy as np
import os

MERGED_FILE = r".\data_all\all_dynamic.csv"
OUT_FILE = r".\data_all\processed_all.csv"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

df = pd.read_csv(MERGED_FILE, parse_dates=["Date"])

df = df.sort_values(["MW_ID", "Date"]).reset_index(drop=True)

# Drop rows without target
df = df.dropna(subset=["GWL"])

# Time features
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year
df["dayofyear"] = df["Date"].dt.dayofyear

# Lag features
df["gwl_lag1"] = df.groupby("MW_ID")["GWL"].shift(1)
df["gwl_lag2"] = df.groupby("MW_ID")["GWL"].shift(2)

# Rolling rainfall (if available)
if "HYRAS_pr" in df.columns:
    df["rain_4sum"] = df.groupby("MW_ID")["HYRAS_pr"].rolling(
        window=4, min_periods=1
    ).sum().reset_index(level=0, drop=True)

# Drop first rows per well where lag is NaN
df = df.dropna(subset=["gwl_lag1"]).reset_index(drop=True)

# Just fill numeric NaNs with median – BUT NO SCALING HERE
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

df.to_csv(OUT_FILE, index=False)
print("Saved processed data WITHOUT scaling to:", OUT_FILE)
