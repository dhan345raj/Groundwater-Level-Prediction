import pandas as pd
import glob
import os
from tqdm import tqdm

DYNAMIC_DIR = r".\data\dynamic"
OUT_FILE = r".\data_all\all_dynamic.csv"



os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

files = sorted(glob.glob(os.path.join(DYNAMIC_DIR, "MW_*.csv")))

if not files:
    raise SystemExit("No MW_*.csv files found in dynamic folder!")

rows = []
for f in tqdm(files, desc="Merging MW files"):
    df = pd.read_csv(f)

    # Fix date column
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    mwid = os.path.splitext(os.path.basename(f))[0]  # MW_94
    df["MW_ID"] = mwid

    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
all_df = all_df.sort_values(["MW_ID", "Date"])

all_df.to_csv(OUT_FILE, index=False)
print(f"Saved merged file: {OUT_FILE}")
