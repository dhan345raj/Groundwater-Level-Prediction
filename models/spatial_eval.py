"""
Spatial Block Cross-Validation & Baseline Benchmark Suite
Evaluates model generalization across UNSEEN monitoring stations (GroupKFold by MW_ID)
to prevent spatial data leakage and compare against standard hydrological baselines.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data_all", "processed_all.csv")

def evaluate_spatial_generalization():
    print("=" * 70)
    print("  SPATIAL GENERALIZATION BENCHMARK (UNSEEN STATIONS EVALUATION)")
    print("=" * 70)
    
    print(f"\n1. Loading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], dayfirst=True)
    df = df.sort_values(["MW_ID", "Date"]).reset_index(drop=True)
    
    # Compute physical delta
    df["gwl_delta"] = df["GWL"] - df["gwl_lag1"]
    df = df.dropna(subset=["gwl_delta"]).reset_index(drop=True)
    
    # Exclude leakage features
    exclude = {"GWL", "Date", "MW_ID", "GWL_flag", "gwl_delta", "gwl_lag1", "gwl_lag2"}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Top physical features
    top_features = [
        "ERA5_sro", "HYRAS_pr", "HYRAS_tasmax", "HYRAS_tas",
        "DWD_soil_moist", "DWD_soil_temp5cm", "rain_4sum",
        "HYRAS_hurs", "DWD_evapo_fao", "DWD_evapo_r"
    ]
    top_features = [f for f in top_features if f in df.columns]
    
    total_wells = df["MW_ID"].nunique()
    print(f"Total samples: {len(df):,} across {total_wells} unique monitoring wells.")
    print(f"Feature space: {top_features}")
    
    # 5-Fold Spatial Group Split (Entire wells are held out in validation folds)
    gkf = GroupKFold(n_splits=5)
    groups = df["MW_ID"]
    
    results = {
        "Naïve Persistence (Δ=0)": {"mae": [], "rmse": []},
        "Linear Hydrological Balance (Ridge)": {"mae": [], "rmse": []},
        "XGBoost Delta Regressor (Physics)": {"mae": [], "rmse": []}
    }
    
    print("\n2. Executing 5-Fold Spatial Cross-Validation (evaluating on strictly unseen stations)...")
    
    X = df[top_features].values
    y = df["gwl_delta"].values
    
    fold = 1
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        test_wells_count = df.iloc[test_idx]["MW_ID"].nunique()
        print(f"   Fold {fold}: Train wells={total_wells - test_wells_count}, Held-out test wells={test_wells_count}")
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        # 1. Baseline: Naïve Persistence (predicts no change, delta=0)
        y_pred_naive = np.zeros_like(y_te)
        results["Naïve Persistence (Δ=0)"]["mae"].append(mean_absolute_error(y_te, y_pred_naive))
        results["Naïve Persistence (Δ=0)"]["rmse"].append(mean_squared_error(y_te, y_pred_naive) ** 0.5)
        
        # 2. Baseline: Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        y_pred_ridge = ridge.predict(X_te)
        results["Linear Hydrological Balance (Ridge)"]["mae"].append(mean_absolute_error(y_te, y_pred_ridge))
        results["Linear Hydrological Balance (Ridge)"]["rmse"].append(mean_squared_error(y_te, y_pred_ridge) ** 0.5)
        
        # 3. XGBoost Physical Model
        xgb = XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        y_pred_xgb = xgb.predict(X_te)
        results["XGBoost Delta Regressor (Physics)"]["mae"].append(mean_absolute_error(y_te, y_pred_xgb))
        results["XGBoost Delta Regressor (Physics)"]["rmse"].append(mean_squared_error(y_te, y_pred_xgb) ** 0.5)
        
        fold += 1
        
    print("\n" + "=" * 70)
    print("  SPATIAL BENCHMARK RESULTS SUMMARY (5-FOLD GROUPKFOLD)")
    print("=" * 70)
    print(f"{'Model Architecture':<38} | {'MAE (m)':<14} | {'RMSE (m)':<14}")
    print("-" * 70)
    
    for name, metrics in results.items():
        avg_mae = np.mean(metrics["mae"])
        std_mae = np.std(metrics["mae"])
        avg_rmse = np.mean(metrics["rmse"])
        std_rmse = np.std(metrics["rmse"])
        print(f"{name:<38} | {avg_mae:.4f} ± {std_mae:.4f}  | {avg_rmse:.4f} ± {std_rmse:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_spatial_generalization()
