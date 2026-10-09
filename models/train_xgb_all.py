import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# ----------------- PATH CONFIGURATION -----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data_all", "processed_all.csv")
MODEL_OUT = os.path.join(BASE_DIR, "models", "models", "xgb_model.pkl")
TOP_FEATURE_FILE = os.path.join(BASE_DIR, "models", "models", "top10_features.txt")
SCALER_OUT = os.path.join(BASE_DIR, "models", "models", "scaler.pkl")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

print("Loading dataset from:", DATA_FILE)
df = pd.read_csv(DATA_FILE, parse_dates=["Date"], dayfirst=True)
df = df.sort_values(["MW_ID", "Date"]).reset_index(drop=True)

# ----------------- TARGET FORMULATION -----------------
# Delta = Today's GWL - Yesterday's GWL (aquifer recharge / drawdown)
df["gwl_delta"] = df["GWL"] - df["gwl_lag1"]
df = df.dropna(subset=["gwl_delta"]).reset_index(drop=True)

# Exclude non-meteorological shortcuts and leakage columns
exclude = {"GWL", "Date", "MW_ID", "GWL_flag", "gwl_delta", "gwl_lag1", "gwl_lag2"}
feature_cols = [c for c in df.columns if c not in exclude]

print(f"\nTraining on {len(df):,} records across {df['MW_ID'].nunique()} monitoring wells.")
print("Candidate physical features:", feature_cols)

# ----------------- TEMPORAL TRAIN/TEST SPLIT -----------------
cutoff = df["Date"].quantile(0.8)
train = df[df["Date"] <= cutoff]
test = df[df["Date"] > cutoff]

print(f"Train split: {len(train):,} records (up to {cutoff.strftime('%Y-%m-%d')})")
print(f"Test split:  {len(test):,} records (after {cutoff.strftime('%Y-%m-%d')})")

X_train_full = train[feature_cols]
y_train_full = train["gwl_delta"]
X_test_full = test[feature_cols]
y_test_full = test["gwl_delta"]

# ----------------- STEP 1: FEATURE SELECTION ON PHYSICAL DRIVERS -----------------
print("\n--- Phase 1: Evaluating Physical Feature Importances ---")
model_selector = XGBRegressor(
    n_estimators=250,
    learning_rate=0.06,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
model_selector.fit(X_train_full, y_train_full)

pred_test_all = model_selector.predict(X_test_full)
print("Initial Delta Test MAE:", mean_absolute_error(y_test_full, pred_test_all))

# Compute importances
importance = model_selector.feature_importances_
feat_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
feat_df = feat_df.sort_values("importance", ascending=False).reset_index(drop=True)

top10 = feat_df.head(10)
top10_features = top10["feature"].tolist()
top10.to_csv(TOP_FEATURE_FILE, index=False)

print("\nTop 10 Hydrological Drivers:")
for idx, row in top10.iterrows():
    print(f"  {row['feature']:<18} : {row['importance']*100:6.2f}%")

# ----------------- STEP 2: TRAIN PRODUCTION MODEL ON TOP FEATURES -----------------
print("\n--- Phase 2: Training Regularized Production Model ---")
scaler = StandardScaler()
X_train_top = scaler.fit_transform(train[top10_features])
X_test_top = scaler.transform(test[top10_features])

final_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=10,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_top, y_train_full)

pred_train_delta = final_model.predict(X_train_top)
pred_test_delta = final_model.predict(X_test_top)

print("\n==== DELTA (DAILY FLUCTUATION) METRICS ====")
print("Train MAE:", mean_absolute_error(y_train_full, pred_train_delta))
print("Test  MAE:", mean_absolute_error(y_test_full, pred_test_delta))
print("Test  RMSE:", mean_squared_error(y_test_full, pred_test_delta)**0.5)
print("Test  R2:  ", r2_score(y_test_full, pred_test_delta))

# ----------------- STEP 3: RECONSTRUCTED GWL EVALUATION -----------------
# Predicted GWL = Previous Day GWL + Predicted Delta
pred_test_gwl = test["gwl_lag1"] + pred_test_delta
y_test_gwl = test["GWL"]

print("\n==== RECONSTRUCTED ABSOLUTE GWL METRICS ====")
print("Reconstructed MAE: ", mean_absolute_error(y_test_gwl, pred_test_gwl))
print("Reconstructed RMSE:", mean_squared_error(y_test_gwl, pred_test_gwl)**0.5)
print("Reconstructed R2:  ", r2_score(y_test_gwl, pred_test_gwl))

# ----------------- SAVE ASSETS -----------------
joblib.dump(final_model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"\nProduction model saved to: {MODEL_OUT}")
print(f"Feature scaler saved to:   {SCALER_OUT}")
print(f"Top 10 features saved to:  {TOP_FEATURE_FILE}")
