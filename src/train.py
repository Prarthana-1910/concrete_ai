import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Load processed dataset
df = pd.read_csv("data/features.csv")

FEATURE_COLUMNS = [
    "Cement_kg_m3", "Fly_Ash_kg_m3", "GGBS_kg_m3", "metakolin_kg_m3",
    "Water_kg_m3", "Sand_kg_m3", "AGE", "admixture", "Coarse aggregate",
    "SCMContent", "Binder", "WBRatio", "AggregateToBinder", "AdmixtureToBinder",
]
TARGET_COLUMN = "Compressive_Strength_MPa"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ── Train/val/test split ──────────────────────────────────────────────────────
# Hold out 15% as a clean test set never seen during tuning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
# A further 15% of training data for early-stopping validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

results = []

# ── BASE MODEL COMPARISON ─────────────────────────────────────────────────────
base_models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42, n_jobs=-1
    ),
    "CatBoost_Baseline": CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, random_state=42
    ),
}

print("\nBase Model Comparison")
print("-" * 60)

for name, model in base_models.items():
    model.fit(X_train, y_train)          # full train split (no val leakage)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    joblib.dump(model, f"models/{name}.joblib")
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name:<22} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

# ── STEP 1 — find best n_estimators via early stopping ────────────────────────
print("\n" + "-" * 60)
print("Step 1 — Early stopping to find optimal n_estimators...")
print("-" * 60)

es_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=2000,          # high ceiling — early stopping cuts it
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="mae",
)

es_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

best_n = es_model.best_iteration
print(f"\nOptimal n_estimators: {best_n}")

# ── STEP 2 — RandomizedSearchCV with best_n fixed ─────────────────────────────
print("\n" + "-" * 60)
print("Step 2 — Tuning remaining hyperparameters (n_iter=80, cv=5)...")
print("-" * 60)

param_dist = {
    "n_estimators":      [best_n],
    "max_depth":         [3, 4, 5, 6, 7, 8],
    "learning_rate":     [0.01, 0.03, 0.05, 0.07, 0.1],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7, 10],
    "gamma":             [0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha":         [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":        [0.5, 1.0, 1.5, 2.0, 5.0],
}

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=-1
    ),
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)   # full train split for CV

best_xgb   = random_search.best_estimator_
best_preds = best_xgb.predict(X_test)

best_mae  = mean_absolute_error(y_test, best_preds)
best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
best_r2   = r2_score(y_test, best_preds)

joblib.dump(best_xgb, "models/ConcreteAI_XGBoost_Best.joblib")

results.append({
    "Model": "XGBoost_Tuned", "MAE": best_mae, "RMSE": best_rmse, "R2": best_r2
})

print("\nTuned XGBoost Results")
print("-" * 60)
print(f"Best Parameters : {random_search.best_params_}")
print(f"MAE             : {best_mae:.4f}")
print(f"RMSE            : {best_rmse:.4f}")
print(f"R2              : {best_r2:.4f}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All Models — Final Comparison")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df.to_string(index=False))
