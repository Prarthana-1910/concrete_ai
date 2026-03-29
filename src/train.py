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

df = pd.read_csv("data/features.csv")

FEATURE_COLUMNS = [
    "Cement_kg_m3", "Fly_Ash_kg_m3", "GGBS_kg_m3", "metakolin_kg_m3",
    "Water_kg_m3", "Sand_kg_m3", "AGE", "admixture", "Coarse aggregate",
    "SCMContent", "Binder", "WBRatio", "AggregateToBinder", "AdmixtureToBinder",
]
TARGET_COLUMN = "Compressive_Strength_MPa"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
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
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    joblib.dump(model, f"models/{name}.joblib")
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name:<22} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

# ── RANDOM FOREST TUNING ──────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Tuning Random Forest...")
print("-" * 60)

rf_param_dist = {
    "n_estimators":      [300, 500, 700],
    "max_depth":         [None, 10, 15, 20, 25, 30],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf":  [1, 2, 4, 6],
    "max_features":      ["sqrt", "log2", 0.5, 0.7],
    "max_samples":       [0.7, 0.8, 0.9, None],
    "bootstrap":         [True],
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    verbose=3,
    random_state=42,
    n_jobs=-1,
)

rf_search.fit(X_train, y_train)

best_rf  = rf_search.best_estimator_
rf_preds = best_rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2   = r2_score(y_test, rf_preds)

joblib.dump(best_rf, "models/ConcreteAI_RF_Best.joblib")
results.append({
    "Model": "RandomForest_Tuned", "MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2
})

print("\nTuned Random Forest Results")
print("-" * 60)
print(f"Best Parameters : {rf_search.best_params_}")
print(f"MAE             : {rf_mae:.4f}")
print(f"RMSE            : {rf_rmse:.4f}")
print(f"R2              : {rf_r2:.4f}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All Models — Final Comparison")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df.to_string(index=False))
