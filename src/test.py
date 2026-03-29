import warnings
import joblib
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Load processed dataset
df = pd.read_csv("data/features.csv")

# Input features
FEATURE_COLUMNS = [
    "Cement_kg_m3",
    "Fly_Ash_kg_m3",
    "GGBS_kg_m3",
    "metakolin_kg_m3",
    "Water_kg_m3",
    "Sand_kg_m3",
    "AGE",
    "admixture",
    "Coarse aggregate",
    "SCMContent",
    "Binder",
    "WBRatio",
    "AggregateToBinder",
    "AdmixtureToBinder",
]

# Target
TARGET_COLUMN = "Compressive_Strength_MPa"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Models to evaluate
models = {
    "RandomForest": "models/RandomForest.joblib",
    "XGBoost": "models/XGBoost.joblib",
    "CatBoost_Baseline": "models/CatBoost_Baseline.joblib",
    "RandomForest_Tuned": "models/ConcreteAI_RF_Best.joblib",
}

table = PrettyTable()
table.field_names = ["Model", "MAE", "RMSE", "R2"]

results = []

print("\nModel Evaluation Results")
print("-" * 60)

for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    table.add_row([name, round(mae, 4), round(rmse, 4), round(r2, 4)])
    residuals = y_pred - y_test
    print("Mean residual:", residuals.mean())
    print("Median residual:", np.median(residuals))

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("Predicted vs Actual")
plt.show()

print(table)

# Best model summary
results_df = pd.DataFrame(results).sort_values(by=["MAE", "RMSE"], ascending=[True, True])
best_row = results_df.iloc[0]


print("TEST SUMMARY")
print("=" * 60)
print(f"Best Model on Test Set : {best_row['Model']}")
print(f"Best MAE               : {best_row['MAE']:.4f}")
print(f"Best RMSE              : {best_row['RMSE']:.4f}")
print(f"Best R2                : {best_row['R2']:.4f}")
print("-" * 60)
