import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/FINAL_PROJECT_DATASET.csv")


print("DATASET OVERVIEW")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print("\nCompressive Strength Summary:")
print(df["Compressive_Strength_MPa"].describe())


# EDA - OUTLIER BOXPLOT
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Compressive_Strength_MPa"])
plt.title("Outlier Detection for Compressive Strength")
plt.tight_layout()
plt.show()


# EDA - CORRELATION HEATMAP
correlation_columns = [
    "Cement_kg_m3",
    "Fly_Ash_kg_m3",
    "GGBS_kg_m3",
    "metakolin_kg_m3",
    "TCM",
    "Water_kg_m3",
    "water/TCM",
    "Sand_kg_m3",
    "AGE",
    "admixture",
    "Coarse aggregate",
    "Compressive_Strength_MPa"
]

corr_matrix = df[correlation_columns].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# FEATURE ENGINEERING
df2 = df.copy()

# Core binder / SCM features
df2["SCMContent"] = (
    df2["Fly_Ash_kg_m3"] +
    df2["GGBS_kg_m3"] +
    df2["metakolin_kg_m3"]
)

df2["Binder"] = (
    df2["Cement_kg_m3"] +
    df2["SCMContent"]
)

# Important ratios
df2["WBRatio"] = df2["Water_kg_m3"] / df2["Binder"]
df2["AggregateToBinder"] = ( df2["Sand_kg_m3"] + df2["Coarse aggregate"]) / df2["Binder"]
df2["AdmixtureToBinder"] = df2["admixture"] / df2["Binder"]


# Clean infinite / NaN values
df2 = df2.replace([np.inf, -np.inf], np.nan)
df2 = df2.fillna(0)
df2.drop(columns=["Density"], inplace=True)
# Save processed dataset
df2.to_csv("data/features.csv", index=False)
print("FEATURE ENGINEERING COMPLETED")
