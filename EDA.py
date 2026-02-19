# ======================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Multiclass Diabetes Dataset
# ======================================================
# + NA Check
# + Summary Statistics
# + Class Distribution
# + HbA1c Quartile Analysis
# + Normality Check
# + Correlation Matrix (Including CLASS)
# ======================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set(style="whitegrid")

# ======================================================
# 1. LOAD DATASET
# ======================================================
df = pd.read_csv("data/diabetes.csv")
print("Dataset loaded:", df.shape)

# ======================================================
# 2. NORMALIZE COLUMN NAMES
# ======================================================
df.columns = (
    df.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# ======================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ======================================================
df["GENDER"] = (
    df["GENDER"]
    .astype(str)
    .str.upper()
    .map({"M": 1, "F": 0})
)

df["CLASS"] = (
    df["CLASS"]
    .astype(str)
    .str.upper()
    .map({"N": 0, "P": 1, "Y": 2})
)

df = df.dropna(subset=["CLASS"])
df["CLASS"] = df["CLASS"].astype(int)

print("\nClass distribution:")
print(df["CLASS"].value_counts().sort_index())

# ======================================================
# 4. NUMERIC CONVERSION
# ======================================================
non_numeric = ["CLASS", "GENDER"]

for col in df.columns:
    if col not in non_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ======================================================
# 5. DROP ID COLUMNS
# ======================================================
id_cols = [c for c in df.columns if "ID" in c or "NO" in c]
df = df.drop(columns=id_cols, errors="ignore")
# ======================================================
# ======================= EDA ==========================
# ======================================================

# ------------------------------------------------------
# EDA 1: CLASS DISTRIBUTION
# ------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="CLASS", data=df, palette="Set2")
plt.xticks([0, 1, 2], ["Normal", "Prediabetes", "Diabetes"])
plt.title("Class Distribution")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# EDA 2: FEATURE DISTRIBUTIONS
# ------------------------------------------------------
numeric_features = [
    col for col in df.columns
    if col not in ["CLASS", "GENDER"]
]

df[numeric_features].hist(
    bins=30,
    figsize=(14, 10),
    edgecolor="black"
)

plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# EDA 3: BOXPLOT BMI vs CLASS
# ------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="CLASS", y="BMI", data=df, palette="Set3")
plt.xticks([0, 1, 2], ["Normal", "Prediabetes", "Diabetes"])
plt.title("BMI Distribution by Class")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# EDA 4: AGE GROUP vs DIABETES STATUS
# ------------------------------------------------------
age_bins = [0, 30, 40, 50, 60, 120]
age_labels = ["<30", "30-39", "40-49", "50-59", "60+"]

df["AGE_GROUP"] = pd.cut(df["AGE"], bins=age_bins, labels=age_labels, right=False)

age_table = pd.crosstab(df["AGE_GROUP"], df["CLASS"])
age_table.columns = ["Normal", "Prediabetes", "Diabetes"]

print("\nDiabetes Distribution by Age Group:")
print(age_table)

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="AGE_GROUP", hue="CLASS", palette="Set2")

for container in ax.containers:
    ax.bar_label(container, fmt="%d", fontsize=9)

plt.legend(title="Class", labels=["Normal", "Prediabetes", "Diabetes"])
plt.title("Diabetes Distribution by Age Group")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# EDA 5: AUTO ANALYSIS FOR ALL NUMERIC FEATURES
# ------------------------------------------------------
print("\n===== AUTO FEATURE-BASED EDA (ALL FEATURES) =====")

for feature in numeric_features:
    if feature in ["AGE"]:
        continue

    print(f"\n--- {feature} vs CLASS ---")

    if df[feature].isna().all():
        print("Skipped (all NaN)")
        continue

    try:
        df[f"{feature}_GROUP"] = pd.qcut(
            df[feature],
            q=4,
            duplicates="drop"
        )
    except ValueError:
        print("Skipped (not enough unique values)")
        continue

    table = pd.crosstab(df[f"{feature}_GROUP"], df["CLASS"])
    table.columns = ["Normal", "Prediabetes", "Diabetes"]
    print(table)

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x=f"{feature}_GROUP",
        hue="CLASS",
        palette="Set2"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8)

    plt.title(f"Diabetes Distribution by {feature}")
    plt.xlabel(f"{feature} Group (Quartiles)")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# ======================================================
# 6. FEATURE CORRELATION HEATMAP (WITH VALUES)
# ======================================================
plt.figure(figsize=(13, 11))

corr = df.drop(["CLASS", "HBA1C"], axis=1).corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Feature Correlation Heatmap (Risk Factors)", fontsize=14)
plt.tight_layout()
plt.show()