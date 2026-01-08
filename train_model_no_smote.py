# ======================================================
# Multiclass Diabetes Risk Prediction (BASELINE)
# Normal (0) - Prediabetes (1) - Diabetes (2)
# Random Forest WITHOUT SMOTE
# + EDA Visualizations
# + Age Group Analysis (WITH NUMBERS)
# + Feature Correlation Heatmap (WITH VALUES)
# + Confusion Matrix Heatmap
# ======================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ======================================================
# 1. LOAD DATASET
# ======================================================
DATA_PATH = "data/diabetes.csv"
df = pd.read_csv(DATA_PATH)

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
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# EDA 2: FEATURE DISTRIBUTIONS
# ------------------------------------------------------
numeric_features = [
    "AGE", "BMI", "UREA", "CR",
    "CHOL", "TG", "HDL", "LDL", "VLDL"
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
# EDA 3: BMI vs CLASS
# ------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="CLASS", y="BMI", data=df, palette="Set3")
plt.xticks([0, 1, 2], ["Normal", "Prediabetes", "Diabetes"])
plt.title("BMI Distribution by Class")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# EDA 4: AGE GROUP vs DIABETES STATUS (WITH NUMBERS)
# ------------------------------------------------------
age_bins = [0, 30, 40, 50, 60, 120]
age_labels = ["<30", "30–39", "40–49", "50–59", "60+"]

df["AGE_GROUP"] = pd.cut(
    df["AGE"],
    bins=age_bins,
    labels=age_labels,
    right=False
)

age_class_table = pd.crosstab(df["AGE_GROUP"], df["CLASS"])
age_class_table.columns = ["Normal", "Prediabetes", "Diabetes"]

print("\nDiabetes Distribution by Age Group:")
print(age_class_table)

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="AGE_GROUP",
    hue="CLASS",
    palette="Set2"
)
plt.legend(title="Class", labels=["Normal", "Prediabetes", "Diabetes"])
plt.title("Diabetes Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
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

plt.title("Feature Correlation Heatmap (Risk Factors)")
plt.tight_layout()
plt.show()

# ======================================================
# 7. SPLIT FEATURES & TARGET
# ======================================================
X = df.drop(["CLASS", "HBA1C"], axis=1)
y = df["CLASS"]

print("\nTraining features:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# 8. PIPELINE (NO SMOTE - BASELINE)
# ======================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# ======================================================
# 9. STRATIFIED K-FOLD CROSS VALIDATION
# ======================================================
print("\n=== STRATIFIED K-FOLD CROSS VALIDATION (NO SMOTE) ===")

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

cv_f1_macro = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring="f1_macro"
)

print(f"CV F1 Macro (mean): {cv_f1_macro.mean():.3f}")
print(f"CV F1 Macro (std) : {cv_f1_macro.std():.3f}")

# ======================================================
# 10. TRAIN FINAL MODEL
# ======================================================
pipeline.fit(X_train, y_train)

# ======================================================
# 11. FINAL EVALUATION
# ======================================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print("\n=== TEST SET PERFORMANCE (BASELINE / NO SMOTE) ===")
print(f"Accuracy        : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"F1-score (Wght) : {f1_score(y_test, y_pred, average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Prediabetes", "Diabetes"]
))

# ======================================================
# 12. CONFUSION MATRIX HEATMAP
# ======================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Prediabetes", "Diabetes"],
    yticklabels=["Normal", "Prediabetes", "Diabetes"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap (Baseline / No SMOTE)")
plt.tight_layout()
plt.show()

# ======================================================
# 13. ROC AUC (OvR)
# ======================================================
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(
    y_test_bin,
    y_prob,
    average="macro",
    multi_class="ovr"
)

print(f"ROC AUC (OvR): {roc_auc:.3f}")

# ======================================================
# 14. SAVE MODEL
# ======================================================
MODEL_PATH = "rf_diabetes_multiclass_no_smote.joblib"
joblib.dump(pipeline, MODEL_PATH)

print(f"\n✅ Model saved as {MODEL_PATH}")
