# ======================================================
# Multiclass Diabetes Risk Prediction
# Normal (0) - Prediabetes (1) - Diabetes (2)
# Random Forest
# ======================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
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
# Gender
df["GENDER"] = (
    df["GENDER"]
    .astype(str)
    .str.upper()
    .map({"M": 1, "F": 0})
)

# Target (MULTICLASS)
df["CLASS"] = (
    df["CLASS"]
    .astype(str)
    .str.upper()
    .map({
        "N": 0,  # Normal
        "P": 1,  # Prediabetes
        "Y": 2   # Diabetes
    })
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
# 5. DROP ID COLUMNS (IF ANY)
# ======================================================
id_cols = [c for c in df.columns if "ID" in c or "NO" in c]
df = df.drop(columns=id_cols, errors="ignore")

# ======================================================
# 6. SPLIT FEATURES & TARGET
# ======================================================
X = df.drop("CLASS", axis=1)
y = df["CLASS"]

print("\nTraining features:")
print(X.columns.tolist())

# SAFETY CHECK (ANTI DATA LEAKAGE)
assert "CLASS" not in X.columns, "❌ ERROR: CLASS is leaking into features!"

# ======================================================
# 7. TRAIN TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ======================================================
# 8. MODEL PIPELINE
# ======================================================
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# ======================================================
# 9. MODEL PERFORMANCE (MULTICLASS)
# ======================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\n=== MODEL PERFORMANCE (MULTICLASS) ===")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy        : {acc:.3f}")

# F1 Scores
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"F1-score (Macro): {f1_macro:.3f}")
print(f"F1-score (Wght) : {f1_weighted:.3f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Prediabetes", "Diabetes"]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Multiclass ROC AUC (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(
    y_test_bin,
    y_prob,
    average="macro",
    multi_class="ovr"
)

print(f"ROC AUC (OvR)   : {roc_auc:.3f}")

# ======================================================
# 10. SAVE MODEL
# ======================================================
MODEL_PATH = "rf_diabetes_multiclass.joblib"
joblib.dump(model, MODEL_PATH)

print(f"\n✅ Model saved as {MODEL_PATH}")
