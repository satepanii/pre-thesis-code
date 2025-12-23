# ======================================================
# Multiclass Diabetes Risk Prediction (BASELINE)
# Normal (0) - Prediabetes (1) - Diabetes (2)
# Random Forest WITHOUT SMOTE
# ======================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

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
# 5. DROP ID COLUMNS
# ======================================================
id_cols = [c for c in df.columns if "ID" in c or "NO" in c]
df = df.drop(columns=id_cols, errors="ignore")

# ======================================================
# 6. SPLIT FEATURES & TARGET
#    (HbA1c dropped → risk analysis)
# ======================================================
X = df.drop(["CLASS", "HBA1C"], axis=1)
y = df["CLASS"]

print("\nTraining features:")
print(X.columns.tolist())

# ======================================================
# 7. TRAIN / TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain set:", X_train.shape)
print("Test set :", X_test.shape)

# ======================================================
# 8. PIPELINE (NO SMOTE)
# ======================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"  # penting karena tidak pakai SMOTE
    ))
])

# ======================================================
# 9. STRATIFIED K-FOLD CROSS VALIDATION
# ======================================================
print("\n=== STRATIFIED K-FOLD CROSS VALIDATION (NO SMOTE) ===")

skf = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

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
# 11. EVALUATION ON TEST SET
# ======================================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print("\n=== TEST SET PERFORMANCE (NO SMOTE) ===")

print(f"Accuracy        : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"F1-score (Wght) : {f1_score(y_test, y_pred, average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Prediabetes", "Diabetes"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(
    y_test_bin,
    y_prob,
    average="macro",
    multi_class="ovr"
)

print(f"ROC AUC (OvR)   : {roc_auc:.3f}")

# ======================================================
# 12. SAVE MODEL
# ======================================================
MODEL_PATH = "rf_diabetes_multiclass_no_smote.joblib"
joblib.dump(pipeline, MODEL_PATH)

print(f"\n✅ Model saved as {MODEL_PATH}")
