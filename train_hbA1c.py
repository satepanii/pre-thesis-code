# ======================================================
# Multiclass Diabetes Risk Prediction
# TOP 5 FEATURES (Highest Correlation to CLASS)
# Random Forest + SMOTE
# ======================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ======================================================
# 1. LOAD DATASET
# ======================================================
df = pd.read_csv("data/diabetes.csv")
print("Dataset loaded:", df.shape)

# ======================================================
# 2. NORMALIZE COLUMN NAMES
# ======================================================
df.columns = (
    df.columns.str.strip()
    .str.upper()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# ======================================================
# 3. ENCODE VARIABLES
# ======================================================
df["GENDER"] = df["GENDER"].astype(str).str.upper().map({"M": 1, "F": 0})
df["CLASS"]  = df["CLASS"].astype(str).str.upper().map({"N": 0, "P": 1, "Y": 2})

df = df.dropna(subset=["CLASS"])
df["CLASS"] = df["CLASS"].astype(int)

# ======================================================
# 4. NUMERIC CONVERSION
# ======================================================
for col in df.columns:
    if col not in ["CLASS", "GENDER"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ======================================================
# 5. DROP ID COLUMNS
# ======================================================
df = df.drop(columns=[c for c in df.columns if "ID" in c or "NO" in c], errors="ignore")

# ======================================================
# 6. NUMERIC FEATURES
# ======================================================
numeric_features = [
    "AGE","BMI","UREA","CR","CHOL","TG","HDL","LDL","VLDL","HBA1C"
]

# ======================================================
# 7. SELECT TOP 5 FEATURES BY CORRELATION TO CLASS
# ======================================================
corr_matrix = df[numeric_features + ["CLASS"]].corr()

corr_to_class = (
    corr_matrix["CLASS"]
    .drop("CLASS")
    .abs()
    .sort_values(ascending=False)
)

top_5_features = corr_to_class.head(5).index.tolist()

print("\n=== TOP 5 FEATURES BY CORRELATION TO CLASS ===")
for i, feat in enumerate(top_5_features, start=1):
    print(f"{i}. {feat} | Corr = {corr_to_class[feat]:.3f}")

# ======================================================
# =================== MACHINE LEARNING =================
# ======================================================
X = df[top_5_features]
y = df["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

# ======================================================
# 8. STRATIFIED K-FOLD
# ======================================================
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

cv_acc, cv_f1 = [], []

print("\n=== STRATIFIED K-FOLD RESULTS (TOP 5 FEATURES) ===")
for fold, (tr, val) in enumerate(skf.split(X_train, y_train), start=1):
    pipeline.fit(X_train.iloc[tr], y_train.iloc[tr])
    y_val_pred = pipeline.predict(X_train.iloc[val])

    acc = accuracy_score(y_train.iloc[val], y_val_pred)
    f1  = f1_score(y_train.iloc[val], y_val_pred, average="macro")

    cv_acc.append(acc)
    cv_f1.append(f1)

    print(f"Fold {fold} | Accuracy: {acc:.3f} | F1 Macro: {f1:.3f}")

print("\n=== CV SUMMARY ===")
print(f"Mean Accuracy : {np.mean(cv_acc):.3f}")
print(f"Mean F1 Macro : {np.mean(cv_f1):.3f}")

# ======================================================
# 9. FINAL TRAINING
# ======================================================
pipeline.fit(X_train, y_train)

# ======================================================
# 10. TEST EVALUATION
# ======================================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print("\n=== TEST SET PERFORMANCE (TOP 5 FEATURES) ===")
print(f"Accuracy        : {accuracy_score(y_test,y_pred):.3f}")
print(f"F1 Macro        : {f1_score(y_test,y_pred,average='macro'):.3f}")
print(f"F1 Weighted     : {f1_score(y_test,y_pred,average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal","Prediabetes","Diabetes"]
))

# ======================================================
# 11. CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal","Prediabetes","Diabetes"],
    yticklabels=["Normal","Prediabetes","Diabetes"]
)
plt.title("Confusion Matrix (Top 5 Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ======================================================
# 12. ROC AUC
# ======================================================
y_test_bin = label_binarize(y_test, classes=[0,1,2])
roc_auc = roc_auc_score(
    y_test_bin, y_prob,
    average="macro",
    multi_class="ovr"
)

print(f"ROC AUC (OvR): {roc_auc:.3f}")

# ======================================================
# 13. SAVE MODEL
# ======================================================
joblib.dump(
    pipeline,
    "rf_diabetes_top5_features_smote.joblib"
)

print("\n✅ MODEL (TOP 5 FEATURES) SAVED SUCCESSFULLY")
