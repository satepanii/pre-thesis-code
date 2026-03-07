# ======================================================
# Multiclass Diabetes Risk Prediction
# TOP 5 FEATURES (Highest Correlation)
# WITH SMOTE DATA COUNT
# ======================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ======================================================
# 1. LOAD DATASET
# ======================================================
df = pd.read_csv("data/diabetes.csv")
print("Dataset loaded:", df.shape)

df.columns = (
    df.columns.str.strip()
    .str.upper()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

df["GENDER"] = df["GENDER"].astype(str).str.upper().map({"M":1,"F":0})
df["CLASS"] = df["CLASS"].astype(str).str.upper().map({"N":0,"P":1,"Y":2})

df = df.dropna(subset=["CLASS"])
df["CLASS"] = df["CLASS"].astype(int)

print("\nClass distribution (Full Dataset):")
print(df["CLASS"].value_counts().sort_index())


# ======================================================
# 2. NUMERIC CONVERSION
# ======================================================
for col in df.columns:
    if col not in ["CLASS", "GENDER"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop(columns=[c for c in df.columns if "ID" in c or "NO" in c], errors="ignore")


# ======================================================
# 3. SELECT TOP 5 FEATURES
# ======================================================
numeric_features = [
    "AGE","BMI","UREA","CR","CHOL",
    "TG","HDL","LDL","VLDL","HBA1C"
]

corr = df[numeric_features + ["CLASS"]].corr()
top5 = (
    corr["CLASS"]
    .drop("CLASS")
    .abs()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)

print("\nTop 5 Features:", top5)

X = df[top5]
y = df["CLASS"]


# ======================================================
# 4. TRAIN TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n=== DATA BEFORE SMOTE ===")
print("Training shape:", X_train.shape)
print(y_train.value_counts().sort_index())


# ======================================================
# 5. SMOTE ANALYSIS
# ======================================================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\n=== DATA AFTER SMOTE ===")
print("Resampled shape:", X_train_sm.shape)
print(pd.Series(y_train_sm).value_counts().sort_index())


# ======================================================
# 6. PIPELINE
# ======================================================
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
# 7. CROSS VALIDATION
# ======================================================
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

cv_f1 = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring="f1_macro"
)

print("\nCV F1 Macro Mean:", round(cv_f1.mean(),3))
print("CV F1 Macro Std :", round(cv_f1.std(),3))


# ======================================================
# 8. TRAIN FINAL MODEL
# ======================================================
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print("\n=== TEST PERFORMANCE (TOP 5) ===")
print("Accuracy:", round(accuracy_score(y_test,y_pred),3))
print("F1 Macro:", round(f1_score(y_test,y_pred,average='macro'),3))

print("\nClassification Report:")
print(classification_report(
    y_test,y_pred,
    target_names=["Normal","Prediabetes","Diabetes"]
))

y_test_bin = label_binarize(y_test, classes=[0,1,2])
roc_auc = roc_auc_score(y_test_bin,y_prob,average="macro",multi_class="ovr")
print("ROC AUC:", round(roc_auc,3))

joblib.dump(pipeline,"rf_diabetes_top5.joblib")
print("\n✅ Model saved: rf_diabetes_top5.joblib")