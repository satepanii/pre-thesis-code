# ======================================================
# Multiclass Diabetes Risk Prediction (BASELINE)
# Normal (0) - Prediabetes (1) - Diabetes (2)
# Random Forest WITHOUT SMOTE
# + FULL EDA (WITH COUNTS)
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
# 3. ENCODE CATEGORICAL VARIABLES
# ======================================================
df["GENDER"] = df["GENDER"].astype(str).str.upper().map({"M": 1, "F": 0})
df["CLASS"] = df["CLASS"].astype(str).str.upper().map({"N": 0, "P": 1, "Y": 2})

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
df = df.drop(columns=[c for c in df.columns if "ID" in c or "NO" in c], errors="ignore")

# ======================================================
# ========================= EDA ========================
# ======================================================

# ---------------- CLASS DISTRIBUTION ------------------
plt.figure(figsize=(6,4))
ax = sns.countplot(x="CLASS", data=df, palette="Set2")
plt.xticks([0,1,2], ["Normal","Prediabetes","Diabetes"])
plt.title("Class Distribution")
for c in ax.containers:
    ax.bar_label(c)
plt.tight_layout()
plt.show()

# ---------------- NUMERIC DISTRIBUTIONS ---------------
numeric_features = ["AGE","BMI","UREA","CR","CHOL","TG","HDL","LDL","VLDL"]

df[numeric_features].hist(
    bins=30,
    figsize=(14,10),
    edgecolor="black"
)
plt.suptitle("Numeric Feature Distributions")
plt.tight_layout()
plt.show()

# ---------------- AGE GROUP ANALYSIS ------------------
age_bins = [0,30,40,50,60,120]
age_labels = ["<30","30–39","40–49","50–59","60+"]

df["AGE_GROUP"] = pd.cut(df["AGE"], bins=age_bins, labels=age_labels, right=False)

age_table = pd.crosstab(df["AGE_GROUP"], df["CLASS"])
age_table.columns = ["Normal","Prediabetes","Diabetes"]

print("\nDiabetes Distribution by Age Group:")
print(age_table)

plt.figure(figsize=(9,5))
ax = sns.countplot(data=df, x="AGE_GROUP", hue="CLASS", palette="Set2")
plt.legend(title="Class", labels=["Normal","Prediabetes","Diabetes"])
for c in ax.containers:
    ax.bar_label(c)
plt.title("Diabetes Distribution by Age Group")
plt.tight_layout()
plt.show()

# ---------------- BMI GROUP ANALYSIS ------------------
bmi_bins = [0,18.5,25,30,100]
bmi_labels = ["Underweight","Normal","Overweight","Obese"]

df["BMI_GROUP"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels)

bmi_table = pd.crosstab(df["BMI_GROUP"], df["CLASS"])
bmi_table.columns = ["Normal","Prediabetes","Diabetes"]

print("\nDiabetes Distribution by BMI Group:")
print(bmi_table)

# ---------------- AUTO BINNING ALL FEATURES -----------
print("\n=== AUTO BINNING ALL NUMERIC FEATURES ===")

for col in numeric_features:
    try:
        df[f"{col}_BIN"] = pd.qcut(df[col], q=4, duplicates="drop")
        table = pd.crosstab(df[f"{col}_BIN"], df["CLASS"])
        table.columns = ["Normal","Prediabetes","Diabetes"]
        print(f"\n{col} Distribution:")
        print(table)
    except:
        pass

# ======================================================
# 6. FEATURE CORRELATION HEATMAP (WITH VALUES)
# ======================================================
plt.figure(figsize=(13,11))
corr = df[numeric_features].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ======================================================
# =================== MACHINE LEARNING =================
# ======================================================

X = df[numeric_features]
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
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# ---------------- CROSS VALIDATION --------------------
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro")

print(f"\nCV F1 Macro (mean): {cv_f1.mean():.3f}")
print(f"CV F1 Macro (std) : {cv_f1.std():.3f}")

# ---------------- TRAIN MODEL -------------------------
pipeline.fit(X_train, y_train)

# ---------------- EVALUATION --------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print("\n=== TEST SET PERFORMANCE (BASELINE) ===")
print(f"Accuracy        : {accuracy_score(y_test,y_pred):.3f}")
print(f"F1-score (Macro): {f1_score(y_test,y_pred,average='macro'):.3f}")
print(f"F1-score (Wght) : {f1_score(y_test,y_pred,average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal","Prediabetes","Diabetes"]
))

# ---------------- CONFUSION MATRIX --------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Prediabetes","Diabetes"],
    yticklabels=["Normal","Prediabetes","Diabetes"]
)
plt.title("Confusion Matrix (Baseline)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------------- ROC AUC -----------------------------
y_test_bin = label_binarize(y_test, classes=[0,1,2])
roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
print(f"ROC AUC (OvR): {roc_auc:.3f}")

# ---------------- SAVE MODEL --------------------------
joblib.dump(pipeline, "rf_diabetes_baseline_no_smote.joblib")
print("\n✅ Baseline model saved successfully")
