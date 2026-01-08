# # ======================================================
# # Multiclass Diabetes Risk Prediction
# # Normal (0) - Prediabetes (1) - Diabetes (2)
# # Random Forest + SMOTE + K-Fold Cross Validation
# # + EDA Visualizations
# # + Age-based Analysis
# # + Auto Feature-based Analysis (ALL FEATURES)
# # + Feature Correlation Heatmap (WITH VALUES)
# # + Confusion Matrix Heatmap
# # ======================================================

# import warnings
# warnings.filterwarnings("ignore")

# import pandas as pd
# import numpy as np
# import joblib

# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, label_binarize
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     classification_report,
#     confusion_matrix,
#     roc_auc_score
# )

# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE

# # ======================================================
# # 1. LOAD DATASET
# # ======================================================
# DATA_PATH = "data/diabetes.csv"
# df = pd.read_csv(DATA_PATH)

# print("Dataset loaded:", df.shape)

# # ======================================================
# # 2. NORMALIZE COLUMN NAMES
# # ======================================================
# df.columns = (
#     df.columns
#     .str.strip()
#     .str.upper()
#     .str.replace(" ", "_")
#     .str.replace("-", "_")
# )

# # ======================================================
# # 3. ENCODE CATEGORICAL VARIABLES
# # ======================================================
# df["GENDER"] = (
#     df["GENDER"]
#     .astype(str)
#     .str.upper()
#     .map({"M": 1, "F": 0})
# )

# df["CLASS"] = (
#     df["CLASS"]
#     .astype(str)
#     .str.upper()
#     .map({"N": 0, "P": 1, "Y": 2})
# )

# df = df.dropna(subset=["CLASS"])
# df["CLASS"] = df["CLASS"].astype(int)

# print("\nClass distribution:")
# print(df["CLASS"].value_counts().sort_index())

# # ======================================================
# # 4. NUMERIC CONVERSION
# # ======================================================
# non_numeric = ["CLASS", "GENDER"]

# for col in df.columns:
#     if col not in non_numeric:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

# # ======================================================
# # 5. DROP ID COLUMNS
# # ======================================================
# id_cols = [c for c in df.columns if "ID" in c or "NO" in c]
# df = df.drop(columns=id_cols, errors="ignore")

# # ======================================================
# # ======================= EDA ==========================
# # ======================================================

# # ------------------------------------------------------
# # EDA 1: CLASS DISTRIBUTION
# # ------------------------------------------------------
# plt.figure(figsize=(6, 4))
# sns.countplot(x="CLASS", data=df, palette="Set2")
# plt.xticks([0, 1, 2], ["Normal", "Prediabetes", "Diabetes"])
# plt.title("Class Distribution")
# plt.tight_layout()
# plt.show()

# # ------------------------------------------------------
# # EDA 2: FEATURE DISTRIBUTIONS
# # ------------------------------------------------------
# numeric_features = [
#     col for col in df.columns
#     if col not in ["CLASS", "GENDER", "HBA1C"]
# ]

# df[numeric_features].hist(
#     bins=30,
#     figsize=(14, 10),
#     edgecolor="black"
# )

# plt.suptitle("Feature Distributions", fontsize=14)
# plt.tight_layout()
# plt.show()

# # ------------------------------------------------------
# # EDA 3: BOXPLOT BMI vs CLASS
# # ------------------------------------------------------
# plt.figure(figsize=(8, 5))
# sns.boxplot(x="CLASS", y="BMI", data=df, palette="Set3")
# plt.xticks([0, 1, 2], ["Normal", "Prediabetes", "Diabetes"])
# plt.title("BMI Distribution by Class")
# plt.tight_layout()
# plt.show()

# # ------------------------------------------------------
# # EDA 4: AGE GROUP vs DIABETES STATUS
# # ------------------------------------------------------
# age_bins = [0, 30, 40, 50, 60, 120]
# age_labels = ["<30", "30-39", "40-49", "50-59", "60+"]

# df["AGE_GROUP"] = pd.cut(df["AGE"], bins=age_bins, labels=age_labels, right=False)

# age_table = pd.crosstab(df["AGE_GROUP"], df["CLASS"])
# age_table.columns = ["Normal", "Prediabetes", "Diabetes"]

# print("\nDiabetes Distribution by Age Group:")
# print(age_table)

# plt.figure(figsize=(8, 5))
# ax = sns.countplot(data=df, x="AGE_GROUP", hue="CLASS", palette="Set2")

# for container in ax.containers:
#     ax.bar_label(container, fmt="%d", fontsize=9)

# plt.legend(title="Class", labels=["Normal", "Prediabetes", "Diabetes"])
# plt.title("Diabetes Distribution by Age Group")
# plt.tight_layout()
# plt.show()

# # ------------------------------------------------------
# # EDA 5: AUTO ANALYSIS FOR ALL NUMERIC FEATURES
# # ------------------------------------------------------
# print("\n===== AUTO FEATURE-BASED EDA (ALL FEATURES) =====")

# for feature in numeric_features:
#     if feature in ["AGE"]:
#         continue

#     print(f"\n--- {feature} vs CLASS ---")

#     if df[feature].isna().all():
#         print("Skipped (all NaN)")
#         continue

#     try:
#         df[f"{feature}_GROUP"] = pd.qcut(
#             df[feature],
#             q=4,
#             duplicates="drop"
#         )
#     except ValueError:
#         print("Skipped (not enough unique values)")
#         continue

#     table = pd.crosstab(df[f"{feature}_GROUP"], df["CLASS"])
#     table.columns = ["Normal", "Prediabetes", "Diabetes"]
#     print(table)

#     plt.figure(figsize=(8, 5))
#     ax = sns.countplot(
#         data=df,
#         x=f"{feature}_GROUP",
#         hue="CLASS",
#         palette="Set2"
#     )

#     for container in ax.containers:
#         ax.bar_label(container, fmt="%d", fontsize=8)

#     plt.title(f"Diabetes Distribution by {feature}")
#     plt.xlabel(f"{feature} Group (Quartiles)")
#     plt.ylabel("Number of Patients")
#     plt.xticks(rotation=30)
#     plt.tight_layout()
#     plt.show()

# # ======================================================
# # 6. FEATURE CORRELATION HEATMAP (WITH VALUES)
# # ======================================================
# plt.figure(figsize=(13, 11))

# corr = df.drop(["CLASS", "HBA1C"], axis=1).corr()

# sns.heatmap(
#     corr,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     linewidths=0.5,
#     square=True,
#     cbar_kws={"shrink": 0.8}
# )

# plt.title("Feature Correlation Heatmap (Risk Factors)", fontsize=14)
# plt.tight_layout()
# plt.show()

# # ======================================================
# # 7. SPLIT FEATURES & TARGET
# # ======================================================
# X = df.drop(["CLASS", "HBA1C"], axis=1)
# y = df["CLASS"]

# print("\nTraining features:")
# print(X.columns.tolist())

# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

# # ======================================================
# # 8. PIPELINE (SMOTE ONLY ON TRAINING)
# # ======================================================
# pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
#     ("smote", SMOTE(random_state=42)),
#     ("rf", RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# # ======================================================
# # 9. STRATIFIED K-FOLD CROSS VALIDATION
# # ======================================================
# print("\n=== STRATIFIED K-FOLD CROSS VALIDATION ===")

# skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# cv_f1_macro = cross_val_score(
#     pipeline,
#     X_train_full,
#     y_train_full,
#     cv=skf,
#     scoring="f1_macro"
# )

# print(f"CV F1 Macro (mean): {cv_f1_macro.mean():.3f}")
# print(f"CV F1 Macro (std) : {cv_f1_macro.std():.3f}")

# # ======================================================
# # 10. TRAIN FINAL MODEL
# # ======================================================
# pipeline.fit(X_train_full, y_train_full)

# # ======================================================
# # 11. FINAL EVALUATION
# # ======================================================
# y_pred = pipeline.predict(X_test)
# y_prob = pipeline.predict_proba(X_test)

# print("\n=== TEST SET PERFORMANCE ===")
# print(f"Accuracy        : {accuracy_score(y_test, y_pred):.3f}")
# print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
# print(f"F1-score (Wght) : {f1_score(y_test, y_pred, average='weighted'):.3f}")

# print("\nClassification Report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=["Normal", "Prediabetes", "Diabetes"]
# ))

# # ======================================================
# # 12. CONFUSION MATRIX HEATMAP
# # ======================================================
# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Normal", "Prediabetes", "Diabetes"],
#     yticklabels=["Normal", "Prediabetes", "Diabetes"]
# )

# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix Heatmap")
# plt.tight_layout()
# plt.show()

# # ======================================================
# # 13. ROC AUC (OvR)
# # ======================================================
# y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
# roc_auc = roc_auc_score(
#     y_test_bin,
#     y_prob,
#     average="macro",
#     multi_class="ovr"
# )

# print(f"ROC AUC (OvR): {roc_auc:.3f}")

# # ======================================================
# # 14. SAVE MODEL
# # ======================================================
# MODEL_PATH = "rf_diabetes_multiclass.joblib"
# joblib.dump(pipeline, MODEL_PATH)

# print(f"\n✅ Model saved as {MODEL_PATH}")

# ======================================================
# Multiclass Diabetes Risk Prediction (RISK MODEL)
# Normal (0) - Prediabetes (1) - Diabetes (2)
# Random Forest + SMOTE
# WITHOUT HbA1c
# + Stratified K-Fold (Accuracy & F1)
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

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

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
df["CLASS"]  = df["CLASS"].astype(str).str.upper().map({"N": 0, "P": 1, "Y": 2})

df = df.dropna(subset=["CLASS"])
df["CLASS"] = df["CLASS"].astype(int)

print("\nClass distribution:")
print(df["CLASS"].value_counts().sort_index())

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
# 6. FEATURE SET (WITHOUT HbA1c)
# ======================================================
X = df.drop(["CLASS", "HBA1C"], axis=1)
y = df["CLASS"]

print("\nTraining features:")
print(X.columns.tolist())

# ======================================================
# 7. TRAIN TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# 8. PIPELINE (SMOTE ONLY ON TRAINING)
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
# 9. STRATIFIED K-FOLD CROSS VALIDATION
# ======================================================
print("\n=== STRATIFIED K-FOLD CROSS VALIDATION ===")

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

cv_acc = []
cv_f1  = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    y_val_pred = pipeline.predict(X_train.iloc[val_idx])

    acc = accuracy_score(y_train.iloc[val_idx], y_val_pred)
    f1  = f1_score(y_train.iloc[val_idx], y_val_pred, average="macro")

    cv_acc.append(acc)
    cv_f1.append(f1)

    print(f"Fold {fold} | Accuracy: {acc:.3f} | F1 Macro: {f1:.3f}")

print("\n=== CROSS-VALIDATION SUMMARY ===")
print(f"Mean Accuracy : {np.mean(cv_acc):.3f}")
print(f"Std Accuracy  : {np.std(cv_acc):.3f}")
print(f"Mean F1 Macro : {np.mean(cv_f1):.3f}")
print(f"Std F1 Macro  : {np.std(cv_f1):.3f}")

# ======================================================
# 10. TRAIN FINAL MODEL
# ======================================================
pipeline.fit(X_train, y_train)

# ======================================================
# 11. FINAL EVALUATION (TEST SET)
# ======================================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print("\n=== TEST SET PERFORMANCE ===")
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
# 12. CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal","Prediabetes","Diabetes"],
    yticklabels=["Normal","Prediabetes","Diabetes"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ======================================================
# 13. ROC AUC (OvR)
# ======================================================
y_test_bin = label_binarize(y_test, classes=[0,1,2])
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
joblib.dump(pipeline, "rf_diabetes_risk_model_no_hba1c.joblib")
print("\n✅ Model saved successfully")
