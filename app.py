import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction (Multiclass)")
st.write(
    "Aplikasi ini memprediksi tingkat risiko diabetes "
    "menjadi Normal, Prediabetes, atau Diabetes "
    "menggunakan algoritma Random Forest."
)

# =====================
# LOAD MODEL
# =====================
model = joblib.load("rf_diabetes_multiclass.joblib")

# =====================
# INPUT FORM
# =====================
st.subheader("Patient Information")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=0.0, value=23.0)
hba1c = st.number_input("HbA1c", min_value=0.0, value=5.5)
chol = st.number_input("Cholesterol", min_value=0.0, value=180.0)
tg = st.number_input("Triglycerides", min_value=0.0, value=120.0)
hdl = st.number_input("HDL", min_value=0.0, value=50.0)
ldl = st.number_input("LDL", min_value=0.0, value=100.0)
vldl = st.number_input("VLDL", min_value=0.0, value=25.0)
urea = st.number_input("Urea", min_value=0.0, value=30.0)
cr = st.number_input("Creatinine", min_value=0.0, value=1.0)

input_df = pd.DataFrame([{
    "GENDER": 1 if gender == "Male" else 0,
    "AGE": age,
    "UREA": urea,
    "CR": cr,
    "HBA1C": hba1c,
    "CHOL": chol,
    "TG": tg,
    "HDL": hdl,
    "LDL": ldl,
    "VLDL": vldl,
    "BMI": bmi
}])

# =====================
# PREDICTION
# =====================
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    labels = ["🟢 Normal", "🟡 Prediabetes", "🔴 Diabetes"]

    st.success(f"Predicted Risk Level: {labels[pred]}")
    st.write("Probability Distribution:")

    for lbl, p in zip(labels, prob):
        st.write(f"{lbl}: {p:.2f}")
