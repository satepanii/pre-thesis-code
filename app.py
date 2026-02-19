import streamlit as st
import pandas as pd
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Diabetes Risk Analysis",
    layout="centered"
)

# ======================================================
# TITLE & INTRO
# ======================================================
st.title("🩺 Diabetes Risk Analysis")

st.write(
    """
    Aplikasi ini menganalisis **tingkat risiko diabetes**
    (**Normal / Prediabetes / Diabetes**) menggunakan algoritma
    **Random Forest**.

    ⚠️ *Hasil yang ditampilkan merupakan analisis risiko relatif
    berbasis data klinis, **bukan diagnosis medis**.*
    """
)

# ======================================================
# DATASET INFO
# ======================================================
with st.expander("📊 Dataset Information"):
    st.write(
        """
        Dataset dikumpulkan dari pasien rumah sakit
        (Medical City Hospital dan Specialized Center for Endocrinology
        and Diabetes – Iraq).

        Dataset berisi informasi medis dan hasil pemeriksaan laboratorium,
        sehingga hasil prediksi merepresentasikan risiko relatif
        dalam konteks klinis.
        """
    )

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "rf_diabetes_without_hba1c.joblib"
model = joblib.load(MODEL_PATH)

# ======================================================
# INPUT FORM
# ======================================================
st.subheader("🧍 Patient Information")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age (years)", 0, 120, 30)

# ======================================================
# BMI
# ======================================================
st.info(
    """
    **Body Mass Index (BMI)** menunjukkan status berat badan terhadap tinggi badan.

    📌 **Rentang normal:** 18.5 – 24.9 kg/m²  
    📌 BMI tinggi sering dikaitkan dengan resistensi insulin.
    """
)
bmi = st.number_input("BMI (kg/m²)", 0.0, 60.0, 23.0)

# ======================================================
# CHOLESTEROL
# ======================================================
st.info(
    """
    **Total Cholesterol** menggambarkan kadar lemak total dalam darah.

    📌 **Nilai normal:** < 5.2 mmol/L  
    📌 Kadar tinggi berhubungan dengan gangguan metabolik.
    """
)
chol = st.number_input("Total Cholesterol (mmol/L)", 0.0, 20.0, 5.0)

# ======================================================
# TRIGLYCERIDES
# ======================================================
st.info(
    """
    **Triglycerides (TG)** adalah jenis lemak darah utama.

    📌 **Nilai normal:** < 1.7 mmol/L  
    📌 Nilai tinggi sering dikaitkan dengan sindrom metabolik.
    """
)
tg = st.number_input("Triglycerides (mmol/L)", 0.0, 15.0, 1.5)

# ======================================================
# HDL
# ======================================================
st.info(
    """
    **HDL (High-Density Lipoprotein)** dikenal sebagai kolesterol baik.

    📌 **Nilai normal:**  
    - Pria  > 1.0 mmol/L  
    - Wanita > 1.3 mmol/L  

    📌 HDL tinggi bersifat protektif terhadap diabetes.
    """
)
hdl = st.number_input("HDL (mmol/L)", 0.0, 5.0, 1.3)

# ======================================================
# LDL
# ======================================================
st.info(
    """
    **LDL (Low-Density Lipoprotein)** dikenal sebagai kolesterol jahat.

    📌 **Nilai optimal:** < 2.6 mmol/L  
    📌 LDL tinggi meningkatkan risiko penyakit metabolik.
    """
)
ldl = st.number_input("LDL (mmol/L)", 0.0, 15.0, 2.5)

# ======================================================
# VLDL
# ======================================================
st.info(
    """
    **VLDL (Very Low-Density Lipoprotein)** membawa trigliserida dalam darah.

    📌 **Nilai normal:** 0.1 – 1.0 mmol/L  
    📌 Sering meningkat pada gangguan metabolisme lemak.
    """
)
vldl = st.number_input("VLDL (mmol/L)", 0.0, 5.0, 0.5)

# ======================================================
# UREA
# ======================================================
st.info(
    """
    **Urea** merupakan indikator fungsi ginjal.

    📌 **Nilai normal:** 2.5 – 7.1 mmol/L  
    📌 Diabetes jangka panjang dapat memengaruhi kadar urea.
    """
)
urea = st.number_input("Urea (mmol/L)", 0.0, 30.0, 5.0)

# ======================================================
# CREATININE
# ======================================================
st.info(
    """
    **Creatinine** digunakan untuk menilai fungsi ginjal.

    📌 **Nilai normal:**  
    - Pria    : 60 – 110 µmol/L  
    - Wanita  : 45 – 90 µmol/L  

    📌 Gangguan ginjal sering menjadi komplikasi diabetes.
    """
)
cr = st.number_input("Creatinine (µmol/L)", 0.0, 2000.0, 90.0)

# ======================================================
# PREPARE INPUT
# ======================================================
input_df = pd.DataFrame([{
    "GENDER": 1 if gender == "Male" else 0,
    "AGE": age,
    "UREA": urea,
    "CR": cr,
    "CHOL": chol,
    "TG": tg,
    "HDL": hdl,
    "LDL": ldl,
    "VLDL": vldl,
    "BMI": bmi
}])

# ======================================================
# PREDICTION
# ======================================================
if st.button("🔍 Analyze Risk"):
    probs = model.predict_proba(input_df)[0]

    p_normal, p_pre, p_diab = probs
    relative_risk = p_diab

    st.markdown("---")

    if relative_risk >= 0.70:
        st.error("🔴 High Risk of Diabetes (Relative Risk)")
    elif relative_risk >= 0.40:
        st.warning("🟡 Medium Risk of Diabetes (Relative Risk)")
    else:
        st.success("🟢 Low Risk of Diabetes (Relative Risk)")

    st.markdown("### 📈 Probability Distribution")
    st.write(f"🟢 Normal      : {p_normal:.2f}")
    st.write(f"🟡 Prediabetes : {p_pre:.2f}")
    st.write(f"🔴 Diabetes    : {p_diab:.2f}")

    st.markdown("### 📊 Relative Risk Score")
    st.write(f"**Relative Risk Index:** `{relative_risk:.2f}`")

    st.caption(
        "Relative Risk menunjukkan tingkat risiko pasien dibandingkan "
        "populasi klinis dalam dataset, bukan diagnosis absolut."
    )
