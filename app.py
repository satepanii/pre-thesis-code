import streamlit as st
import pandas as pd
import joblib

<<<<<<< HEAD
# ======================================================
# PAGE CONFIG
# ======================================================
=======
# test push lagi hehe

>>>>>>> e0dd9ae721ea9666e853c293a79c8a1edcda9916
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
        sehingga hasil prediksi merepresentasikan **risiko relatif dalam
        konteks klinis**, bukan populasi umum.
        """
    )

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "rf_diabetes_multiclass.joblib"
model = joblib.load(MODEL_PATH)

# ======================================================
# INPUT FORM
# ======================================================
st.subheader("🧍 Patient Information")

# ---- BASIC INFO (NO EXPLANATION) ----
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

    📌 **Nilai normal:** < 200 mg/dL  
    📌 Kadar tinggi berhubungan dengan gangguan metabolik.
    """
)
chol = st.number_input("Total Cholesterol (mg/dL)", 0.0, 400.0, 180.0)

# ======================================================
# TRIGLYCERIDES
# ======================================================
st.info(
    """
    **Triglycerides (TG)** adalah jenis lemak darah utama.

    📌 **Nilai normal:** < 150 mg/dL  
    📌 Nilai tinggi sering dikaitkan dengan sindrom metabolik.
    """
)
tg = st.number_input("Triglycerides (mg/dL)", 0.0, 500.0, 120.0)

# ======================================================
# HDL
# ======================================================
st.info(
    """
    **HDL (High-Density Lipoprotein)** dikenal sebagai kolesterol baik.

    📌 **Nilai normal:**  
    - Pria  > 40 mg/dL  
    - Wanita > 50 mg/dL  

    📌 HDL tinggi bersifat protektif terhadap diabetes.
    """
)
hdl = st.number_input("HDL (mg/dL)", 0.0, 150.0, 50.0)

# ======================================================
# LDL
# ======================================================
st.info(
    """
    **LDL (Low-Density Lipoprotein)** dikenal sebagai kolesterol jahat.

    📌 **Nilai optimal:** < 100 mg/dL  
    📌 LDL tinggi meningkatkan risiko penyakit metabolik.
    """
)
ldl = st.number_input("LDL (mg/dL)", 0.0, 300.0, 100.0)

# ======================================================
# VLDL
# ======================================================
st.info(
    """
    **VLDL (Very Low-Density Lipoprotein)** membawa trigliserida dalam darah.

    📌 **Nilai normal:** 5 – 40 mg/dL  
    📌 Sering meningkat pada gangguan metabolisme lemak.
    """
)
vldl = st.number_input("VLDL (mg/dL)", 0.0, 100.0, 25.0)

# ======================================================
# UREA
# ======================================================
st.info(
    """
    **Urea** merupakan indikator fungsi ginjal.

    📌 **Nilai normal:** 15 – 45 mg/dL  
    📌 Diabetes jangka panjang dapat memengaruhi kadar urea.
    """
)
urea = st.number_input("Urea (mg/dL)", 0.0, 100.0, 30.0)

# ======================================================
# CREATININE
# ======================================================
st.info(
    """
    **Creatinine** digunakan untuk menilai fungsi ginjal.

    📌 **Nilai normal:**  
    - Pria    : 0.7 – 1.3 mg/dL  
    - Wanita  : 0.6 – 1.1 mg/dL  

    📌 Gangguan ginjal sering menjadi komplikasi diabetes.
    """
)
cr = st.number_input("Creatinine (mg/dL)", 0.0, 10.0, 1.0)

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
