import streamlit as st
import pandas as pd
import joblib
import numpy as np

def prediction_app():

    st.subheader("🫀 Prediksi Risiko Serangan Jantung (Random Forest)")

    # ===============================
    # LOAD MODEL & METADATA
    # ===============================
    try:
        model = joblib.load("model_random_forest.pkl")
        features = joblib.load("model_features.pkl")
        scaler = joblib.load("scaler.pkl")
        numeric_cols = joblib.load("numeric_columns.pkl")
    except:
        st.error("Model Random Forest atau file pendukung tidak ditemukan")
        return

    # ===============================
    # INPUT USER
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["M", "F"])
        age = st.number_input("Usia", 1, 120, 40)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoker = st.selectbox("Perokok", [0, 1])
        sbp = st.number_input("Tekanan Darah Sistolik", 80.0, 250.0, 120.0)

    with col2:
        diabetes = st.selectbox("Diabetes", [0, 1])
        family_history = st.selectbox("Riwayat Kardiovaskular", [0, 1])
        atrial_fib = st.selectbox("Atrial Fibrillation", [0, 1])
        ckd = st.selectbox("CKD", [0, 1])
        fev1 = st.number_input("FEV1", 0.0, 5.0, 3.0)

    # ===============================
    # DATA PREPARATION (KONSISTEN)
    # ===============================
    input_data = {
        "age": age,
        "body_mass_index": bmi,
        "systolic_blood_pressure": sbp,
        "forced_expiratory_volume_1": fev1,
        "smoker": smoker,
        "diabetes": diabetes,
        "atrial_fibrillation": atrial_fib,
        "chronic_kidney_disease": ckd,
        "family_history_of_cardiovascular_disease": family_history,
        "hypertension_treated": 1 if sbp >= 140 else 0,
        "rheumatoid_arthritis": 0,
        "chronic_obstructive_pulmonary_disorder": 1 if fev1 < 2 else 0,
        "gender_M": 1 if gender == "M" else 0
    }

    df = pd.DataFrame([input_data])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # ===============================
    # PREDIKSI RANDOM FOREST
    # ===============================
    if st.button("Prediksi Risiko"):
        prob = model.predict_proba(df)[0][1]
        threshold = 0.5
        pred_class = 1 if prob >= threshold else 0

        st.write("---")
        st.metric("Probabilitas Serangan Jantung", f"{prob*100:.2f}%")

        # ===============================
        # KATEGORI RISIKO
        # ===============================
        if prob >= 0.80:
            risk_level = "SANGAT TINGGI"
            st.error("🔴 Risiko Sangat Tinggi Serangan Jantung")
        elif prob >= 0.60:
            risk_level = "TINGGI"
            st.error("🟠 Risiko Tinggi Serangan Jantung")
        elif prob >= 0.40:
            risk_level = "MENENGAH"
            st.warning("🟡 Risiko Menengah Serangan Jantung")
        else:
            risk_level = "RENDAH"
            st.success("🟢 Risiko Rendah Serangan Jantung")

        # ===============================
        # INTERPRETASI OTOMATIS
        # ===============================
        st.subheader("📊 Interpretasi Hasil")
        st.write(
            f"Model Random Forest memprediksi bahwa pasien berada pada **tingkat risiko {risk_level}** "
            f"dengan probabilitas **{prob*100:.2f}%** mengalami serangan jantung."
        )

        # ===============================
        # IDENTIFIKASI FAKTOR RISIKO
        # ===============================
        st.subheader("⚠️ Faktor Risiko yang Terdeteksi")

        risk_factors = []

        if age >= 55:
            risk_factors.append("Usia berisiko tinggi")
        if bmi >= 25:
            risk_factors.append("Berat badan berlebih / obesitas")
        if smoker == 1:
            risk_factors.append("Kebiasaan merokok")
        if sbp >= 140:
            risk_factors.append("Tekanan darah tinggi (hipertensi)")
        if diabetes == 1:
            risk_factors.append("Diabetes")
        if atrial_fib == 1:
            risk_factors.append("Atrial fibrillation")
        if ckd == 1:
            risk_factors.append("Penyakit ginjal kronis")
        if family_history == 1:
            risk_factors.append("Riwayat keluarga penyakit jantung")
        if fev1 < 2:
            risk_factors.append("Gangguan fungsi paru")

        if risk_factors:
            for rf in risk_factors:
                st.write(f"• {rf}")
        else:
            st.write("Tidak ditemukan faktor risiko dominan")

        # ===============================
        # REKOMENDASI KESEHATAN
        # ===============================
        st.subheader("💡 Rekomendasi Tindak Lanjut")

        recommendations = []

        if smoker == 1:
            recommendations.append("Hentikan kebiasaan merokok secara bertahap")
        if bmi >= 25:
            recommendations.append("Menurunkan berat badan melalui diet seimbang dan aktivitas fisik")
        if sbp >= 140:
            recommendations.append("Mengontrol tekanan darah secara rutin")
        if diabetes == 1:
            recommendations.append("Mengontrol kadar gula darah")
        if atrial_fib == 1:
            recommendations.append("Pemantauan irama jantung secara berkala")
        if ckd == 1:
            recommendations.append("Konsultasi lanjutan terkait fungsi ginjal")

        if risk_level in ["TINGGI", "SANGAT TINGGI"]:
            recommendations.append("Disarankan konsultasi langsung dengan dokter spesialis jantung")

        for rec in recommendations:
            st.write(f"✔ {rec}")

        st.warning(
            "⚠️ Hasil ini merupakan prediksi berbasis machine learning dan tidak menggantikan diagnosis medis profesional."
        )


if __name__ == "__main__":
    prediction_app()
