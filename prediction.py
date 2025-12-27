import streamlit as st
import pandas as pd
import joblib
import numpy as np

def prediction_app():
    st.subheader("🫀 Prediksi Risiko Serangan Jantung")

    # Memuat model dan metadata
    try:
        model = joblib.load("model_logistic_regression.pkl") 
        features = joblib.load("model_features.pkl")
        numeric_cols = joblib.load("numeric_columns.pkl")
    except:
        st.error("Model tidak ditemukan! Silakan jalankan menu Training terlebih dahulu.")
        return

    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["M", "F"])
        age = st.number_input("Usia (Age)", 1, 120, 40)
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", 10.0, 60.0, 25.0)
        smoker = st.selectbox("Status Perokok (Smoker)", [0, 1], help="0: Tidak, 1: Ya")
        sbp = st.number_input("Tekanan Darah Sistolik (Systolic BP)", 80.0, 250.0, 120.0)

    with col2:
        diabetes = st.selectbox("Diabetes", [0, 1], help="0: Tidak, 1: Ya")
        family_history = st.selectbox("Riwayat Keluarga Kardiovaskular", [0, 1])
        atrial_fib = st.selectbox("Atrial Fibrillation", [0, 1])
        ckd = st.selectbox("Penyakit Ginjal Kronis (CKD)", [0, 1])
        fev1 = st.number_input("Forced Expiratory Volume 1 (FEV1)", 0.0, 5.0, 3.0)

    # Menyiapkan data untuk prediksi
    input_data = {
        "gender": gender,
        "age": age,
        "body_mass_index": bmi,
        "smoker": smoker,
        "systolic_blood_pressure": sbp,
        "hypertension_treated": 1 if sbp >= 140 else 0,
        "family_history_of_cardiovascular_disease": family_history,
        "atrial_fibrillation": atrial_fib,
        "chronic_kidney_disease": ckd,
        "rheumatoid_arthritis": 0,
        "diabetes": diabetes,
        "chronic_obstructive_pulmonary_disorder": 1 if fev1 < 2.0 else 0,
        "forced_expiratory_volume_1": fev1,
        "time_to_event_or_censoring": 10 
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    df[numeric_cols] = df[numeric_cols].astype(float)

    if st.button("Prediksi"):
        prob = model.predict_proba(df)[0][1]
        
        st.write("---")
        st.metric("Probabilitas Heart Attack", f"{prob*100:.2f}%")

        if prob >= 0.5:
            st.error("⚠️ Risiko Tinggi Serangan Jantung")
        else:
            st.success("✅ Risiko Rendah")

        # --- BAGIAN REKOMENDASI & SOLUSI (TAMBAHAN BARU) ---
        st.subheader("💡 Rekomendasi & Solusi Kesehatan")
        
        rekomendasi = []
        
        if sbp >= 140:
            rekomendasi.append("🔴 **Hipertensi:** Kurangi konsumsi garam (natrium), hindari stres, dan rutin cek tekanan darah.")
        
        if diabetes == 1:
            rekomendasi.append("🟠 **Diabetes:** Kontrol kadar gula darah secara ketat, batasi asupan karbohidrat sederhana, dan rutin berolahraga.")
            
        if smoker == 1:
            rekomendasi.append("🚬 **Merokok:** Segera hentikan kebiasaan merokok karena zat kimia dalam rokok mempercepat penyempitan pembuluh darah.")
            
        if bmi >= 25.0:
            rekomendasi.append("⚖️ **Obesitas/Overweight:** Jaga berat badan ideal dengan diet seimbang dan aktivitas fisik minimal 30 menit sehari.")
            
        if ckd == 1:
            rekomendasi.append("💧 **Penyakit Ginjal (CKD):** Konsultasikan asupan protein dan cairan dengan dokter untuk meringankan beban kerja ginjal.")

        if atrial_fib == 1:
            rekomendasi.append("💓 **Atrial Fibrillation:** Waspadai detak jantung tidak teratur; hindari kafein berlebih dan alkohol.")

        # Menampilkan rekomendasi jika ada masalah kesehatan yang terdeteksi
        if rekomendasi:
            for item in rekomendasi:
                st.write(item)
        else:
            st.write("🌟 Kondisi klinis Anda terpantau baik. Tetap pertahankan gaya hidup sehat!")
            
        st.warning("ℹ️ **Catatan:** Hasil ini adalah prediksi mesin. Selalu konsultasikan kondisi kesehatan Anda dengan dokter ahli.")

        # Menampilkan ringkasan input
        with st.expander("Lihat Detail Data Input"):
            st.dataframe(df)

if __name__ == "__main__":
    prediction_app()