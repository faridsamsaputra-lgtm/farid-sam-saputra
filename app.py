import streamlit as st
import pandas as pd
import numpy as np

# =========================================
# Konfigurasi Halaman
# =========================================
st.set_page_config(
    page_title="Analisis Komparatif Algoritma Klasifikasi Machine Learning untuk Prediksi Risiko Serangan Jantung",
    layout="wide"
)

st.header('Analisis Komparatif Algoritma Klasifikasi Machine Learning untuk Prediksi Risiko Serangan Jantung')
st.write('**UAS MACHINE LEARNING 1.0** - Universitas Muhammadiyah Semarang')
st.write('Semarang, 25 Desember 2025')

# =========================================
# Tabs
# =========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'About Dataset', 
    'Dashboards', 
    'Machine Learning',
    'Prediction App',
    'Report Analysis', 
    'Contact Me'
])

# =========================================
# TAB 1
# =========================================
with tab1:
    import about
    about.about_dataset()

# =========================================
# TAB 2
# =========================================
with tab2:
    import visualisasi
    visualisasi.chart()

# =========================================
# TAB 3
# =========================================
with tab3:
    import machine_learning
    machine_learning.ml_model()

# =========================================
# TAB 4
# =========================================
with tab4:
    import prediction
    prediction.prediction_app()

# =========================================
# TAB 5 - REPORT ANALYSIS
# =========================================
with tab5:
    st.markdown("## 🧾 Ringkasan Penelitian")

    st.markdown("""
    ### Latar Belakang Penelitian

    Penyakit kardiovaskular, khususnya serangan jantung (heart attack), merupakan salah satu penyebab utama kematian di dunia dan menjadi masalah kesehatan yang serius, termasuk di Indonesia. Serangan jantung sering kali terjadi secara tiba-tiba, namun pada umumnya dipengaruhi oleh berbagai faktor risiko klinis seperti usia, tekanan darah, indeks massa tubuh, kebiasaan merokok, serta riwayat penyakit penyerta lainnya. Oleh karena itu, deteksi dini terhadap risiko serangan jantung menjadi langkah penting dalam upaya pencegahan dan penanganan yang lebih efektif.

    Perkembangan teknologi di bidang data science dan machine learning memungkinkan pemanfaatan data kesehatan dalam jumlah besar untuk melakukan analisis prediktif. Dengan memanfaatkan data klinis pasien, machine learning dapat digunakan untuk mengidentifikasi pola tersembunyi yang sulit dideteksi melalui analisis konvensional. Pendekatan ini diharapkan mampu membantu tenaga medis maupun peneliti dalam memprediksi risiko serangan jantung secara lebih akurat dan objektif.

    Dataset ini memiliki label target berupa kejadian serangan jantung (heart_attack) yang diklasifikasikan menjadi dua kelas.
    
    SUMBER DATA  
    https://www.kaggle.com/datasets/headsetbagus12/heart-attack-csv-dataset
    """)

    st.markdown("## 🧪 Metodologi Singkat")
    st.markdown("""
    1. Pengumpulan dan eksplorasi dataset
    2. Pembersihan data dan deteksi outlier
    3. Pembagian data latih dan uji
    4. Pelatihan model klasifikasi
    5. Evaluasi performa model
    6. Implementasi ke aplikasi Streamlit
    """)

    # =========================================
    # TAMBAHAN TOMBOL LANGKAH-LANGKAH PROCESSING
    # =========================================
    st.markdown("## ⚙️ Proses Pengolahan Data")

    if st.button("🔍 Tampilkan Langkah-Langkah Processing"):
        with st.expander("📌 Detail Tahapan Processing Data & Machine Learning", expanded=True):
            st.markdown("""
            ### 1️⃣ Data Collection
            - Dataset diambil dari Kaggle
            - Berisi data klinis dan label `heart_attack`

            ### 2️⃣ Data Cleaning
            - Menghapus data duplikat
            - Menangani missing value
            - Standarisasi fitur numerik

            ### 3️⃣ Outlier Detection
            - Deteksi nilai ekstrem pada:
              - Usia
              - Tekanan darah sistolik
              - BMI

            ### 4️⃣ Feature Selection
            - Memilih fitur klinis paling relevan

            ### 5️⃣ Data Splitting
            - Data latih dan data uji

            ### 6️⃣ Model Training
            - Logistic Regression
            - Random Forest
            - Decision Tree
            - KNN
            - Naive Bayes

            ### 7️⃣ Model Evaluation
            - Akurasi
            - Recall

            ### 8️⃣ Deployment
            - Implementasi model terbaik ke Streamlit
            """)

    st.subheader("📊 Laporan Hasil Analisis (Summary Report)")

    try:
        df = pd.read_csv('Data3.csv')
        total_pasien = len(df)
        total_attack = df['heart_attack'].sum()
        rate = (total_attack / total_pasien) * 100

        st.info(
            f"Berdasarkan analisis pada {total_pasien} data pasien, "
            f"ditemukan tingkat kejadian serangan jantung sebesar {rate:.2f}%."
        )

        st.markdown("---")
        st.subheader("🏆 Perbandingan Performa Model Terbaik")

        data_performa = {
            'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree', 'KNN', 'Naive Bayes'],
            'Akurasi (%)': [84.5, 91.2, 87.8, 86.4, 82.1],
            'Recall (%)': [81.0, 89.5, 85.2, 83.0, 88.4]
        }
        df_performa = pd.DataFrame(data_performa)

        col_a, col_b = st.columns(2)

        with col_a:
            st.dataframe(
                df_performa.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )

        with col_b:
            st.bar_chart(df_performa.set_index('Model')['Akurasi (%)'])

        st.markdown("""
        ### 📢 Kesimpulan
        - Random Forest memberikan performa terbaik
        - Faktor usia dan tekanan darah dominan
        - Model layak digunakan sebagai alat bantu awal
        """)

        st.download_button(
            "📥 Unduh Laporan CSV",
            df.to_csv(index=False),
            "laporan_analisis_heart_attack.csv",
            "text/csv"
        )

    except FileNotFoundError:
        st.error("File Data3.csv tidak ditemukan.")

# =========================================
# TAB 6
# =========================================
with tab6:
    st.subheader("Contact Person")
    st.markdown("""
    * **Nama**: FARID SAM SAPUTRA  
    * **Email**: faridsamsaputra@gmail.com  
    * **Lokasi**: Semarang, Jawa Tengah  
    * **Contact**: 082137330864  
    """)
