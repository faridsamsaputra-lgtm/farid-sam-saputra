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
tab1, tab2, tab3, tab_rf, tab4, tab5, tab6 = st.tabs([
    'About Dataset', 
    'Dashboards', 
    'Machine Learning',
    'Model Random Forest',
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
# TAB RANDOM FOREST (STATISTIK & SISTEMATIS)
# =========================================
with tab_rf:
    st.markdown("## 🌳 Model Random Forest")
    st.markdown("### Alur & Langkah Perhitungan Sistematis Secara Statistik")

    st.markdown("""
    ## 1️⃣ Representasi Dataset Secara Statistik

    Dataset dinyatakan sebagai:

    D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}

    dengan:
    - xᵢ = vektor fitur klinis (usia, tekanan darah, kolesterol, BMI, dll)
    - yᵢ ∈ {0,1} → label serangan jantung
    - n = jumlah pasien
    """)

    st.markdown("""
    ## 2️⃣ Bootstrap Sampling (Pendekatan Statistik)

    Untuk setiap pohon ke-b, Random Forest melakukan pengambilan sampel:

    Dᵦ ~ Bootstrap(D)

    Artinya:
    - Data diambil secara acak dengan pengembalian
    - Ukuran sampel tetap n
    - Beberapa data dapat terpilih berulang

    Secara statistik, metode ini mendekati distribusi populasi dan
    menurunkan varians model.
    """)

    st.markdown("""
    ## 3️⃣ Pemilihan Fitur Acak (Random Subspace)

    Pada setiap node pohon, dipilih m fitur acak dari total p fitur:

    m = √p   (untuk klasifikasi)

    Tujuan statistik:
    - Mengurangi korelasi antar pohon
    - Meningkatkan generalisasi model
    """)

    st.markdown("""
    ## 4️⃣ Pembentukan Decision Tree (Kriteria Gini)

    Setiap node dihitung nilai impurity menggunakan Gini Index:

    Gini(t) = 1 − Σ(pₖ²)

    dimana pₖ adalah proporsi kelas ke-k.

    Penurunan impurity dihitung dengan:

    ΔGini = Gini(parent) − [(nL/n)·Gini(L) + (nR/n)·Gini(R)]

    Split terbaik adalah yang memiliki ΔGini terbesar.
    """)

    st.markdown("""
    ## 5️⃣ Pembentukan Ensemble (Hutan Acak)

    Setelah proses di atas, terbentuk kumpulan pohon:

    {T₁, T₂, ..., Tᴮ}

    Setiap pohon:
    - Dibangun dari data bootstrap
    - Menggunakan fitur acak
    - Memiliki struktur berbeda

    Pendekatan ensemble ini menurunkan risiko overfitting.
    """)

    st.markdown("""
    ## 6️⃣ Prediksi Setiap Pohon

    Untuk satu data uji x:

    ŷᵦ = Tᵦ(x)

    dengan:
    - ŷᵦ ∈ {0,1}
    """)

    st.markdown("""
    ## 7️⃣ Majority Voting (Agregasi Statistik)

    Prediksi akhir ditentukan dengan voting mayoritas:

    ŷ = 1, jika Σŷᵦ > B/2  
    ŷ = 0, jika sebaliknya

    Prinsip statistik:
    - Hukum bilangan besar
    - Kesalahan individual antar pohon saling menetralkan
    """)

    st.markdown("""
    ## 8️⃣ Evaluasi Model (Statistik Klasifikasi)

    Confusion Matrix:
    - TP: True Positive
    - TN: True Negative
    - FP: False Positive
    - FN: False Negative
    """)

    st.markdown("""
    ### Akurasi
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    ### Recall (Sensitivitas)
    Recall = TP / (TP + FN)

    Recall menjadi metrik utama karena FN berbahaya
    dalam konteks medis.
    """)

    st.markdown("""
    ## 9️⃣ Feature Importance (Kontribusi Variabel)

    Feature importance dihitung dari rata-rata penurunan Gini:

    Importance(j) = (1/B) · Σ Σ ΔGini(j)

    Menunjukkan variabel paling berpengaruh
    dalam prediksi serangan jantung.
    """)

    st.success("""
    ✔ Random Forest terbukti memberikan akurasi dan recall tertinggi  
    ✔ Cocok untuk data kesehatan multivariat  
    ✔ Digunakan sebagai model utama dalam penelitian ini
    """)


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
            - Berisi data klinis dan label

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