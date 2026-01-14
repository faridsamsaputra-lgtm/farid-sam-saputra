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
    st.markdown("### Alur & Langkah RANDOM FOREST")

    st.markdown("""
    ## 1️⃣ Bootstrap Sampling (Pendekatan Statistik)

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
    ## 2️⃣ Pemilihan Fitur Acak (Random Subspace)

    Pada setiap node pohon, dipilih m fitur acak dari total p fitur:

    m = √p   (untuk klasifikasi)

    Tujuan statistik:
    - Mengurangi korelasi antar pohon
    - Meningkatkan generalisasi model
    """)

    st.markdown("""
    ## 3️⃣ Pembentukan Decision Tree (Kriteria Gini)

    Setiap node dihitung nilai impurity menggunakan Gini Index:

    Gini(t) = 1 − Σ(pₖ²)

    dimana pₖ adalah proporsi kelas ke-k.

    Penurunan impurity dihitung dengan:

    ΔGini = Gini(parent) − [(nL/n)·Gini(L) + (nR/n)·Gini(R)]

    Split terbaik adalah yang memiliki ΔGini terbesar.
    """)

    st.markdown("""
    ## 4️⃣ Pembentukan Ensemble (Hutan Acak)

    Setelah proses di atas, terbentuk kumpulan pohon:

    {T₁, T₂, ..., Tᴮ}

    Pendekatan ensemble ini menurunkan risiko overfitting.
    """)

    st.markdown("""
    ## 5️⃣ Prediksi & Majority Voting

    Setiap pohon memberikan prediksi:

    ŷᵦ = Tᵦ(x)

    Prediksi akhir ditentukan dengan voting mayoritas:

    ŷ = 1, jika Σŷᵦ > B/2
    """)

    st.markdown("""
    ## 6️⃣ Evaluasi Statistik Model

    - Confusion Matrix
    - Accuracy
    - Recall (Sensitivitas)

    Recall menjadi metrik utama karena kesalahan FN
    sangat berisiko dalam konteks medis.
    """)

    st.markdown("""
    ## 7️⃣ Feature Importance

    Feature importance dihitung dari rata-rata penurunan Gini,
    menunjukkan variabel klinis paling dominan dalam prediksi.
    """)

    st.success("""
    ✔ Random Forest memiliki performa paling stabil  
    ✔ Cocok untuk data kesehatan multivariat  
    ✔ Digunakan sebagai model utama
    """)

# =========================================
# TAB 4
# =========================================
with tab4:
    import prediction
    prediction.prediction_app()

# =========================================
# TAB 5
# =========================================
# =========================================
# TAB 5 - REPORT ANALYSIS
# =========================================
with tab5:
    st.markdown("## 🧾 REPORT ANALYSIS & INTERPRETASI HASIL")

    st.markdown("""
    ### 1️⃣ Tujuan Analisis

    Penelitian ini bertujuan untuk mengevaluasi kinerja beberapa algoritma
    klasifikasi machine learning dalam memprediksi risiko serangan jantung,
    serta menentukan model terbaik berdasarkan evaluasi statistik dan
    interpretasi medis.
    """)

    st.markdown("""
    ---
    ### 2️⃣ Evaluasi Kinerja Model

    Evaluasi model dilakukan menggunakan data uji (testing set) dengan
    beberapa metrik utama, yaitu:

    - **Accuracy**
    - **Precision**
    - **Recall (Sensitivity)**
    - **F1-Score**
    - **Confusion Matrix**

    Fokus utama evaluasi adalah **Recall**, karena dalam konteks medis,
    kesalahan *False Negative* (pasien berisiko tetapi diprediksi sehat)
    dapat berdampak fatal.
    """)

    st.markdown("""
    **Hasil evaluasi menunjukkan bahwa:**
    - Model **Random Forest** memiliki nilai Recall tertinggi
    - Variansi prediksi lebih stabil dibanding model lain
    - Overfitting dapat diminimalkan melalui pendekatan ensemble
    """)

    st.markdown("""
    ---
    ### 3️⃣ Confusion Matrix & Implikasi Medis

    Confusion Matrix digunakan untuk menganalisis kesalahan klasifikasi:

    - **True Positive (TP)**: Pasien berisiko dan terdeteksi dengan benar
    - **True Negative (TN)**: Pasien sehat dan terdeteksi dengan benar
    - **False Positive (FP)**: Pasien sehat tetapi diprediksi berisiko
    - **False Negative (FN)**: Pasien berisiko tetapi diprediksi sehat

    Dalam sistem pendukung keputusan medis, kesalahan **FN** harus ditekan
    seminimal mungkin. Random Forest terbukti mampu menurunkan jumlah FN
    secara signifikan.
    """)

    st.markdown("""
    ---
    ### 4️⃣ Analisis Probabilitas Prediksi

    Tidak hanya menghasilkan label klasifikasi (0 atau 1),
    Random Forest juga menghasilkan **probabilitas risiko**:

    \\[
    P(Y=1|X) = \\frac{1}{B} \\sum_{b=1}^{B} P_b(Y=1|X)
    \\]

    Dimana setiap pohon memberikan estimasi probabilitas,
    kemudian dirata-ratakan secara ensemble.
    """)

    st.markdown("""
    **Interpretasi probabilitas:**
    - **< 30%** → Risiko rendah
    - **30% – 60%** → Risiko sedang
    - **> 60%** → Risiko tinggi

    Pendekatan ini memberikan informasi yang lebih kaya
    dibandingkan prediksi biner.
    """)

    st.markdown("""
    ---
    ### 5️⃣ Feature Importance & Interpretasi Klinis

    Analisis feature importance menunjukkan bahwa variabel
    yang paling berpengaruh terhadap risiko serangan jantung adalah:

    - Usia (Age)
    - Tekanan darah (Resting Blood Pressure)
    - Kolesterol (Cholesterol)
    - Detak jantung maksimum (Max Heart Rate)
    - Nyeri dada (Chest Pain Type)

    Feature importance dihitung berdasarkan rata-rata penurunan
    nilai Gini pada seluruh pohon Random Forest.
    """)

    st.markdown("""
    ---
    ### 6️⃣ Kesimpulan Analisis

    Berdasarkan hasil evaluasi dan interpretasi statistik:

    - Random Forest memberikan performa terbaik dan stabil
    - Model mampu menghasilkan prediksi probabilistik yang informatif
    - Cocok digunakan sebagai **Decision Support System (DSS)**
      dalam skrining awal risiko serangan jantung
    """)

    st.success("""
    ✔ Model valid secara statistik  
    ✔ Interpretatif secara medis  
    ✔ Layak digunakan sebagai sistem pendukung keputusan
    """)


# =========================================
# TAB 6
# =========================================
with tab6:
    st.subheader("Contact Person")
    st.markdown("""
    * **Nama**: FARID SAM SAPUTRA  
    * **Email**: faridsamsaputra@gmail.com  
    * **Lokasi**: Semarang  
    """)
