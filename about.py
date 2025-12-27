import streamlit as st
import os

def about_dataset():
    st.title("📘 Tentang Dataset & Metode")

    # Dua kolom: kiri gambar (2), kanan teks (3)
    col_img, col_text = st.columns([2, 3])

    # =========================
    # KOLOM KIRI (GAMBAR)
    # =========================
    with col_img:
        if os.path.exists("jantung.jpg"):
            st.image(
                "jantung.jpg",
                caption="Ilustrasi Jantung Manusia",
                use_container_width=True
            )
        else:
            st.image(
                "images.png",
                caption="Diagram Anatomi Jantung",
                use_container_width=True
            )

    # =========================
    # KOLOM KANAN (TEKS)
    # =========================
    with col_text:
        st.markdown("""
        ## 🫀 Dataset – Heart Attack
        Dataset ini berisi informasi medis pasien yang berkaitan 
        dengan faktor risiko dan kejadian serangan jantung
        (heart attack). Data dikumpulkan dari berbagai sumber
        terpercaya dan mencakup sejumlah variabel penting seperti
        identitas pasien (ID anonim), jenis kelamin, usia, indeks massa 
        tubuh (BMI), status perokok, tekanan darah sistolik, serta riwayat
        pengobatan hipertensi. Selain itu, dataset ini juga mencatat riwayat 
        kesehatan pasien seperti adanya penyakit ginjal kronis, atrial fibrilasi, 
        artritis reumatoid, diabetes, hingga penyakit paru obstruktif kronis (COPD).
        Terdapat pula variabel fungsi paru (Forced Expiratory Volume 1), waktu 
        hingga terjadinya kejadian klinis (time-to-event), serta label target yang 
        menunjukkan apakah pasien mengalami serangan jantung atau tidak. Dataset
        ini dapat digunakan untuk analisis statistik, penelitian medis, maupun 
        pengembangan model machine learning dalam rangka memprediksi risiko serangan 
        jantung berdasarkan faktor-faktor klinis. Data disediakan dalam format CSV 
        dengan jumlah entri pasien yang cukup representatif, dan ditujukan untuk 
        kepentingan penelitian, edukasi, serta pengembangan ilmu data kesehatan,
        bukan sebagai pengganti diagnosis medis profesional.

        **Target:**
        - `heart_attack`
          - 0 = Tidak
          - 1 = Ya

        **Metode:**
        - Klasifikasi (Supervised Learning)
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        - Decision Tree
        - KNN
        - Naive Bayes
        """)

    st.divider()
    st.success("📌 Aplikasi ini fokus pada penggunaan **Machine Learning untuk Klasifikasi Risiko Kesehatan**.")

if __name__ == "__main__":
    about_dataset()