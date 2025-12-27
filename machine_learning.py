import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE 
import joblib

def ml_model():
    # 1. Membaca Dataset (Data3.csv)
    df = pd.read_csv('Data3.csv')
    # Menghapus patient_id karena bersifat unik dan tidak membantu prediksi
    df = df.drop(columns=['patient_id'])

    # Menangani missing values (Nilai kosong) agar tidak error saat pemodelan
    # Mengisi kolom numerik dengan median
    df['body_mass_index'] = df['body_mass_index'].fillna(df['body_mass_index'].median())
    df['systolic_blood_pressure'] = df['systolic_blood_pressure'].fillna(df['systolic_blood_pressure'].median())
    df['forced_expiratory_volume_1'] = df['forced_expiratory_volume_1'].fillna(df['forced_expiratory_volume_1'].median())

    # 2. Membagi kolom numerik dan kategorik (Target: heart_attack)
    numbers = ['age', 'body_mass_index', 'systolic_blood_pressure', 'forced_expiratory_volume_1']
    
    # 3. Deteksi dan penanganan outlier dengan IQR Method
    st.write('### 1. Deteksi Outlier') 
    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    st.write(f"Jumlah data sebelum pembersihan: **{df.shape[0]} baris**")
    df = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]
    st.write(f"Jumlah data setelah pembersihan outlier: **{df.shape[0]} baris**")

    # Tampilkan dataset awal
    st.write('**Dataset yang digunakan (Preview)**')
    st.dataframe(df.head())

    # 4. Feature Encoding
    df_select = pd.get_dummies(df, drop_first=True)

    # 5. Normalisasi kolom numerik dengan MinMax Scaler
    st.write('### 2. Normalisasi menggunakan MinMax Scaler')
    scaler = MinMaxScaler()
    df_select[numbers] = scaler.fit_transform(df_select[numbers])

    # Tampilkan Data Bersih (Setelah Encoding & Normalisasi)
    st.write('**Hasil Data Bersih (Setelah Preprocessing)**')
    st.dataframe(df_select.head())

    col1, col2 = st.columns(2)
    with col1:
        st.write('**Sebelum Normalisasi (Distribusi Asli)**')
        chart_old = (alt.Chart(df).transform_density('age', as_=['age', 'density'])
                    .mark_area(opacity=0.5).encode(x="age:Q", y="density:Q")
                    .properties(width=350, height=200, title="Density Plot: Age (Original)"))
        st.altair_chart(chart_old, use_container_width=True)

    with col2:
        st.write('**Setelah Normalisasi (Skala 0-1)**')
        chart_new = (alt.Chart(df_select).transform_density('age', as_=['age', 'density'])
                    .mark_area(opacity=0.5, color='orange').encode(x="age:Q", y="density:Q")
                    .properties(width=350, height=200, title="Density Plot: Age (Scaled)"))
        st.altair_chart(chart_new, use_container_width=True) 

    # 6. Correlation Heatmap
    st.write('### 3. Korelasi antar Kolom Numerik')
    corr = df[numbers].corr().reset_index().melt('index')
    corr.columns = ['Variable1', 'Variable2', 'Correlation']
    
    heatmap = (alt.Chart(corr).mark_rect().encode(
                x=alt.X('Variable2:N', title=None),
                y=alt.Y('Variable1:N', title=None),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['Variable1', 'Variable2', alt.Tooltip('Correlation:Q', format='.2f')]
            ).properties(width=400, height=400, title="Correlation Heatmap"))
    
    text = (alt.Chart(corr).mark_text(fontSize=12, color='black')
            .encode(x='Variable2:N', y='Variable1:N', text=alt.Text('Correlation:Q', format=".2f")))
    st.altair_chart(heatmap + text, use_container_width=True)

    # 7. Train Test Split
    st.write("### 4. Train–Test Split")
    X = df_select.drop("heart_attack", axis=1)
    y = df_select["heart_attack"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Ukuran Data Training: {len(X_train)} baris, Ukuran Data Testing: {len(X_test)} baris")

    # 8. Handling Imbalance Class dengan SMOTE
    st.write("### 5. Handling Imbalance Class (SMOTE)")
    sm = SMOTE(random_state=42)
    X_train_balance, y_train_balance = sm.fit_resample(X_train, y_train)
    
    col_im1, col_im2 = st.columns(2)
    with col_im1:
        st.write('**Sebelum SMOTE**')
        st.write(y_train.value_counts())
    with col_im2:
        st.write('**Setelah SMOTE**')
        st.write(pd.Series(y_train_balance).value_counts())

    # 9. Pemodelan (Multi-Model Selection)
    st.write('### 6. Pemodelan & Evaluasi')
    
    # Tombol pilihan ke samping (Sidebar)
    st.sidebar.title("Pilihan Model Klasifikasi")
    selected_model = st.sidebar.radio(
        "Pilih Model untuk Dilihat Hasilnya:",
        ("Logistic Regression", "Random Forest", "Decision Tree", "K-Nearest Neighbors (KNN)", "Naive Bayes")
    )

    # Inisialisasi Model berdasarkan pilihan
    if selected_model == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif selected_model == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif selected_model == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif selected_model == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
    elif selected_model == "Naive Bayes":
        model = GaussianNB()

    # Training Proses
    st.write(f"**Sedang Memproses Model: {selected_model}...**")
    model.fit(X_train_balance, y_train_balance)
    
    train_accuracy = model.score(X_train_balance, y_train_balance)
    st.success(f"Akurasi Training ({selected_model}): {round(train_accuracy * 100, 2)}%")

    # Koefisien atau Feature Importance (Jika tersedia)
    feature_names = X_train_balance.columns
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({"Feature": feature_names, "Value": model.coef_[0]}).sort_values(by="Value", ascending=False)
        st.write(f"**Koefisien Model {selected_model}**")
        st.dataframe(coef_df)
    elif hasattr(model, 'feature_importances_'):
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
        st.write(f"**Feature Importance {selected_model}**")
        st.dataframe(feat_df)

    # 10. Evaluasi Model
    st.write(f'### 7. Evaluasi Model {selected_model} (Data Testing)')
    y_pred = model.predict(X_test)
    
    # Cek probabilitas untuk ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred # Untuk model yang tidak dukung proba secara langsung

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=['Pred No', 'Pred Yes'], index=['Actual No', 'Actual Yes']).reset_index().melt(id_vars='index')
        cm_df.columns = ['Actual', 'Predicted', 'Count']
        
        cm_chart = (alt.Chart(cm_df).mark_rect().encode(
            x='Predicted:N', y='Actual:N', color=alt.Color('Count:Q', scale=alt.Scale(scheme='greens'))
        ))
        cm_text = cm_chart.mark_text(color="black").encode(text='Count:Q')
        st.altair_chart(cm_chart + cm_text, use_container_width=True)

    with metrics_col2:
        st.metric("Akurasi", f"{round(accuracy_score(y_test, y_pred)*100,2)}%")
        st.metric("Recall", f"{round(recall_score(y_test, y_pred)*100,2)}%")
        st.metric("Precision", f"{round(precision_score(y_test, y_pred)*100,2)}%")
        st.metric("ROC AUC", f"{round(roc_auc_score(y_test, y_proba)*100,2)}%")

    # 11. Simpan Model Pilihan
    filename = f"model_{selected_model.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    joblib.dump(feature_names, "model_features.pkl")
    joblib.dump(numbers, "numeric_columns.pkl")
    st.info(f"Model {selected_model} telah disimpan sebagai {filename}")

if __name__ == "__main__":
    ml_model()