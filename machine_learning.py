import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
import joblib


def ml_model():

    st.title("Analisis Komparatif Model Klasifikasi Risiko Serangan Jantung")

    # ==============================
    # 1. Load Dataset
    # ==============================
    df = pd.read_csv("Data3.csv")
    df = df.drop(columns=["patient_id"])

    df["body_mass_index"].fillna(df["body_mass_index"].median(), inplace=True)
    df["systolic_blood_pressure"].fillna(df["systolic_blood_pressure"].median(), inplace=True)
    df["forced_expiratory_volume_1"].fillna(df["forced_expiratory_volume_1"].median(), inplace=True)

    numeric_cols = [
        "age",
        "body_mass_index",
        "systolic_blood_pressure",
        "forced_expiratory_volume_1"
    ]

    # ==============================
    # 2. Outlier Detection (IQR)
    # ==============================
    st.subheader("1. Deteksi & Pembersihan Outlier")

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    df_clean = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                     (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.write(f"Jumlah data awal: **{df.shape[0]}**")
    st.write(f"Setelah pembersihan outlier: **{df_clean.shape[0]}**")

    st.dataframe(df_clean.head())

    # ==============================
    # 3. Encoding & Normalisasi
    # ==============================
    st.subheader("2. Encoding & Normalisasi")

    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    scaler = MinMaxScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    st.dataframe(df_encoded.head())

    # ==============================
    # 4. Correlation Heatmap
    # ==============================
    st.subheader("3. Korelasi Antar Fitur Numerik")

    corr = df_clean[numeric_cols].corr().reset_index().melt("index")
    corr.columns = ["Variabel_X", "Variabel_Y", "Korelasi"]

    heatmap = alt.Chart(corr).mark_rect().encode(
        x="Variabel_Y:N",
        y="Variabel_X:N",
        color=alt.Color("Korelasi:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Variabel_X", "Variabel_Y", alt.Tooltip("Korelasi:Q", format=".2f")]
    ).properties(
        width=400,
        height=400,
        title="Correlation Heatmap"
    )

    text = alt.Chart(corr).mark_text(color="black").encode(
        x="Variabel_Y:N",
        y="Variabel_X:N",
        text=alt.Text("Korelasi:Q", format=".2f")
    )

    st.altair_chart(heatmap + text, use_container_width=True)

    # ==============================
    # 5. Train Test Split
    # ==============================
    st.subheader("4. Train Test Split")

    X = df_encoded.drop("heart_attack", axis=1)
    y = df_encoded["heart_attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.write(f"Data training: **{len(X_train)}**")
    st.write(f"Data testing: **{len(X_test)}**")

    # ==============================
    # 6. SMOTE
    # ==============================
    st.subheader("5. Penanganan Imbalance Class (SMOTE)")

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Sebelum SMOTE")
        st.write(y_train.value_counts())
    with col_b:
        st.write("Setelah SMOTE")
        st.write(pd.Series(y_train_sm).value_counts())

    # ==============================
    # 7. Sidebar Model Selection (FIXED KEY)
    # ==============================
    st.sidebar.title("Pemilihan Model")

    selected_model = st.sidebar.radio(
        "Pilih Model Klasifikasi",
        (
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "KNN",
            "Naive Bayes"
        ),
        key="radio_model_selector"  # <<< FIX ERROR
    )

    if selected_model == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif selected_model == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif selected_model == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif selected_model == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = GaussianNB()

    # ==============================
    # 8. Training
    # ==============================
    st.subheader("6. Training Model")

    model.fit(X_train_sm, y_train_sm)

    train_acc = model.score(X_train_sm, y_train_sm)
    st.success(f"Akurasi Training: **{train_acc*100:.2f}%**")

    # ==============================
    # 9. Evaluasi Model
    # ==============================
    st.subheader("7. Evaluasi Model (Data Testing)")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    else:
        roc = 0.0

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Pred No", "Pred Yes"]
        )
        st.write("Confusion Matrix")
        st.dataframe(cm_df)

    with col2:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
        st.metric("Precision", f"{precision_score(y_test, y_pred)*100:.2f}%")
        st.metric("Recall", f"{recall_score(y_test, y_pred)*100:.2f}%")
        st.metric("ROC AUC", f"{roc*100:.2f}%")

    # ==============================
    # 10. Save Model
    # ==============================
    joblib.dump(model, f"model_{selected_model.replace(' ', '_').lower()}.pkl")
    joblib.dump(X.columns, "model_features.pkl")

    st.info("Model berhasil disimpan")


if __name__ == "__main__":
    ml_model()
