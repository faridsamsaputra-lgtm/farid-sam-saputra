import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

def chart():
    # 1. Membaca Dataset (Data3.csv)
    df = pd.read_csv('Data3.csv')
    
    # Pre-processing untuk menyesuaikan dengan struktur kode asli
    # Mengganti nama kolom agar sesuai logika visualisasi
    df['stroke'] = df['heart_attack']
    df['bmi'] = df['body_mass_index']
    df['avg_glucose_level'] = df['systolic_blood_pressure'] # Menggunakan Tekanan Darah sebagai pengganti Glukosa
    
    # Membuat kolom kategori pendukung yang dibutuhkan visualisasi asli
    def get_bmi_cat(bmi):
        if bmi < 18.5: return 'Underweight'
        elif 18.5 <= bmi < 25: return 'Normal'
        elif 25 <= bmi < 30: return 'Overweight'
        else: return 'Obesity'
    
    def get_age_cat(age):
        if age < 30: return 'Youth'
        elif 30 <= age < 60: return 'Adult'
        else: return 'Senior'

    def get_bp_cat(bp):
        if bp < 120: return 'Normal'
        elif 120 <= bp < 140: return 'Pre-Hypertension'
        else: return 'Hypertension'

    df['bmi_category'] = df['bmi'].apply(lambda x: get_bmi_cat(x) if pd.notnull(x) else 'Unknown')
    df['age_category'] = df['age'].apply(get_age_cat)
    df['glucose_category'] = df['avg_glucose_level'].apply(lambda x: get_bp_cat(x) if pd.notnull(x) else 'Unknown')
    
    # Menghitung Risk Score sederhana (akumulasi faktor risiko)
    df['risk_score'] = df['smoker'] + df['hypertension_treated'] + df['diabetes'] + df['atrial_fibrillation']
    
    # Kolom pembanding tambahan
    df['smoking_status'] = df['smoker'].map({0: 'Never Smoked', 1: 'Smoker'})
    df['Residence_type'] = df['family_history_of_cardiovascular_disease'].map({0: 'No Family History', 1: 'Family History'})
    df['work_type'] = df['diabetes'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    df['lifestyle_risk'] = df['chronic_obstructive_pulmonary_disorder'].map({0: 'Low COPD Risk', 1: 'High COPD Risk'})
    df['hypertension'] = df['hypertension_treated']

    pasien_count = df.shape[0]
    pasien_stroke = df['stroke'].sum()
    stroke_rate = (pasien_stroke / pasien_count) * 100

    # Card Metrics dan Button Filter
    col1, col2, col3, col4, col5, col6 = st.columns([2,2,3,2,2,1])
    with col1:
        st.metric(label="Total Pasien", value = pasien_count)
    with col2:
        st.metric(label="Pasien Heart Attack", value = pasien_stroke)
    with col3:
        st.metric(label="Persentase", value = f"{stroke_rate:.2f}%")
    
    # Initialize session state untuk filter
    if 'selected_gender' not in st.session_state:
        st.session_state.selected_gender = None
    if 'selected_stroke' not in st.session_state:
        st.session_state.selected_stroke = None
    
    with col4:
        st.write('**Jenis Kelamin**')
        if st.button("Laki-laki"):
            st.session_state.selected_gender = 'M'
        if st.button("Perempuan"):
            st.session_state.selected_gender = 'F'
    with col5:
        st.write('**Status**')
        if st.button("Heart Attack"):
            st.session_state.selected_stroke = 1
        if st.button("No Heart Attack"):
            st.session_state.selected_stroke = 0
    
    # Reset button
    with col6:
        if st.button("🔄"):
            st.session_state.selected_gender = None
            st.session_state.selected_stroke = None
            st.rerun()
    
    # Apply filter
    filtered_df = df.copy()
    if st.session_state.selected_gender:
        filtered_df = filtered_df[filtered_df['gender'] == st.session_state.selected_gender]
    if st.session_state.selected_stroke is not None:
        filtered_df = filtered_df[filtered_df['stroke'] == st.session_state.selected_stroke]
    
    st.dataframe(df.head(5))
    
    # Pie Chart
    col1, col2 = st.columns([5,5])
    def pie_chart_from_counts(counts_df, label_col, value_col, title):
        chart = alt.Chart(counts_df).mark_arc(innerRadius=40).encode(
            theta=alt.Theta(field=value_col, type='quantitative'),
            color=alt.Color(field=label_col, type='nominal', sort=alt.EncodingSortField(field=value_col, order='descending')),
            tooltip=[alt.Tooltip(label_col+':N'), alt.Tooltip(value_col+':Q')]
        ).properties(height=300, title=title)
        return chart

    with col1:
        # Persentase Jenis Kelamin
        gender_counts = filtered_df['gender'].fillna('Unknown').value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        chart_gender = pie_chart_from_counts(gender_counts, 'gender', 'count', 'Jenis Kelamin')
        st.altair_chart(chart_gender, use_container_width=True)
    
    with col2:
        # Persentase Hipertensi
        hyper_counts = filtered_df['hypertension'].fillna(0).map({0: 'Tidak', 1: 'Hipertensi (Treated)'}).value_counts().reset_index()
        hyper_counts.columns = ['hypertension', 'count']
        chart_hyper = pie_chart_from_counts(hyper_counts, 'hypertension', 'count', 'Riwayat Hipertensi')
        st.altair_chart(chart_hyper, use_container_width=True)
    
    col1, col2 = st.columns([5,5])
    with col1:
        # Persentase Riwayat Merokok
        smoke_counts = filtered_df['smoking_status'].fillna('Unknown').value_counts().reset_index()
        smoke_counts.columns = ['smoking_status', 'count']
        chart_smoke = pie_chart_from_counts(smoke_counts, 'smoking_status', 'count', 'Riwayat Merokok')
        st.altair_chart(chart_smoke, use_container_width=True)
    
    with col2:
        # Persentase Riwayat Keluarga (Pengganti Residence Type)
        residence_counts = filtered_df['Residence_type'].fillna('Unknown').value_counts().reset_index()
        residence_counts.columns = ['Residence_type', 'count']
        chart_res = pie_chart_from_counts(residence_counts, 'Residence_type', 'count', 'Riwayat Keluarga Kardiovaskular')
        st.altair_chart(chart_res, use_container_width=True)    

    # Histogram BMI
    st.write("**Distribusi BMI Pasien**")
    bmi_chart = alt.Chart(filtered_df).mark_bar().encode(
            alt.X('bmi:Q', bin=alt.Bin(maxbins=30), title='BMI (Body Mass Index)'),
            alt.Y('count():Q', title='Frekuensi'),
            tooltip=[alt.Tooltip('count():Q', title='Frekuensi')]
        ).properties(height=300)
    st.altair_chart(bmi_chart, use_container_width=True)

    # Scatter Plot
    st.write("**Hubungan BMI & Tekanan Darah Sistolik**")
    scatter1 = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('avg_glucose_level:Q', title='Tekanan Darah Sistolik'),
            y=alt.Y('bmi:Q', title='BMI'),
            color=alt.Color('stroke:N', title='Heart Attack', scale=alt.Scale(scheme='viridis')),
            tooltip=['age', 'bmi', 'avg_glucose_level', 'stroke']
        ).interactive().properties(height=320)
    st.altair_chart(scatter1, use_container_width=True)
    
    st.write("**Hubungan BMI & Usia**")
    scatter2 = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('age:Q', title='Usia'),
            y=alt.Y('bmi:Q', title='BMI'),
            color=alt.Color('stroke:N', title='Heart Attack', scale=alt.Scale(scheme='viridis')),
            tooltip=['age', 'bmi', 'avg_glucose_level', 'stroke']
        ).interactive().properties(height=320)
    st.altair_chart(scatter2, use_container_width=True)

    # Distribusi Heart Attack dengan Box Plot
    st.write("**Distribusi Usia Pasien berdasarkan Heart Attack (Box Plot)**")
    box = alt.Chart(filtered_df).mark_boxplot(extent=1.5).encode(
            x=alt.X('stroke:N', title='Heart Attack (1=Yes, 0=No)'),
            y=alt.Y('age:Q', title='Usia'),
            color=alt.Color('stroke:N', legend=None),
            tooltip=[alt.Tooltip('count():Q', title='Jumlah')]
        ).properties(height=320)
    st.altair_chart(box, use_container_width=True)

    # Distribusi Usia berdasarkan Kasus Heart Attack
    st.write("**Distribusi Usia berdasarkan Kasus Heart Attack**")
    age_stroke_counts = filtered_df.groupby(['age', 'stroke']).size().reset_index(name='counts')
    age_stroke_counts['stroke'] = age_stroke_counts['stroke'].astype(str)
    line_age = alt.Chart(age_stroke_counts).mark_line(point=True).encode(
        x=alt.X('age:Q', title='Usia'),
        y=alt.Y('counts:Q', title='Jumlah Pasien'),
        color=alt.Color('stroke:N', title='Heart Attack'),
        tooltip=['age', 'counts', 'stroke']
    ).properties(height=350)
    st.altair_chart(line_age, use_container_width=True)

    # Perkembangan dan Perbandingan Berdasarkan Kategori
    col1, col2 = st.columns([5,5])
    with col1:
        st.write('**Perkembangan Penderita Heart Attack**')
        # Berdasarkan BMI Category
        bmi_stroke_counts = filtered_df.groupby(['bmi_category', 'stroke']).size().reset_index(name='counts')
        bmi_stroke_counts['stroke_label'] = bmi_stroke_counts['stroke'].map({0: 'No Heart Attack', 1: 'Heart Attack'}).astype(str)
        chart_bmi_cat = alt.Chart(bmi_stroke_counts).mark_line(point=True).encode(
            x=alt.X('bmi_category:N', sort=alt.EncodingSortField(field='counts', op='sum', order='descending'), title='BMI Category'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Heart Attack'),
            tooltip=['bmi_category', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_bmi_cat, use_container_width=True)

        # Berdasarkan Age Category
        age_cat_stroke_counts = filtered_df.groupby(['age_category', 'stroke']).size().reset_index(name='counts')
        age_cat_stroke_counts['stroke_label'] = age_cat_stroke_counts['stroke'].map({0: 'No Heart Attack', 1: 'Heart Attack'}).astype(str)
        chart_age_cat = alt.Chart(age_cat_stroke_counts).mark_line(point=True).encode(
            x=alt.X('age_category:N', title='Age Category'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Heart Attack'),
            tooltip=['age_category', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_age_cat, use_container_width=True)
    
        # Berdasarkan Risk Score
        risk_stroke_counts = filtered_df.groupby(['risk_score', 'stroke']).size().reset_index(name='counts')
        risk_stroke_counts['stroke_label'] = risk_stroke_counts['stroke'].map({0: 'No Heart Attack', 1: 'Heart Attack'}).astype(str)
        chart_risk = alt.Chart(risk_stroke_counts).mark_line(point=True).encode(
            x=alt.X('risk_score:O', title='Risk Score (Total Faktor Risiko)'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Heart Attack'),
            tooltip=['risk_score', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_risk, use_container_width=True)
    
    with col2:
        st.write('**Perbandingan Berdasarkan Kategori**')
        # Berdasarkan Status Diabetes (Pengganti Tipe Pekerjaan)
        work_stroke_counts = filtered_df.groupby(['work_type', 'stroke']).size().reset_index(name='counts')
        work_stroke_counts['stroke_label'] = work_stroke_counts['stroke'].map({0: 'No Heart Attack', 1: 'Heart Attack'}).astype(str)
        chart_work = alt.Chart(work_stroke_counts).mark_bar().encode(
            x=alt.X('work_type:N', title='Status Diabetes'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Heart Attack'),
            tooltip=['work_type', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_work, use_container_width=True)
    
        # Berdasarkan Kategori Tekanan Darah (Pengganti Kategori Glukosa)
        glucose_stroke_counts = filtered_df.groupby(['glucose_category', 'stroke']).size().reset_index(name='counts')
        glucose_stroke_counts['stroke_label'] = glucose_stroke_counts['stroke'].map({0: 'No Heart Attack', 1: 'Heart Attack'}).astype(str)
        chart_glucose = alt.Chart(glucose_stroke_counts).mark_bar().encode(
            x=alt.X('glucose_category:N', title='Kategori Tekanan Darah'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Heart Attack'),
            tooltip=['glucose_category', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_glucose, use_container_width=True)
    
        # Berdasarkan COPD Risk (Pengganti Lifestyle Risk)
        lifestyle_stroke_counts = filtered_df.groupby(['lifestyle_risk', 'stroke']).size().reset_index(name='counts')
        lifestyle_stroke_counts['stroke_label'] = lifestyle_stroke_counts['stroke'].map({0: 'No Heart Attack', 1: 'Heart Attack'}).astype(str)
        chart_lifestyle = alt.Chart(lifestyle_stroke_counts).mark_bar().encode(
            x=alt.X('lifestyle_risk:N', title='Resiko COPD'),
            y=alt.Y('counts:Q', title='Jumlah Pasien'),
            color=alt.Color('stroke_label:N', title='Heart Attack'),
            tooltip=['lifestyle_risk', 'counts', 'stroke_label']
        ).properties(height=300)
        st.altair_chart(chart_lifestyle, use_container_width=True)