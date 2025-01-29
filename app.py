import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Judul aplikasi
st.title("Prediksi Diabetes dengan Machine Learning")
st.write(
    "Masukkan data pasien untuk mendapatkan prediksi apakah pasien menderita diabetes atau tidak."
)

# Input fitur dari pengguna
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glukosa", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input(
    "Ketebalan Kulit", min_value=0, max_value=100, value=20
)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f"
)
age = st.number_input("Usia", min_value=0, max_value=120, value=30)

# Nama fitur yang sesuai dengan model saat training
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Tombol prediksi
if st.button("Prediksi"):
    # Membuat DataFrame dengan nama fitur yang sama seperti saat model dilatih
    input_data = pd.DataFrame(
        [
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
            ]
        ],
        columns=feature_names,
    )

    # Standarisasi data
    input_data_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_data_scaled)
    result = "Positif Diabetes" if prediction[0] == 1 else "Negatif Diabetes"

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"Pasien diprediksi: **{result}**")
