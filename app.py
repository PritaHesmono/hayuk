import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Memuat model dan scaler yang sudah disimpan
@st.cache_resource
def load_model_and_scaler():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Inisialisasi model dan scaler
model, scaler = load_model_and_scaler()

# Judul Aplikasi
st.title("Prediksi Model Machine Learning dengan Data Excel")

# Upload file Excel
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Membaca data dari file Excel
    df = pd.read_excel(uploaded_file)

    # Kolom yang digunakan untuk prediksi
    selected_columns = [
        # Kolom untuk layanan_balita
        'Bayi usia kurang dari 6 bulan mendapat air susu ibu (ASI) eksklusif',
        'Anak usia 6-23 bulan yang mendapat Makanan Pendamping Air Susu Ibu (MP-ASI)',
        'Anak berusia di bawah lima tahun (balita) gizi buruk yang mendapat pelayanan tata laksana gizi buruk',
        'Anak berusia di bawah lima tahun (balita) yang dipantau pertumbuhan dan perkembangannya',
        'Anak berusia di bawah lima tahun (balita) gizi kurang yang mendapat tambahan asupan gizi',
        'Balita yang memperoleh imunisasi dasar lengkap',

        # Kolom untuk layanan_ibu_hamil
        'Ibu hamil Kurang Energi Kronik (KEK) yang mendapatkan tambahan asupan gizi',
        'Ibu hamil yang mengonsumsi Tablet Tambah Darah (TTD) minimal 90 tablet selama masa kehamilan',

        # Kolom untuk layanan_sosial_gizi
        'Kelompok Keluarga Penerima Manfaat (KPM) Program Keluarga Harapan (PKH) yang mengikuti Pertemuan Peningkatan Kemampuan Keluarga (P2K2) dengan modul kesehatan dan gizi',
        'Keluarga Penerima Manfaat (KPM) dengan ibu hamil, ibu menyusui, dan baduta yang menerima variasi bantuan pangan selain beras dan telur'
    ]

    # Memastikan kolom yang dibutuhkan ada di DataFrame
    if all(col in df.columns for col in selected_columns):
        st.write("Kolom yang ditemukan:", df.columns.tolist())

        # Menampilkan opsi untuk memilih fitur yang akan digunakan untuk prediksi
        features = st.multiselect("Pilih fitur untuk prediksi", selected_columns, default=selected_columns)
        st.write("Fitur yang dipilih:", features)

        # Mengambil data fitur yang relevan
        input_data = df[features].values

        # Scaling data
        scaled_data = scaler.transform(input_data)
        st.write("Data setelah scaling:")
        st.write(scaled_data)

        # Prediksi menggunakan model
        predictions = model.predict(scaled_data)
        st.write("Hasil Prediksi (scaled):", predictions)

        # Unscaling data (untuk menampilkan hasil asli)
        unscaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        st.write("Hasil Prediksi (unscaled):", unscaled_predictions.flatten())

    else:
        # Jika kolom yang dibutuhkan tidak ada
        st.error("Data yang di-upload tidak memiliki kolom yang sesuai untuk prediksi.")