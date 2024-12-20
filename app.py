import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

# Memuat model dan scaler yang sudah disimpan
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return None, None
    return model, scaler

# Inisialisasi model dan scaler
model, scaler = load_model_and_scaler()

# Judul Aplikasi
st.title("Prediksi Stunting dengan Model Machine Learning")

# Upload file
uploaded_file = st.file_uploader("Upload file ", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Memeriksa jenis file dan membaca data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Kolom yang digunakan untuk prediksi
    selected_columns = [
        'Pendek (anak)', 'Sangat Pendek (anak)', 'Balita Kurus dapat PMT',
        'Bayi 0-11 Bulan Imunisasi Lengkap', 'Anak Ikut PAUD',
        'Balita Dapat Zinc', 'Bumil Konseling Gizi',
        'Keluarga Bina Balita', 'RT Air Minum Layak', 'RT Sanitasi Layak'
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
        try:
            scaled_data = scaler.transform(input_data)
            st.write("Data setelah scaling:")
            st.write(scaled_data)
        except NotFittedError:
            st.error("Scaler belum di-fit. Pastikan file scaler.pkl adalah hasil fit dari data yang sesuai.")

        # Prediksi menggunakan model
        try:
            predictions = model.predict(scaled_data)
            st.write("Hasil Prediksi:", predictions)

            # Jika regresi, tampilkan unscaled predictions
            if isinstance(predictions[0], np.float64):  # Check for regression
                unscaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                st.write("Hasil Prediksi (unscaled):", unscaled_predictions.flatten())
            else:
                st.write("Hasil Prediksi tidak membutuhkan unscaled karena ini klasifikasi.")
        except NotFittedError:
            st.error("Model belum di-fit. Pastikan file model.pkl adalah hasil fit dari data yang sesuai.")

    else:
        # Jika kolom yang dibutuhkan tidak ada
        st.error("Data yang di-upload tidak memiliki kolom yang sesuai untuk prediksi.")
