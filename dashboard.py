import streamlit as st
import numpy as np
from PIL import Image
import os

# Coba impor YOLO dan OpenCV
try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    st.error(f"⚠️ Modul belum terinstal dengan benar: {e}")
    st.stop()

# Judul Aplikasi
st.title("🚀 Dashboard Deteksi Objek YOLO")

# Upload Model
model_path = "nadia_shabrina_Laporan2.h5"

if not os.path.exists(model_path):
    st.warning("⚠️ File model tidak ditemukan. Pastikan file 'nadia_shabrina_Laporan2.h5' sudah diunggah ke folder yang sama.")
else:
    st.success("✅ Model berhasil dimuat!")

# Upload Gambar
uploaded_file = st.file_uploader("📸 Unggah gambar untuk deteksi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_container_width=True)

    # Jalankan prediksi YOLO jika model tersedia
    try:
        model = YOLO(model_path)
        st.write("🔍 Melakukan deteksi...")
        results = model.predict(source=np.array(image), conf=0.5)
        result_image = results[0].plot()  # hasil deteksi
        st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai!")
    except Exception as e:
        st.error(f"❌ Gagal menjalankan model: {e}")
else:
    st.info("📤 Silakan unggah gambar terlebih dahulu.")
