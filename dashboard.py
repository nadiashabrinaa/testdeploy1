# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import tempfile
import os

# --- Judul ---
st.set_page_config(page_title="Dashboard Klasifikasi & Deteksi", layout="wide")
st.title("📊 Dashboard Analisis Gambar")
st.markdown("### Klasifikasi Daun Teh & Deteksi Objek Makanan")

# --- Load Model ---
@st.cache_resource
def load_models():
    try:
        klasifikasi_model = tf.keras.models.load_model("nadia_shabrina_Laporan2.h5")
        st.success("✅ Model Klasifikasi (daun teh) berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model .h5: {e}")
        klasifikasi_model = None

    try:
        deteksi_model = YOLO("Nadia_Laporan 4.pt")
        st.success("✅ Model Deteksi (makanan) berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model .pt: {e}")
        deteksi_model = None

    return klasifikasi_model, deteksi_model


klasifikasi_model, deteksi_model = load_models()

# --- Sidebar ---
st.sidebar.title("🔍 Pilih Mode")
mode = st.sidebar.radio("Mode Analisis:", ["Klasifikasi Daun Teh", "Deteksi Objek Makanan"])

uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    if mode == "Klasifikasi Daun Teh" and klasifikasi_model:
        st.subheader("📗 Hasil Klasifikasi Daun Teh")

        # Preprocessing sederhana
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        # Prediksi
        pred = klasifikasi_model.predict(img_array)
        label_idx = np.argmax(pred)
        confidence = np.max(pred) * 100

        label_dict = {0: "Daun Teh Hijau", 1: "Daun Teh Hitam", 2: "Daun Teh Oolong"}
        label = label_dict.get(label_idx, "Tidak diketahui")

        st.success(f"**Prediksi:** {label}")
        st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")

    elif mode == "Deteksi Objek Makanan" and deteksi_model:
        st.subheader("🍱 Hasil Deteksi Objek Makanan")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file.name)
            results = deteksi_model(temp_file.name)

        # Simpan hasil dan tampilkan
        result_img_path = results[0].save(filename="result.jpg")
        st.image("result.jpg", caption="Hasil Deteksi", use_container_width=True)

    else:
        st.warning("⚠️ Model belum dimuat dengan benar.")
else:
    st.info("Silakan unggah gambar untuk mulai analisis.")
