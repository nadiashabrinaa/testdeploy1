import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="AI Vision Dashboard", layout="wide")

# ---------------------------------------------------------
# Cek dan import YOLO
# ---------------------------------------------------------
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

# ---------------------------------------------------------
# Load CNN model daun teh
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    model_path = "tea_leaf_disease_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.warning("Model CNN belum tersedia. Upload terlebih dahulu di repositori.")
        return None

cnn_model = load_cnn_model()

def predict_tea_leaf(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = cnn_model.predict(img_array)
    labels = ['Algal Leaf Spot', 'Anthracnose', 'Bird Eye Spot', 'Brown Blight', 'Gray Blight', 'Healthy']
    return labels[np.argmax(prediction)], np.max(prediction)

def detect_food(img):
    if not yolo_available:
        st.error("YOLO tidak tersedia: Ultralytics (YOLO) belum terinstal.")
        return None
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Model YOLO (.pt) belum ditemukan di repositori.")
        return None
    model = YOLO(model_path)
    return model.predict(img)

# =========================================================
# 🏠 Halaman Utama
# =========================================================
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🏠 Selamat Datang di AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:justify; font-size:18px;'>
<p>Selamat datang di <b>AI Vision Dashboard</b> — platform interaktif berbasis kecerdasan buatan untuk 
melakukan <b>analisis citra</b>.</p>
<p>Klik tombol di bawah untuk memulai analisis:</p>
</div>
""", unsafe_allow_html=True)

start = st.button("🚀 Mulai Analisis", use_container_width=True)

# =========================================================
# 🌿 / 🍽️ Mode Analisis (muncul setelah klik tombol)
# =========================================================
if start:
    st.markdown("### Pilih Mode Analisis:")
    mode = st.radio("", ["🌿 Klasifikasi Penyakit Daun Teh", "🍽️ Deteksi Jenis Makanan"], horizontal=True)

    if mode == "🌿 Klasifikasi Penyakit Daun Teh":
        st.header("🌿 Klasifikasi Penyakit Daun Teh")
        uploaded = st.file_uploader("Unggah gambar daun teh", type=["jpg", "jpeg", "png"])
        if uploaded and cnn_model:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar Diupload", width=300)
            label, conf = predict_tea_leaf(img)
            st.success(f"Prediksi: {label} (Confidence: {conf:.2f})")

    elif mode == "🍽️ Deteksi Jenis Makanan":
        st.header("🍽️ Deteksi Jenis Makanan dengan YOLOv8")
        uploaded = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar Diupload", width=300)
            if yolo_available:
                results = detect_food(img)
                if results:
                    st.image(results[0].plot(), caption="Hasil Deteksi YOLOv8", use_container_width=True)
            else:
                st.warning("Fungsi deteksi YOLO tidak tersedia — periksa apakah paket ultralytics terinstal dan file .pt berada di repo.")
