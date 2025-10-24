import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import os

# =========================================================
# 📊 AI Dashboard: Klasifikasi Penyakit Daun Teh & Deteksi Makanan
# =========================================================

st.set_page_config(page_title="AI Vision Dashboard", layout="wide")

# ---------------------------------------------------------
# 🔧 Cek & Import YOLO jika tersedia
# ---------------------------------------------------------
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

# ---------------------------------------------------------
# 🧠 Muat Model CNN untuk Klasifikasi Penyakit Daun Teh
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

# ---------------------------------------------------------
# ⚙️ Fungsi Prediksi Daun Teh
# ---------------------------------------------------------
def predict_tea_leaf(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = cnn_model.predict(img_array)
    labels = ['Algal Leaf Spot', 'Anthracnose', 'Bird Eye Spot', 'Brown Blight', 'Gray Blight', 'Healthy']
    pred_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return pred_label, confidence

# ---------------------------------------------------------
# ⚙️ Fungsi Deteksi Makanan (YOLO)
# ---------------------------------------------------------
def detect_food(img):
    if not yolo_available:
        st.error("YOLO tidak tersedia: Ultralytics (YOLO) belum terinstal.")
        return None

    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Model YOLO (.pt) belum ditemukan di repositori.")
        return None

    model = YOLO(model_path)
    results = model.predict(img)
    return results

# ---------------------------------------------------------
# 🧭 Sidebar Navigasi
# ---------------------------------------------------------
st.sidebar.title("🔍 Pilih Mode Analisis")
mode = st.radio("Mode Analisis:", ["🏠 Halaman Selamat Datang", "🌿 Klasifikasi Penyakit Daun Teh", "🍽️ Deteksi Jenis Makanan"])

# =========================================================
# 🏠 Halaman Selamat Datang
# =========================================================
if mode == "🏠 Halaman Selamat Datang":
    st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🏠 Selamat Datang di AI Vision Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:justify; font-size:18px;'>
    <p>Selamat datang di <b>AI Vision Dashboard</b> — platform interaktif berbasis kecerdasan buatan 
    untuk melakukan <b>analisis citra secara otomatis</b>.</p>

    <p>Melalui dashboard ini, kamu dapat melakukan dua hal menarik:</p>
    <ul>
        <li>🌿 <b>Klasifikasi Penyakit Daun Teh</b> — Mengidentifikasi jenis penyakit daun teh berdasarkan citra digital menggunakan model Deep Learning.</li>
        <li>🍽️ <b>Deteksi Jenis Makanan</b> — Mendeteksi berbagai objek makanan (meal, dessert, drink) menggunakan model YOLOv8.</li>
    </ul>

    <p>Dashboard ini cocok untuk keperluan <b>penelitian, pembelajaran AI,</b> maupun <b>eksperimen visualisasi data</b> 
    dengan pendekatan Machine Learning modern.</p>

    <p style='text-align:center; color:#2E8B57; font-size:20px;'><b>✨ Jelajahi, Eksperimen, dan Temukan Insight dari Gambar! ✨</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.image("https://cdn.pixabay.com/photo/2021/06/24/12/45/ai-6358895_1280.jpg", use_container_width=True)

# =========================================================
# 🌿 Klasifikasi Penyakit Daun Teh
# =========================================================
elif mode == "🌿 Klasifikasi Penyakit Daun Teh":
    st.header("🌿 Klasifikasi Penyakit Daun Teh")
    uploaded_file = st.file_uploader("Unggah gambar daun teh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and cnn_model is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar Diupload", width=300)
        pred_label, confidence = predict_tea_leaf(img)
        st.success(f"Prediksi: {pred_label} (Confidence: {confidence:.2f})")

# =========================================================
# 🍽️ Deteksi Jenis Makanan
# =========================================================
elif mode == "🍽️ Deteksi Jenis Makanan":
    st.header("🍽️ Deteksi Jenis Makanan dengan YOLOv8")

    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])
    if uploaded_food is not None:
        img = Image.open(uploaded_food).convert("RGB")
        st.image(img, caption="Gambar Diupload", width=300)

        if yolo_available:
            results = detect_food(img)
            if results:
                st.image(results[0].plot(), caption="Hasil Deteksi YOLOv8", use_container_width=True)
        else:
            st.warning("Fungsi deteksi YOLO tidak tersedia — periksa apakah paket ultralytics terinstal dan file .pt berada di repo.")
