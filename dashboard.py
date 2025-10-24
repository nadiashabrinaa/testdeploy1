import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import io
import os

# ----------------------------------------------------------
# Import YOLO (Ultralytics)
# ----------------------------------------------------------
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
st.sidebar.title("🧠 AI Vision Dashboard")
mode = st.radio(
    "Mode Analisis:",
    ["🏠 Halaman Selamat Datang", "🌿 Klasifikasi Penyakit Daun Teh", "🍽️ Deteksi Jenis Makanan"]
)

# ----------------------------------------------------------
# 🏠 Halaman Selamat Datang
# ----------------------------------------------------------
if mode == "🏠 Halaman Selamat Datang":
    st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🏠 Selamat Datang di AI Vision Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:justify; font-size:18px;'>
    <p>Selamat datang di <b>AI Vision Dashboard</b> — platform interaktif berbasis kecerdasan buatan 
    yang dirancang untuk melakukan <b>analisis citra</b> secara otomatis.</p>

    <p>Melalui dashboard ini, kamu dapat melakukan dua hal menarik:</p>
    <ul>
        <li>🌿 <b>Klasifikasi Penyakit Daun Teh</b> — Mengidentifikasi jenis penyakit daun teh berdasarkan citra digital menggunakan model Deep Learning.</li>
        <li>🍽️ <b>Deteksi Jenis Makanan</b> — Mendeteksi dan mengenali berbagai objek makanan (Meal, Dessert, Drink) secara otomatis menggunakan YOLOv8.</li>
    </ul>

    <p>Dashboard ini cocok untuk keperluan <b>penelitian, pembelajaran AI,</b> maupun <b>eksperimen visualisasi data</b> 
    dengan pendekatan Machine Learning modern.</p>

    <p style='text-align:center; color:#2E8B57; font-size:20px;'><b>✨ Jelajahi, Eksperimen, dan Temukan Insight dari Gambar! ✨</b></p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 🌿 Klasifikasi Penyakit Daun Teh
# ----------------------------------------------------------
elif mode == "🌿 Klasifikasi Penyakit Daun Teh":
    st.header("🌿 Klasifikasi Penyakit Daun Teh")
    st.write("Unggah gambar daun teh untuk memprediksi jenis penyakitnya menggunakan model deep learning.")

    uploaded_file = st.file_uploader("Unggah Gambar Daun Teh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_column_width=True)

        # Muat model klasifikasi daun teh
        try:
            model = tf.keras.models.load_model("tea_disease_model.h5")
            st.success("✅ Model daun teh berhasil dimuat.")
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            classes = ["Anthracnose", "Algal Leaf", "Bird Eye Spot", "Brown Blight", "Gray Blight", "Healthy"]
            result = classes[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.markdown(f"### 🩺 Prediksi: **{result}** (Confidence: {confidence:.3f})")
        except Exception as e:
            st.error(f"Gagal memuat model daun teh: {e}")

# ----------------------------------------------------------
# 🍽️ Deteksi Jenis Makanan (YOLO)
# ----------------------------------------------------------
elif mode == "🍽️ Deteksi Jenis Makanan":
    st.header("🍽️ Deteksi Jenis Makanan")
    st.write("Unggah gambar makanan untuk mendeteksi jenis objek menggunakan model YOLOv8.")

    uploaded_food = st.file_uploader("Unggah Gambar Makanan", type=["jpg", "jpeg", "png"])

    if not yolo_available:
        st.error("⚠️ YOLO tidak tersedia: Ultralytics (YOLO) belum terinstal.")
        st.info("Pastikan paket `ultralytics` ada di file requirements.txt dan sudah diinstal di server.")
    else:
        if uploaded_food is not None:
            image = Image.open(uploaded_food)
            st.image(image, caption="Gambar yang Diupload", use_column_width=True)

            # Coba muat model YOLO
            try:
                model_path = "yolov8n.pt"
                if not os.path.exists(model_path):
                    from ultralytics import YOLO
                    st.info("Mengunduh model YOLOv8n.pt...")
                    model = YOLO("yolov8n.pt")  # otomatis download
                else:
                    model = YOLO(model_path)

                results = model(image)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Hasil Deteksi YOLO", use_column_width=True)

                st.markdown("### 🍱 Objek Terdeteksi:")
                for box in results[0].boxes.data.tolist():
                    cls = int(box[5])
                    conf = box[4]
                    label = results[0].names[cls]
                    st.write(f"- {label} ({conf:.2f})")

            except Exception as e:
                st.error(f"Fungsi deteksi YOLO tidak tersedia — periksa apakah paket ultralytics terinstal dan file .pt berada di repo.\n\n**Error:** {e}")
