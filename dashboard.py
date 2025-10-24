# =========================================================
# 📊 AI Dashboard: Klasifikasi Penyakit Daun Teh & Deteksi Makanan
# =========================================================
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import os

# ==== Import model library ====
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

# ==== Konfigurasi Halaman ====
st.set_page_config(page_title="AI Vision Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🤖 AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.write("🌿 Klasifikasi Penyakit Daun Teh  |  🍽️ Deteksi Jenis Makanan")

# ==== Path Model ====
MODEL_TEA_PATH = "model_uts/nadia_shabrina_Laporan2.h5"         # ganti sesuai nama file kamu
MODEL_FOOD_PATH = "Nadia_Laporan 4..pt"         # ganti sesuai file YOLO kamu

# ==== Daftar kelas ====
TEA_CLASSES = [
    "Red Leaf Spot", "Algal Leaf Spot", "Bird’s Eyespot", 
    "Gray Blight", "White Spot", "Anthracnose", 
    "Brown Blight", "Healthy Tea Leaves"
]
FOOD_CLASSES = ["Meal", "Dessert", "Drink"]

# ==== Sidebar ====
with st.sidebar:
    st.header("⚙️ Pilih Mode")
    mode = st.radio("Mode Analisis:", ["🌿 Klasifikasi Penyakit Daun Teh", "🍽️ Deteksi Jenis Makanan"])
    conf_thresh = st.slider("Confidence Threshold (untuk YOLO)", 0.1, 1.0, 0.45, 0.01)
    st.markdown("---")
    st.info("Pastikan file model (.h5 dan .pt) ada di folder yang sama dengan dashboard.py")

# ==== Fungsi Bantu ====
@st.cache_resource
def load_keras_model():
    if tf is None:
        st.error("❌ TensorFlow belum terinstal.")
        return None
    if not os.path.exists(MODEL_TEA_PATH):
        st.error("❌ File model daun teh tidak ditemukan.")
        return None
    return load_model(MODEL_TEA_PATH)

@st.cache_resource
def load_yolo_model():
    if YOLO is None:
        st.warning("⚠️ Ultralytics (YOLO) belum terinstal, fungsi deteksi dinonaktifkan.")
        return None
    if not os.path.exists(MODEL_FOOD_PATH):
        st.error("❌ File model makanan (.pt) tidak ditemukan.")
        return None
    return YOLO(MODEL_FOOD_PATH)

def preprocess_image(image, size=(224, 224)):
    img = image.resize(size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def preds_to_df(preds, classes):
    return pd.DataFrame({"Class": classes, "Confidence": preds})

# ======================================================
# 🌿 MODE 1 — KLASIFIKASI PENYAKIT DAUN TEH
# ======================================================
if mode == "🌿 Klasifikasi Penyakit Daun Teh":
    st.subheader("🌿 Deteksi Penyakit Daun Teh Berdasarkan Citra")
    uploaded_img = st.file_uploader("Unggah gambar daun teh", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        model = load_keras_model()
        if model:
            arr = preprocess_image(image)
            try:
                preds = model.predict(arr)[0]
                if preds.sum() == 0 or np.isnan(preds).any():
                    st.error("⚠️ Prediksi tidak valid — periksa format model atau data input.")
                else:
                    top_idx = np.argmax(preds)
                    label = TEA_CLASSES[top_idx]
                    conf = preds[top_idx]
                    st.success(f"**Prediksi: {label}** (Confidence: {conf:.3f})")

                    df = preds_to_df(preds, TEA_CLASSES)
                    st.bar_chart(df.set_index("Class"))
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# ======================================================
# 🍽️ MODE 2 — DETEKSI OBJEK MAKANAN
# ======================================================
else:
    st.subheader("🍽️ Deteksi Jenis Makanan (Meal, Dessert, Drink)")
    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])

    if uploaded_food:
        image = Image.open(uploaded_food).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        model_yolo = load_yolo_model()
        if model_yolo:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            results = model_yolo(tmp_path, conf=conf_thresh)
            result_img = results[0].plot()
            result_pil = Image.fromarray(result_img)
            st.image(result_pil, caption="Hasil Deteksi", use_container_width=True)

            det_data = []
            for box in results[0].boxes:
                det_data.append({
                    "Label": results[0].names[int(box.cls)],
                    "Confidence": float(box.conf),
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                })
            if det_data:
                df = pd.DataFrame(det_data)
                st.dataframe(df)
                st.bar_chart(df["Label"].value_counts())
            else:
                st.warning("⚠️ Tidak ada objek makanan terdeteksi.")
