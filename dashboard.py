import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import os

# ---- Import library model ----
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    import tensorflow as tf
except Exception as e:
    tf = None

# ---- Konfigurasi halaman ----
st.set_page_config(page_title="AI Vision Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧠 AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.write("🌿 Klasifikasi Daun Teh  •  🍽️ Deteksi Jenis Makanan")

# ---- Nama file model ----
KERAS_MODEL_PATH = "model_uts/nadia_shabrina_Laporan2.h5"
YOLO_MODEL_PATH = "model_uts/Nadia_Laporan 4.pt"
TEA_CLASSES = ["Green Tea", "Black Tea", "White Tea"]
IMAGE_SIZE = (224, 224)

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Pilih Mode")
    mode = st.radio("Pilih Mode:", ["🌿 Klasifikasi Daun Teh", "🍽️ Deteksi Objek Makanan"])
    st.markdown("---")
    conf_thresh = st.slider("Confidence Threshold (YOLO)", 0.0, 1.0, 0.45, 0.01)
    st.markdown("---")
    st.info("Pastikan file model disimpan di folder utama repository.")

# ---- Fungsi bantu ----
@st.cache_resource
def load_keras_model():
    if tf is None:
        st.error("TensorFlow belum terinstal.")
        return None
    return tf.keras.models.load_model(KERAS_MODEL_PATH)

@st.cache_resource
def load_yolo_model():
    if YOLO is None:
        st.error("Ultralytics (YOLO) belum terinstal.")
        return None
    return YOLO(YOLO_MODEL_PATH)

def preds_to_df(preds, class_names):
    return pd.DataFrame({"Class": class_names, "Confidence": preds})

# ---- Mode 1: Klasifikasi Daun Teh ----
if mode == "🌿 Klasifikasi Daun Teh":
    st.subheader("🌿 Klasifikasi Jenis Daun Teh")
    uploaded_img = st.file_uploader("Unggah gambar daun teh", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        try:
            model = load_keras_model()
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            model = None

        if model is not None:
            if model is not None:
    # Ambil ukuran input dari model agar sesuai otomatis
    input_shape = model.input_shape[1:3]  # contoh: (224, 224)
    channels = model.input_shape[3] if len(model.input_shape) == 4 else 3

    img_resized = image.resize(input_shape)
    arr = np.array(img_resized) / 255.0

    # Jika model hanya butuh grayscale
    if channels == 1:
        arr = np.mean(arr, axis=-1, keepdims=True)

    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    # Prediksi
    preds = model.predict(arr)[0]
    if preds.sum() == 0 or (preds.max() > 1.0):
        preds = np.exp(preds) / np.sum(np.exp(preds))
    top_idx = int(np.argmax(preds))
    label = TEA_CLASSES[top_idx] if top_idx < len(TEA_CLASSES) else f"Class {top_idx}"

    st.success(f"**Prediksi: {label}** (Confidence: {preds[top_idx]:.3f})")
    df = preds_to_df(preds, TEA_CLASSES)
    st.bar_chart(df.set_index("Class"))

            if preds.sum() == 0 or (preds.max() > 1.0):
                preds = np.exp(preds) / np.sum(np.exp(preds))
            top_idx = int(np.argmax(preds))
            label = TEA_CLASSES[top_idx] if top_idx < len(TEA_CLASSES) else f"Class {top_idx}"

            st.success(f"**Prediksi: {label}** (Confidence: {preds[top_idx]:.3f})")
            df = preds_to_df(preds, TEA_CLASSES)
            st.bar_chart(df.set_index("Class"))

# ---- Mode 2: Deteksi Objek Makanan ----
else:
    st.subheader("🍽️ Deteksi Jenis Makanan (YOLO)")
    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])

    if uploaded_food:
        image = Image.open(uploaded_food).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        try:
            model_yolo = load_yolo_model()
        except Exception as e:
            st.error(f"Gagal memuat model YOLO: {e}")
            model_yolo = None

        if model_yolo is not None:
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
                st.warning("Tidak ada objek terdeteksi.")
