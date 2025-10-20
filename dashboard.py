import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

# Try importing YOLO & TensorFlow
try:
    from ultralytics import YOLO
except:
    YOLO = None

try:
    import tensorflow as tf
except:
    tf = None

st.set_page_config(page_title="Object Detection Dashboard", layout="wide")
st.title("🎯 Object Detection Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    model_option = st.selectbox("Pilih Model", ["YOLO (.pt)", "Keras (.h5)"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    uploaded_image = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

# Load model sesuai pilihan
@st.cache_resource
def load_yolo_model():
    return YOLO("models/Nadia_Laporan4.pt")

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model("models/nadia_shabrina_Laporan2.h5")

# Jalankan prediksi jika gambar diunggah
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼️ Gambar Input", use_container_width=True)

    if model_option == "YOLO (.pt)" and YOLO:
        st.write("🔄 Memuat model YOLO...")
        model = load_yolo_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name, conf=confidence_threshold)
        
        st.image(results[0].plot(), caption="Hasil Deteksi YOLO", use_container_width=True)
        st.subheader("📋 Objek Terdeteksi")
        for box in results[0].boxes:
            st.write(f"- {model.names[int(box.cls)]} ({float(box.conf):.2f})")

    elif model_option == "Keras (.h5)" and tf:
        st.write("🔄 Memuat model Keras...")
        model = load_keras_model()
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        preds = model.predict(img_array)
        st.subheader("📊 Output Prediksi")
        st.write(preds)

    else:
        st.warning("❌ Model belum dapat dimuat. Pastikan dependensi sudah terinstal.")
