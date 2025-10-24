import numpy as np
from tensorflow.keras.preprocessing import image

# Pastikan ukuran gambar sesuai dengan ukuran input model
img = image.load_img(uploaded_file, target_size=(224, 224))  # sesuaikan dengan model kamu
arr = image.img_to_array(img)

# Pastikan array punya 3 channel (RGB)
if arr.shape[-1] != 3:
    arr = np.stack((arr,)*3, axis=-1)

# Tambahkan dimensi batch
arr = np.expand_dims(arr, axis=0)

# Pastikan tipe data sesuai
arr = arr.astype('float32') / 255.0

# Prediksi
preds = model.predict(arr)[0]

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# ---- Konfigurasi halaman ----
st.set_page_config(page_title="Klasifikasi Daun Teh", layout="centered")
st.markdown("<h1 style='text-align:center;'>🌿 Dashboard Klasifikasi Penyakit Daun Teh</h1>", unsafe_allow_html=True)
st.write("Unggah gambar daun teh untuk mengetahui jenis penyakit atau kondisi daunnya.")

# ---- Path model ----
MODEL_PATH = "model_uts/nadia_shabrina_Laporan2.h5"

# ---- Daftar kelas daun teh ----
TEA_CLASSES = [
    "Healthy Tea Leaves",
    "Red Leaf Spot",
    "Algal Leaf Spot",
    "Bird’s Eyespot",
    "Gray Blight",
    "White Spot",
    "Anthracnose",
    "Brown Blight"
]

IMAGE_SIZE = (224, 224)

# ---- Fungsi untuk memuat model ----
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# ---- Fungsi preprocessing ----
def preprocess_image(image):
    img_resized = image.resize(IMAGE_SIZE)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# ---- Konversi prediksi ke DataFrame ----
def preds_to_df(preds, class_names):
    return pd.DataFrame({
        "Class": class_names,
        "Confidence": np.round(preds * 100, 2)
    })

# ---- Upload gambar ----
uploaded_img = st.file_uploader("📤 Unggah gambar daun teh", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="📷 Gambar yang diunggah", use_container_width=True)

    # Muat model
    model = load_model()

    if model:
        arr = preprocess_image(image)
        preds = model.predict(arr)[0]

        # Jika output belum berbentuk probabilitas
        if preds.sum() == 0 or preds.max() > 1.0:
            preds = np.exp(preds) / np.sum(np.exp(preds))

        # Ambil hasil prediksi tertinggi
        top_idx = int(np.argmax(preds))
        label = TEA_CLASSES[top_idx]
        confidence = preds[top_idx] * 100

        st.success(f"🧠 **Prediksi:** {label}")
        st.info(f"📊 Tingkat keyakinan: {confidence:.2f}%")

        # Tampilkan hasil dalam bentuk tabel dan grafik
        df = preds_to_df(preds, TEA_CLASSES)
        st.dataframe(df)
        st.bar_chart(df.set_index("Class"))

else:
    st.warning("Silakan unggah gambar daun teh terlebih dahulu.")
