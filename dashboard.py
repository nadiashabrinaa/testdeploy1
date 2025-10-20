import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
import io

# Judul dashboard
st.title("🌿 Dashboard Klasifikasi Daun Teh & Deteksi Objek Makanan 🍱")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Klasifikasi Daun Teh", "Deteksi Objek Makanan"]
)

# Fungsi untuk preprocessing gambar (untuk model .h5)
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- MODE KLASIFIKASI DAUN TEH (.h5) ---
if menu == "Klasifikasi Daun Teh":
    st.header("🌱 Klasifikasi Jenis Daun Teh")
    uploaded_file = st.file_uploader("Upload gambar daun teh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Muat model h5
        model = tf.keras.models.load_model("nadia_shabrina_Laporan2.h5")

        # Preprocess dan prediksi
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]

        st.success(f"Prediksi kelas daun teh: **{pred_class}**")
        st.write("Probabilitas:", prediction)

# --- MODE DETEKSI OBJEK MAKANAN (.pt) ---
else:
    st.header("🍛 Deteksi Objek Jenis Makanan")
    uploaded_file = st.file_uploader("Upload gambar makanan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Muat model PyTorch
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='Nadia_Laporan 4.pt', force_reload=True)

        # Simpan gambar sementara
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Deteksi objek
        results = model("temp.jpg")

        # Tampilkan hasil deteksi
        st.image(np.squeeze(results.render()), caption="Hasil Deteksi", use_column_width=True)
        st.write("📋 Detil Deteksi:")
        st.write(results.pandas().xyxy[0])
