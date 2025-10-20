import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
import tempfile
import os

# Judul dashboard
st.title("🌿 Dashboard Klasifikasi Daun Teh & Deteksi Makanan 🍱")

# Pilihan mode
mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi Daun Teh", "Deteksi Objek Makanan"])

# Fungsi preprocessing untuk model .h5
def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================
# MODE KLASIFIKASI DAUN TEH
# ==========================
if mode == "Klasifikasi Daun Teh":
    st.header("🌱 Klasifikasi Jenis Daun Teh")

    uploaded_file = st.file_uploader("Upload gambar daun teh", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        try:
            # Muat model Keras (.h5)
            model = tf.keras.models.load_model("nadia_shabrina_Laporan2.h5")

            # Preprocess dan prediksi
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)
            pred_class = np.argmax(prediction, axis=1)[0]

            st.success(f"Prediksi kelas daun teh: **{pred_class}**")
            st.write("Probabilitas prediksi:", prediction.tolist())

        except Exception as e:
            st.error(f"Gagal memuat model .h5: {e}")

# ==========================
# MODE DETEKSI MAKANAN (YOLO)
# ==========================
elif mode == "Deteksi Objek Makanan":
    st.header("🍛 Deteksi Objek Makanan")

    uploaded_file = st.file_uploader("Upload gambar makanan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        try:
            # Muat model YOLOv5 (.pt)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='Nadia_Laporan 4.pt', force_reload=False)

            # Simpan sementara untuk prediksi
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            # Deteksi objek
            results = model(tmp_path)
            results.render()  # render hasil deteksi

            # Tampilkan hasil deteksi
            st.image(results.ims[0], caption="Hasil Deteksi", use_column_width=True)
            st.write("📋 Detil Deteksi:")
            st.dataframe(results.pandas().xyxy[0])

            # Hapus file sementara
            os.remove(tmp_path)

        except Exception as e:
            st.error(f"Gagal memuat model .pt: {e}")
