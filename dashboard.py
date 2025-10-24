import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# =========================
# Judul Aplikasi
# =========================
st.set_page_config(page_title="Tea Leaf & Food Classifier", layout="centered")
st.title("🍃 Tea Leaf & 🍽️ Food Image Classifier")
st.write("Upload gambar daun teh atau makanan untuk mendeteksi jenis penyakit atau kategori makanan.")

# =========================
# Pilihan Mode
# =========================
mode = st.radio(
    "Pilih jenis deteksi:",
    ("Klasifikasi Penyakit Daun Teh", "Deteksi Jenis Makanan"),
    horizontal=True
)

# =========================
# Load Model Sesuai Mode
# =========================
if mode == "Klasifikasi Penyakit Daun Teh":
    model_path = "model_uts/nadia_shabrina_Laporan2.h5"  # ganti dengan model kamu
    labels = [
        "Red Leaf Spot",
        "Algal Leaf Spot",
        "Bird’s Eyespot",
        "Gray Light",
        "White Spot",
        "Anthracnose",
        "Brown Blight",
        "Healthy Tea Leaf"
    ]
else:
    model_path = "model_uts/Nadia_Laporan 4.pt"  # ganti dengan model kamu
    labels = ["Meal", "Drink", "Dessert"]

# Load model (gunakan cache supaya cepat)
@st.cache_resource
def load_selected_model(path):
    return load_model(path)

model = load_selected_model(model_path)

# =========================
# Upload Gambar
# =========================
uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

# Dapatkan ukuran input model
input_shape = model.input_shape[1:3]  # (height, width)

# Preprocessing
img_resized = img.resize(input_shape)
arr = image.img_to_array(img_resized)
arr = np.expand_dims(arr, axis=0)
arr = arr.astype("float32") / 255.0

# Prediksi
preds = model.predict(arr)
if preds.ndim > 1:
    preds = preds[0]

    pred_class = labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # =========================
    # Tampilkan Hasil
    # =========================
    st.markdown("---")
    st.subheader("🔍 Hasil Prediksi:")
    st.success(f"**{pred_class}** ({confidence:.2f}% confidence)")

    # Tampilkan probabilitas tiap kelas
    st.markdown("#### Rincian Kelas:")
    for i, label in enumerate(labels):
        st.write(f"- {label}: {preds[i]*100:.2f}%")

else:
    st.info("Silakan upload gambar untuk memulai deteksi.")

