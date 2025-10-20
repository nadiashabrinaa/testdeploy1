import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
import numpy as np

# ============================
# 1️⃣  Load kedua model
# ============================
@st.cache_resource
def load_models():
    # Model .pt (PyTorch) untuk klasifikasi daun teh
    model_pt = torch.load("Nadia_Laporan 4.pt", map_location=torch.device("cpu"))
    model_pt.eval()

    # Model .h5 (Keras/TensorFlow) untuk deteksi makanan
    model_h5 = tf.keras.models.load_model("nadia_shabrina_Laporan2.h5")

    return model_pt, model_h5

model_pt, model_h5 = load_models()

# ============================
# 2️⃣  Fungsi prediksi PyTorch (daun teh)
# ============================
def predict_tealeaf(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    return predicted.item(), probs

# ============================
# 3️⃣  Fungsi prediksi Keras (makanan)
# ============================
def predict_food(model, image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)
    return predicted_class, confidence

# ============================
# 4️⃣  Dashboard
# ============================
st.title("🍃 Dashboard Klasifikasi & Deteksi Gambar")
st.write("Upload gambar dan pilih model yang ingin digunakan untuk prediksi.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Pilih Model", ["Model Daun Teh (.pt)", "Model Makanan (.h5)"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    if st.button("Prediksi"):
        if model_choice == "Model Daun Teh (.pt)":
            predicted_class, probs = predict_tealeaf(model_pt, image)
            st.success(f"Hasil Prediksi: **Kelas {predicted_class}**")
            st.write("Probabilitas:")
            for i, p in enumerate(probs):
                st.write(f" - Kelas {i}: {p:.3f}")

        elif model_choice == "Model Makanan (.h5)":
            predicted_class, confidence = predict_food(model_h5, image)
            st.success(f"Hasil Deteksi: **Kelas {predicted_class}** dengan kepercayaan {confidence:.2f}")

st.markdown("---")
st.caption("Dibuat oleh Nadia Shabrina 🌿")
