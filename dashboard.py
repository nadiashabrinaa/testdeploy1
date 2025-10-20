import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd

# Library deteksi & klasifikasi
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    import tensorflow as tf
except Exception as e:
    tf = None

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Vision Hub", layout="wide")

DEFAULT_KERAS_MODEL = "models/nadia_shabrina_Laporan2.h5"
DEFAULT_YOLO_MODEL = "models/Nadia_Laporan 4.pt"

TEA_CLASSES = ["Green Tea", "Black Tea", "White Tea"]
IMAGE_SIZE = (224, 224)

# ------------------ HEADER ------------------
st.markdown("""
    <h1 style='text-align:center; color:#4B4B4B;'>🧠 Dashboard AI — Klasifikasi Daun Teh & Deteksi Menu Makanan</h1>
    <p style='text-align:center; color:gray;'>🌿 Klasifikasi Daun Teh &nbsp;&nbsp; | &nbsp;&nbsp; 🍽️ Deteksi Menu Makanan (Meal / Dessert / Drink)</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Helper ------------------
@st.cache_resource
def load_keras_model_from_path(path):
    return tf.keras.models.load_model(path) if tf else None

@st.cache_resource
def load_yolo_model_from_path(path):
    return YOLO(path) if YOLO else None

def save_uploaded_file(u_file, suffix=""):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(u_file.read())
        return tmp.name

def preds_to_bar(preds, class_names):
    df = pd.DataFrame({"class": class_names, "confidence": preds})
    return df

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio("Pilih Mode:", ["🌿 Klasifikasi Daun Teh", "🍽️ Deteksi Menu Makanan"])
    st.markdown("---")
    st.write("📂 Unggah Model (Opsional)")
    keras_u = st.file_uploader("Model Klasifikasi (.h5)", type=["h5"], key="h5_upload")
    yolo_u = st.file_uploader("Model Deteksi (.pt)", type=["pt"], key="pt_upload")
