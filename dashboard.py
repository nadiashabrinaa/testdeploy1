import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd

# Coba impor library, tangani jika belum terinstal
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    import tensorflow as tf
except Exception as e:
    tf = None

# ------------------ Konfigurasi ------------------
st.set_page_config(page_title="Dashboard AI — Klasifikasi & Deteksi", layout="wide")
st.title("🧠 Dashboard AI — Klasifikasi Daun Teh & Deteksi Menu Makanan")

# Lokasi file model (gunakan file yang kamu punya)
KERAS_MODEL_PATH = "nadia_shabrina_Laporan2.h5"
YOLO_MODEL_PATH = "Nadia_Laporan 4.pt"

# Label untuk klasifikasi daun teh — ubah sesuai model kamu
TEA_CLASSES = ["Green Tea", "Black Tea", "White Tea"]  
IMAGE_SIZE = (224, 224)  # ubah sesuai input model

# ------------------ Fungsi bantu ------------------
@st.cache_resource
def load_keras_model(path):
    if tf is None:
        st.error("TensorFlow belum terinstal.")
        return None
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_yolo_model(path):
    if YOLO is None:
        st.error("Ultralytics belum terinstal.")
        return None
    return YOLO(path)

def preds_to_df(preds, class_names):
    df = pd.DataFrame({"Kelas": class_names, "Probabilitas": preds})
    return
