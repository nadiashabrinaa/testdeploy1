# dashboard.py
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import os

# ---------------------------
# Imports: TensorFlow & YOLO
# ---------------------------
# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None  # nanti tampilkan pesan kalau belum terinstal

# Ultralytics YOLO (opsional)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------------------
# Config page
# ---------------------------
st.set_page_config(page_title="AI Vision Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🤖 AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.write("🌿 Klasifikasi Penyakit Daun Teh  |  🍽️ Deteksi Jenis Makanan")

# ---------------------------
# Model paths (cek & sesuaikan)
# ---------------------------
# Letakkan model .h5 dan .pt di folder model_uts/ atau di root repo
POSSIBLE_TEA_PATHS = [
    "model_uts/nadia_shabrina_Laporan2.h5",
    "model_uts/model_daun_teh.h5",
    "nadia_shabrina_Laporan2.h5",
    "model_daun_teh.h5",
]

POSSIBLE_FOOD_PATHS = [
    "model_uts/Nadia_Laporan4.pt",
    "model_uts/Nadia_Laporan 4.pt",
    "Nadia_Laporan4.pt",
    "Nadia_Laporan 4.pt",
    "model_makanan.pt",
    "model_makanan.pt",
]

def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

MODEL_TEA_PATH = find_existing_path(POSSIBLE_TEA_PATHS)
MODEL_FOOD_PATH = find_existing_path(POSSIBLE_FOOD_PATHS)

# ---------------------------
# Classes
# ---------------------------
TEA_CLASSES = [
    "Red Leaf Spot", "Algal Leaf Spot", "Bird’s Eyespot",
    "Gray Blight", "White Spot", "Anthracnose",
    "Brown Blight", "Healthy Tea Leaves"
]
FOOD_CLASSES = ["Meal", "Dessert", "Drink"]

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Pilih Mode")
    mode = st.radio("Mode Analisis:", ["🌿 Klasifikasi Penyakit Daun Teh", "🍽️ Deteksi Jenis Makanan"])
    conf_thresh = st.slider("Confidence Threshold (untuk YOLO)", 0.1, 1.0, 0.45, 0.01)
    st.markdown("---")
    st.write("Model yang ditemukan (searched paths):")
    st.write(f"- Keras (.h5): {MODEL_TEA_PATH or 'NOT FOUND'}")
    st.write(f"- YOLO (.pt): {MODEL_FOOD_PATH or 'NOT FOUND'}")
    st.markdown("---")
    st.info("Jika model tidak ditemukan, taruh file model di folder `model_uts/` atau di root repo.")

# ---------------------------
# Helpers: load models
# ---------------------------
@st.cache_resource
def load_keras_model_safe(path):
    if tf is None:
        raise RuntimeError("TensorFlow belum terinstal di environment.")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File model Keras tidak ditemukan: {path}")
    return load_model(path)

@st.cache_resource
def load_yolo_model_safe(path):
    if YOLO is None:
        raise RuntimeError("Ultralytics (YOLO) belum terinstal.")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File model YOLO tidak ditemukan: {path}")
    return YOLO(path)

def safe_softmax(x):
    x = np.array(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------------------
# Preprocessing otomatis berdasar model input
# ---------------------------
def preprocess_for_keras(pil_image, model):
    """
    Resize to model.input_shape automatically and normalize to [0,1].
    Returns batch array shape (1, H, W, C) float32.
    """
    if model is None:
        raise RuntimeError("Model Keras belum dimuat.")
    # model.input_shape bisa berupa (None, H, W, C) atau (None, features)
    input_shape = getattr(model, "input_shape", None)
    if not input_shape:
        raise RuntimeError("Tidak dapat membaca input_shape dari model.")
    # handle sequential dense-only models (1D input) — not common for image models
    if len(input_shape) == 2:
        raise RuntimeError("Model tampak menerima input 1D — skrip ini mengharapkan model citra (H,W,3).")
    h, w = input_shape[1], input_shape[2]
    if h is None or w is None:
        # fallback ke 224x224
        h, w = 224, 224
    img = pil_image.resize((w, h))
    arr = np.array(img).astype("float32") / 255.0
    # if model expects grayscale channel =1, convert
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 4:
        # drop alpha if present
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)  # batch
    return arr

# ---------------------------
# Main UI logic
# ---------------------------
if mode == "🌿 Klasifikasi Penyakit Daun Teh":
    st.subheader("🌿 Deteksi Penyakit Daun Teh Berdasarkan Citra")
    uploaded_img = st.file_uploader("Unggah gambar daun teh", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        # load model (with friendly messages)
        try:
            if MODEL_TEA_PATH is None:
                st.error("File model .h5 tidak ditemukan. Pastikan model berada di salah satu path yang dicari.")
            else:
                model_tea = load_keras_model_safe(MODEL_TEA_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model Keras: {e}")
            model_tea = None

        if model_tea:
            try:
                arr = preprocess_for_keras(image, model_tea)
                preds = model_tea.predict(arr)
                # flatten batch dim if exists
                if preds.ndim > 1:
                    preds = preds[0]
                # ensure probabilities
                if preds.max() > 1.0 or preds.min() < 0:
                    preds = safe_softmax(preds)
                top_idx = int(np.argmax(preds))
                label = TEA_CLASSES[top_idx] if top_idx < len(TEA_CLASSES) else f"Class {top_idx}"
                conf = float(preds[top_idx])
                st.success(f"**Prediksi: {label}** (Confidence: {conf:.3f})")
                df = pd.DataFrame({"Class": TEA_CLASSES, "Confidence": np.round(preds, 4)})
                st.dataframe(df)
                st.bar_chart(df.set_index("Class"))
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

else:
    st.subheader("🍽️ Deteksi Jenis Makanan (Meal, Dessert, Drink)")
    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])

    if uploaded_food:
        image = Image.open(uploaded_food).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        # load YOLO model safely
        try:
            if MODEL_FOOD_PATH is None:
                st.error("File model .pt tidak ditemukan. Pastikan file YOLO (.pt) ada dan namanya benar.")
                model_yolo = None
            else:
                model_yolo = load_yolo_model_safe(MODEL_FOOD_PATH)
        except Exception as e:
            st.warning(f"YOLO tidak tersedia: {e}")
            model_yolo = None

        if model_yolo:
            try:
                # save temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                results = model_yolo(tmp_path, conf=conf_thresh)
                # plot result robustly
                plotted = results[0].plot()
                if isinstance(plotted, np.ndarray):
                    result_pil = Image.fromarray(plotted)
                else:
                    result_pil = Image.fromarray(np.array(plotted))
                st.image(result_pil, caption="Hasil Deteksi", use_container_width=True)

                # collect detections robustly
                det_rows = []
                names = getattr(results[0], "names", None)
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None:
                    for b in boxes:
                        # handle attributes possibly as tensors
                        try:
                            cls = int(getattr(b, "cls").item() if hasattr(b, "cls") and hasattr(b.cls, "item") else getattr(b, "cls"))
                        except Exception:
                            try:
                                cls = int(b.cls)
                            except Exception:
                                cls = None
                        try:
                            conf = float(getattr(b, "conf").item() if hasattr(b, "conf") and hasattr(b.conf, "item") else getattr(b, "conf"))
                        except Exception:
                            conf = None
                        # xyxy may be list or tensor
                        try:
                            xyxy = getattr(b, "xyxy")
                            if hasattr(xyxy, "cpu"):
                                xy = xyxy.cpu().numpy().tolist()
                            else:
                                # sometimes xyxy is array-like
                                xy = np.array(xyxy).tolist()
                        except Exception:
                            xy = [None, None, None, None]
                        label = names[int(cls)] if (names is not None and cls is not None and int(cls) in names) else (f"class_{cls}" if cls is not None else "")
                        det_rows.append({
                            "label": label,
                            "class_id": cls,
                            "confidence": conf,
                            "x1": xy[0] if len(xy) >= 1 else None,
                            "y1": xy[1] if len(xy) >= 2 else None,
                            "x2": xy[2] if len(xy) >= 3 else None,
                            "y2": xy[3] if len(xy) >= 4 else None,
                        })

                if det_rows:
                    df = pd.DataFrame(det_rows)
                    st.subheader("📋 Daftar Objek Terdeteksi")
                    st.dataframe(df)
                    st.subheader("📊 Ringkasan Kategori")
                    st.bar_chart(df["label"].value_counts())
                    # download csv
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download hasil (CSV)", csv, "detection_results.csv", "text/csv")
                else:
                    st.warning("Tidak ada objek terdeteksi di atas threshold.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat inferensi YOLO: {e}")
            finally:
                # cleanup temp file if exists
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        else:
            st.info("Fungsi deteksi YOLO tidak tersedia — periksa apakah paket `ultralytics` terinstal dan file .pt berada di repo.")
