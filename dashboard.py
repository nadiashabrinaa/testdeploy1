# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import io
import pandas as pd

# Try import libraries; handle gracefully
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    import tensorflow as tf
except Exception as e:
    tf = None

st.set_page_config(page_title="AI Vision Hub", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧠 AI Vision Hub — Tea Classifier & Food Detector</h1>", unsafe_allow_html=True)
st.write("Dual-mode: 🌿 Klasifikasi Daun Teh  •  🍽️ Deteksi Menu Makanan (meal / dessert / drink)")

# ---------- Configuration (ubah sesuai kebutuhan) ----------
# Paths default (letakkan file di folder models/)
DEFAULT_KERAS_MODEL = "models/nadia_shabrina_Laporan2.h5"
DEFAULT_YOLO_MODEL = "models/Nadia_Laporan4.pt"

# Untuk klasifikasi daun teh — sesuaikan label ini dengan label pada model .h5-mu
TEA_CLASSES = ["Green Tea", "Black Tea", "White Tea"]  # ubah jika perlu
IMAGE_SIZE = (224, 224)  # ubah sesuai input model Keras

# -----------------------------------------------------------

# Sidebar navigation
with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio("Pilih Mode:", ["🌿 Klasifikasi Daun Teh", "🍽️ Deteksi Menu Makanan"])
    st.markdown("---")
    st.write("Model files (opsional: unggah atau pakai default models/ folder)")
    keras_u = st.file_uploader("Upload Keras model (.h5)", type=["h5"], key="h5_upload")
    yolo_u = st.file_uploader("Upload YOLO model (.pt)", type=["pt"], key="pt_upload")
    st.markdown("---")
    st.write("Pengaturan deteksi")
    conf_thresh = st.slider("Confidence threshold (YOLO)", 0.0, 1.0, 0.45, 0.01)
    st.write("Tips: unggah gambar di masing-masing tab")

# ----------------- helper functions -----------------
@st.cache_resource
def load_keras_model_from_path(path):
    if tf is None:
        return None
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_yolo_model_from_path(path):
    if YOLO is None:
        return None
    return YOLO(path)

def save_uploaded_file(u_file, suffix=""):
    # returns a local path to saved file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(u_file.read())
        return tmp.name

def preds_to_bar(preds, class_names):
    # preds: 1D array
    df = pd.DataFrame({"class": class_names, "confidence": preds})
    return df

# ----------------- Load models (either uploaded or default) -----------------
keras_model = None
yolo_model = None

# Keras model
if keras_u is not None:
    st.sidebar.write("Keras model uploaded")
    keras_path = save_uploaded_file(keras_u, suffix=".h5")
    try:
        keras_model = load_keras_model_from_path(keras_path)
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model .h5: {e}")
else:
    if os.path.exists(DEFAULT_KERAS_MODEL):
        try:
            keras_model = load_keras_model_from_path(DEFAULT_KERAS_MODEL)
            st.sidebar.write(f"Using default Keras model: {DEFAULT_KERAS_MODEL}")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat default .h5: {e}")
    else:
        st.sidebar.info("No Keras model found in models/. Upload .h5 if needed.")

# YOLO model
if yolo_u is not None:
    st.sidebar.write("YOLO model uploaded")
    yolo_path = save_uploaded_file(yolo_u, suffix=".pt")
    try:
        yolo_model = load_yolo_model_from_path(yolo_path)
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model .pt: {e}")
else:
    if os.path.exists(DEFAULT_YOLO_MODEL):
        try:
            yolo_model = load_yolo_model_from_path(DEFAULT_YOLO_MODEL)
            st.sidebar.write(f"Using default YOLO model: {DEFAULT_YOLO_MODEL}")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat default .pt: {e}")
    else:
        st.sidebar.info("No YOLO model found in models/. Upload .pt if needed.")

# ----------------- Mode: Klasifikasi Daun Teh -----------------
if mode == "🌿 Klasifikasi Daun Teh":
    st.subheader("🌿 Klasifikasi Daun Teh")
    st.write("Model: Keras (.h5). Pastikan label pada `TEA_CLASSES` sesuai modelmu.")
    uploaded_img = st.file_uploader("Unggah gambar daun teh (jpg/png)", type=["jpg","jpeg","png"], key="tea_img")
    col1, col2 = st.columns([1, 1])

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        with col2:
            if keras_model is None:
                st.error("Model Keras belum tersedia. Upload .h5 di sidebar atau taruh file di models/")
            else:
                st.info("Menjalankan prediksi...")
                # preprocess
                img_resized = image.resize(IMAGE_SIZE)
                arr = np.array(img_resized) / 255.0
                arr = np.expand_dims(arr, axis=0).astype(np.float32)
                try:
                    preds = keras_model.predict(arr)[0]  # 1D
                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")
                    preds = None

                if preds is not None:
                    # If model outputs logits, apply softmax
                    if preds.sum() == 0 or (preds.max() > 1.0 and preds.min() < 0):
                        # try softmax
                        exp = np.exp(preds - np.max(preds))
                        preds = exp / exp.sum()
                    top_idx = int(np.argmax(preds))
                    label = TEA_CLASSES[top_idx] if top_idx < len(TEA_CLASSES) else f"Class {top_idx}"
                    st.markdown(f"### 🔎 Hasil: **{label}**")
                    st.write(f"Confidence: **{preds[top_idx]:.3f}**")
                    st.markdown("**Distribusi confidence per kelas:**")
                    df = preds_to_bar(preds, TEA_CLASSES)
                    st.dataframe(df.set_index("class"))
                    st.bar_chart(df.set_index("class"))

    else:
        st.info("Unggah gambar daun teh untuk melakukan klasifikasi.")

# ----------------- Mode: Deteksi Menu Makanan -----------------
else:
    st.subheader("🍽️ Deteksi Menu Makanan (meal / dessert / drink)")
    st.write("Model: YOLO (.pt). Output akan menampilkan bounding box, label, dan confidence.")
    uploaded_img_food = st.file_uploader("Unggah gambar makanan (jpg/png)", type=["jpg","jpeg","png"], key="food_img")
    show_legend = st.checkbox("Tampilkan legend kategori warna", value=True)

    if uploaded_img_food is not None:
        image = Image.open(uploaded_img_food).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

        if yolo_model is None:
            st.error("Model YOLO belum tersedia. Upload .pt di sidebar atau taruh file di models/")
        else:
            st.info("Menjalankan deteksi...")
            # save temp image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            try:
                results = yolo_model(tmp_path, conf=conf_thresh)
            except Exception as e:
                st.error(f"Error saat inferensi YOLO: {e}")
                results = None

            if results is not None:
                # plot image with boxes
                try:
                    plotted = results[0].plot()  # numpy array
                    if isinstance(plotted, np.ndarray):
                        plotted_img = Image.fromarray(plotted)
                    else:
                        # sometime ultralytics returns PIL image-like object
                        plotted_img = Image.fromarray(np.uint8(plotted))
                    st.image(plotted_img, caption="Hasil Deteksi", use_container_width=True)
                except Exception as e:
                    st.warning(f"Gagal menampilkan bounding box plot: {e}")

                # Collect detections into dataframe
                det_rows = []
                # results[0].boxes: iterable; attributes -> xyxy / xywh, conf, cls
                boxes = getattr(results[0], "boxes", None)
                names = getattr(results[0], "names", None)
                if boxes is not None:
                    for b in boxes:
                        # b.xyxy, b.conf, b.cls
                        try:
                            xyxy = b.xyxy.cpu().numpy().tolist() if hasattr(b.xyxy, "cpu") else b.xyxy.numpy().tolist()
                        except Exception:
                            # fallback to .xyxy if attribute exists as tensor or list
                            try:
                                xyxy = b.xyxy.tolist()
                            except:
                                xyxy = []
                        try:
                            conf = float(b.conf) if hasattr(b, "conf") else float(b.conf.item())
                        except:
                            conf = None
                        try:
                            cls = int(b.cls) if hasattr(b, "cls") else int(b.cls.item())
                        except:
                            cls = None
                        label = names[cls] if (names is not None and cls is not None and cls in names) else (f"class_{cls}" if cls is not None else "")
                        det_rows.append({
                            "label": label,
                            "class_id": cls,
                            "confidence": conf,
                            "x1": xyxy[0] if len(xyxy) >= 1 else None,
                            "y1": xyxy[1] if len(xyxy) >= 2 else None,
                            "x2": xyxy[2] if len(xyxy) >= 3 else None,
                            "y2": xyxy[3] if len(xyxy) >= 4 else None
                        })

                if len(det_rows) == 0:
                    st.write("Tidak ada objek terdeteksi di atas threshold.")
                else:
                    df_det = pd.DataFrame(det_rows)
                    st.subheader("📋 Daftar Objek Terdeteksi")
                    st.dataframe(df_det)

                    # Summary counts per label
                    summary = df_det['label'].value_counts().rename_axis('label').reset_index(name='count')
                    st.subheader("📊 Ringkasan Deteksi")
                    st.bar_chart(summary.set_index('label'))

                    # Download CSV
                    csv = df_det.to_csv(index=False).encode('utf-8')
                    st.download_button(label="⬇️ Download hasil (CSV)", data=csv, file_name="detection_results.csv", mime="text/csv")

                # legend
                if show_legend and names is not None:
                    st.subheader("Legenda Kelas (Model names)")
                    # show simple list
                    legend_list = [{"class_id": k, "name": v} for k, v in names.items()]
                    st.table(pd.DataFrame(legend_list))

    else:
        st.info("Unggah gambar menu makanan untuk melakukan deteksi.")
