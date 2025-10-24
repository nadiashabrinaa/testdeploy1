else:
    st.subheader("🍽️ Deteksi Jenis Makanan (Meal, Dessert, Drink)")
    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])

    if uploaded_food:
        image = Image.open(uploaded_food).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        # ---------------------------
        # Load YOLO model (sudah pakai file .pt / download otomatis)
        # ---------------------------
        try:
            if YOLO is None:
                st.warning("Ultralytics (YOLO) belum terinstal.")
                model_yolo = None
            else:
                # pastikan folder model_uts ada
                os.makedirs("model_uts", exist_ok=True)

                # jika file model tidak ada, download otomatis
                if MODEL_FOOD_PATH is None or not os.path.exists(MODEL_FOOD_PATH):
                    MODEL_FOOD_PATH = "model_uts/yolov8n.pt"
                    YOLO_URL = "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8n.pt"
                    download_file(YOLO_URL, MODEL_FOOD_PATH)

                # load model YOLO
                model_yolo = load_yolo_model_safe(MODEL_FOOD_PATH)
        except Exception as e:
            st.warning(f"YOLO tidak tersedia: {e}")
            model_yolo = None

        # ---------------------------
        # Inferensi YOLO jika model tersedia
        # ---------------------------
        if model_yolo:
            try:
                # save temporary image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                results = model_yolo(tmp_path, conf=conf_thresh)
                plotted = results[0].plot()
                result_pil = Image.fromarray(plotted) if isinstance(plotted, np.ndarray) else Image.fromarray(np.array(plotted))
                st.image(result_pil, caption="Hasil Deteksi", use_container_width=True)

                # collect detections
                det_rows = []
                names = getattr(results[0], "names", None)
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None:
                    for b in boxes:
                        try:
                            cls = int(getattr(b, "cls").item() if hasattr(b, "cls") and hasattr(b.cls, "item") else getattr(b, "cls"))
                        except:
                            cls = None
                        try:
                            conf = float(getattr(b, "conf").item() if hasattr(b, "conf") and hasattr(b.conf, "item") else getattr(b, "conf"))
                        except:
                            conf = None
                        try:
                            xyxy = getattr(b, "xyxy")
                            xy = xyxy.cpu().numpy().tolist() if hasattr(xyxy, "cpu") else np.array(xyxy).tolist()
                        except:
                            xy = [None, None, None, None]
                        label = names[int(cls)] if (names and cls is not None and int(cls) in names) else (f"class_{cls}" if cls is not None else "")
                        det_rows.append({
                            "label": label, "class_id": cls, "confidence": conf,
                            "x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3]
                        })

                if det_rows:
                    df = pd.DataFrame(det_rows)
                    st.subheader("📋 Daftar Objek Terdeteksi")
                    st.dataframe(df)
                    st.subheader("📊 Ringkasan Kategori")
                    st.bar_chart(df["label"].value_counts())
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download hasil (CSV)", csv, "detection_results.csv", "text/csv")
                else:
                    st.warning("Tidak ada objek terdeteksi di atas threshold.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat inferensi YOLO: {e}")
            finally:
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except:
                    pass
        else:
            st.info("Fungsi deteksi YOLO tidak tersedia — periksa apakah paket `ultralytics` terinstal dan file .pt berada di repo.")
