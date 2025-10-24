# =====================================
# 🏠 HALAMAN PEMBUKA (WELCOME PAGE)
# =====================================
import streamlit as st

if "start_dashboard" not in st.session_state:
    st.session_state.start_dashboard = False

if not st.session_state.start_dashboard:
    st.markdown("<h1 style='text-align:center;'>🤖 Selamat Datang di <b>AI Vision Dashboard</b></h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>"
        "Temukan kecerdasan buatan yang membantu Anda <b>mendeteksi jenis makanan</b> "
        "dan <b>mendiagnosis penyakit daun teh</b> secara otomatis dengan teknologi AI modern."
        "</p>", unsafe_allow_html=True
    )

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712105.png",
        width=250,
        caption="AI Vision System powered by Streamlit, TensorFlow & YOLOv8",
    )

    st.markdown("---")
    st.markdown("### ✨ Mengapa Memilih Dashboard Ini?")
    st.markdown("""
    - 🚀 **Cepat & Akurat:** Deteksi otomatis dengan model **CNN** dan **YOLOv8**  
    - 🧠 **Cerdas:** Menggunakan pembelajaran mesin untuk menganalisis citra secara real-time  
    - 🌿 **Multifungsi:** Dapat digunakan untuk *agriculture monitoring* dan *food recognition*  
    - 📊 **Interaktif:** Hasil visualisasi langsung dalam bentuk grafik dan tabel  
    - ☁️ **Praktis:** Cukup unggah gambar — sistem akan bekerja untuk Anda!  
    """)

    st.markdown("---")
    st.markdown("### 📌 Kegunaan Utama")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🍃 Klasifikasi Daun Teh")
        st.markdown("Mengidentifikasi jenis penyakit daun teh berdasarkan citra.")
    with col2:
        st.markdown("#### 🍽️ Deteksi Jenis Makanan")
        st.markdown("Membedakan kategori *Meal, Dessert,* atau *Drink* menggunakan YOLOv8.")

    st.markdown("---")
    if st.button("🚀 Mulai Dashboard", use_container_width=True):
        st.session_state.start_dashboard = True
        st.experimental_rerun()
    st.stop()
