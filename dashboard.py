# ---------------------------
# Halaman Pembuka (Tampilan Awal)
# ---------------------------
if st.session_state.page == "home":
    st.markdown(
        "<h1 style='text-align:center; font-size:42px;'>🤖 Selamat Datang di <b>AI Vision Dashboard</b></h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>"
        "Sebuah sistem berbasis <b>Kecerdasan Buatan (AI)</b> untuk "
        "<b>klasifikasi penyakit daun teh</b> dan <b>deteksi jenis makanan</b> secara otomatis dan interaktif."
        "</p>",
        unsafe_allow_html=True
    )

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712105.png",
        width=250,
        caption="AI Vision System — Powered by Streamlit, TensorFlow & YOLOv8"
    )

    st.markdown("---")
    st.markdown("### ✨ Kelebihan Dashboard Ini")
    st.markdown("""
    - ⚡ **Cepat & Akurat** – Proses gambar hanya dalam hitungan detik  
    - 🧠 **Ditenagai AI Modern** – Menggunakan model CNN & YOLOv8  
    - 🌿 **Dua Fungsi Utama** – Analisis daun teh & deteksi jenis makanan  
    - 📊 **Interaktif & Informatif** – Hasil tampil otomatis dalam grafik  
    - ☁️ **Ramah Pengguna** – Tidak perlu instalasi tambahan, cukup unggah gambar  
    """)

    st.markdown("---")
    st.markdown("### 🧭 Kegunaan Dashboard")
    st.markdown("""
    Dashboard ini dirancang untuk membantu penelitian, pembelajaran, dan demonstrasi teknologi
    **Computer Vision**.  
    Pengguna dapat:
    - 🔍 Mengidentifikasi penyakit pada daun teh berdasarkan citra  
    - 🍽️ Mendeteksi jenis makanan (Meal, Dessert, Drink) secara otomatis  
    - 💾 Menyimpan hasil deteksi ke file **CSV** untuk analisis lanjutan  
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Mulai Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

    st.stop()
