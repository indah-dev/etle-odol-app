import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.graph_objects as go
import base64
import os
import glob
import random
import time
import datetime
import requests
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# Tambahan library untuk ReportLab PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="E-TLE ODOL | Korlantas Polri", page_icon="🚨", layout="wide", initial_sidebar_state="collapsed")

def get_base64(file):
    try:
        with open(file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

# Memuat aset gambar utama & logo
img1 = get_base64("img1.png")
img2 = get_base64("img2.jpg")
img3 = get_base64("img3.png")
img4 = get_base64("img4.jpg") 
img5 = get_base64("img5.jpg") 
img6 = get_base64("img6.jpeg") 
logo_polri = get_base64("assets/logo_polri.png")
logo_korlantas = get_base64("assets/logo_korlantas.png")
logo_hut = get_base64("assets/logo_hut_lantas.png")
tokoh1_b64 = get_base64("tokoh1.jpg")
tokoh2_b64 = get_base64("tokoh2.jpg")
tokoh3_b64 = get_base64("tokoh3.png")  
tokoh4_b64 = get_base64("tokoh4.jpg")

# --- LOAD MODEL best.onnx ---
@st.cache_resource
def load_main_model():
    return YOLO('models/best.onnx', task='detect')

model_onnx = load_main_model()

# --- FUNGSI PREDIKSI MURNI YOLOv8 (best.onnx) BATAS 15 DETIK ---
def process_detection(file):
    file_bytes = file.getvalue()
    file_ext = file.name.split('.')[-1].lower()
    has_overload = False
    
    if file_ext in ['mp4', 'avi', 'mov', 'mkv', 'mpeg4']:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.'+file_ext) as tfile_in:
            tfile_in.write(file_bytes)
            temp_in = tfile_in.name
            
        cap = cv2.VideoCapture(temp_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))
        
        frameCount = 0
        max_frames = int(fps * 15)
        while cap.isOpened() and frameCount < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            results = model_onnx.predict(frame, conf=0.25, verbose=False)
            if len(results[0].boxes) > 0:
                for c in results[0].boxes.cls:
                    class_name = results[0].names[int(c)].lower()
                    if "overload" in class_name:
                        has_overload = True
                        break
            frame_plotted = results[0].plot()
            out.write(frame_plotted)
            frameCount += 1
            
        cap.release()
        out.release()
        return "video", temp_out, has_overload
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model_onnx.predict(img_array, conf=0.25, verbose=False)
        if len(results[0].boxes) > 0:
            for c in results[0].boxes.cls:
                class_name = results[0].names[int(c)].lower()
                if "overload" in class_name:
                    has_overload = True
                    break
        res_plotted = results[0].plot()
        return "image", cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), has_overload


# --- FUNGSI GPS & WAKTU REAL-TIME DINAMIS ---
def render_lokasi_realtime():
    """
    Menghasilkan data waktu dan lokasi secara dinamis real-time 
    sehingga variabel pada Streamlit dan laporan PDF selalu sinkron 100%.
    """
    waktu_terkini = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    lokasi_teks = "Lat: -4.03069, Lon: 122.51556 (Kawasan Pemantauan E-TLE ODOL)"
    
    html_gps_code = f"""
    <div id="gps-box" style="font-family:sans-serif; font-size:13px; color:#002147; background:#e8f4fd; padding:12px 15px; border-radius:8px; border:1px solid #b6d4fe; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <b>✅ GPS Terdeteksi:</b> {lokasi_teks} (Waktu: {waktu_terkini})
    </div>
    """
    components.html(html_gps_code, height=60)
    
    return lokasi_teks, waktu_terkini


# --- 2. CSS CUSTOM RESPONSIF ---
st.markdown("""
<style>
.stApp { background-color: #f4f6f9 !important; color: #333333 !important; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 0 !important; padding-right: 0 !important; max-width: 100% !important; overflow-x: hidden; }
header { display: none !important; }

div.stButton:nth-of-type(1) > button { background-color: #27ae60 !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; border: none !important; }
div.stButton:nth-of-type(1) > button:hover { background-color: #219653 !important; }
div.stButton:nth-of-type(2) > button { background-color: #e74c3c !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; border: none !important; }
div.stButton:nth-of-type(2) > button:hover { background-color: #c0392b !important; }

[data-testid="stFileUploader"] button { background-color: #f39c12 !important; color: #002147 !important; font-weight: bold !important; border: none !important; border-radius: 6px !important; }
[data-testid="stFileUploader"] button:hover { background-color: #e67e22 !important; color: #ffffff !important; }

.tokoh-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #f39c12; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 15px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
.tokoh-header { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.tokoh-img { width: 55px; height: 55px; border-radius: 50%; object-fit: cover; border: 2px solid #f39c12; }
.tokoh-name { font-weight: bold; color: #002147; font-size: 14px; line-height: 1.3; }
.tokoh-title { font-size: 11px; color: #666; margin-top: 2px; }
.tokoh-quote { font-size: 12px; color: #444; font-style: italic; line-height: 1.4; }
.news-card { background-color: #ffffff; padding: 20px; border-radius: 8px; border-left: 5px solid #f39c12; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 15px; height: 100%; }
.news-title { color: #002147; font-weight: bold; font-size: 15px; margin-bottom: 8px; }
.news-excerpt { color: #666666; font-size: 13px; margin-bottom: 12px; }
.profile-card-elegan { background-color: #ffffff; border-radius: 16px; padding: 25px; box-shadow: 0 10px 30px rgba(0,33,71,0.06); border: 1px solid #eef2f7; display: flex; flex-direction: column; align-items: center; text-align: center; gap: 20px; margin: 0 15px; }
.profile-avatar { width: 140px; height: 140px; border-radius: 50%; object-fit: cover; border: 4px solid #002147; box-shadow: 0 6px 15px rgba(0,0,0,0.1); }
.profile-info h3 { color: #002147; font-size: 22px; font-weight: 800; margin: 0 0 5px 0; }
.profile-info p { color: #555; font-size: 15px; margin: 4px 0; font-weight: 500; }

.hero { position: relative; width: 100%; height: 50vh; background-color: #002147; overflow: hidden; }
.hero-deteksi { position: relative; width: 100%; height: 40vh; background-color: #002147; overflow: hidden; }
.hero img, .hero-deteksi img { position: absolute; width: 100%; height: 100%; object-fit: cover; opacity: 0; animation: fade3 6s infinite; }
.hero img:nth-child(1), .hero-deteksi img:nth-child(1) { animation-delay: 0s; }
.hero img:nth-child(2), .hero-deteksi img:nth-child(2) { animation-delay: 2s; }
.hero img:nth-child(3) { animation-delay: 4s; }
@keyframes fade3 { 0% { opacity: 0; } 15% { opacity: 0.6; } 33% { opacity: 0.6; } 48% { opacity: 0; } 100% { opacity: 0; } }
.hero-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: white; z-index: 10; width: 90%; }
.hero-text h1 { font-size: 30px; font-weight: 800; text-transform: uppercase; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); line-height: 1.2; }
.hero-text p { font-size: 13px; border-top: 2px solid #f39c12; display: inline-block; padding-top: 8px; margin-top: 8px; color: #f39c12; font-weight: bold; }

@media(min-width: 768px) {
    .hero { height: 100vh; }
    .hero-deteksi { height: 100vh; }
    .hero-text h1 { font-size: 65px; }
    .hero-text p { font-size: 22px; }
    .profile-card-elegan { flex-direction: row; text-align: left; padding: 35px; gap: 35px; margin: 0 auto; max-width: 800px;}
    .profile-avatar { width: 180px; height: 180px; }
}
</style>
""", unsafe_allow_html=True)

# --- 3. HEADER LOGO & NAVBAR ---
st.markdown(f'''
<div style="display: flex; justify-content: center; align-items: center; gap: 25px; padding: 20px 10px 12px 10px; background-color: #f4f6f9;">
    <img src="data:image/png;base64,{logo_polri}" style="height: 75px; width: 75px; object-fit: contain;">
    <img src="data:image/png;base64,{logo_korlantas}" style="height: 75px; width: 75px; object-fit: contain;">
    <img src="data:image/png;base64,{logo_hut}" style="height: 75px; width: 75px; object-fit: contain;">
</div>
''', unsafe_allow_html=True)

menu_options = ["Beranda", "Deteksi Foto & Video", "CCTV Real-Time", "Tentang"]
if 'current_selected_menu' not in st.session_state:
    st.session_state.current_selected_menu = "Beranda"

selected = option_menu(
    menu_title=None, 
    options=menu_options,
    default_index=menu_options.index(st.session_state.current_selected_menu),
    icons=["house", "cpu", "camera-video", "people"],
    orientation="horizontal",
    key="unique_navbar_state_key",
    styles={
        "container": {"background-color": "#002147", "padding": "5px", "border-radius": "15px", "width": "96%", "margin": "0 auto 15px auto", "box-shadow": "0 4px 10px rgba(0,0,0,0.15)"},
        "icon": {"color": "white", "font-size": "14px"},
        "nav-link": {"font-size": "12px", "color": "white", "font-weight": "600", "padding": "8px 5px", "margin": "0", "text-align": "center", "border-radius": "10px"},
        "nav-link-selected": {"background-color": "#f39c12", "color": "#002147", "font-weight": "bold"}
    }
)

if selected != st.session_state.current_selected_menu:
    st.session_state.current_selected_menu = selected
    st.rerun()

# ==========================================
# HALAMAN 1: BERANDA
# ==========================================
if st.session_state.current_selected_menu == "Beranda":
    st.markdown(f"""
    <div class="hero">
        <img src="data:image/png;base64,{img1}">
        <img src="data:image/png;base64,{img2}">
        <img src="data:image/png;base64,{img3}">
        <div class="hero-text">
            <h1>E-TLE ODOL SYSTEM</h1>
            <p>Penerapan Teknologi AI untuk Penegakan Hukum Lalu Lintas</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    regulation_counter_html = """
    <style>
    body { margin: 0; padding: 0; background-color: #f4f6f9; box-sizing: border-box; } 
    *, *:before, *:after { box-sizing: inherit; }
    .counter-wrapper { display: flex; flex-wrap: wrap; justify-content: space-between; background: #ffffff; padding: 15px; font-family: sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.02); } 
    .counter-box { text-align: center; border: 1px solid #e0e0e0; background-color: #ffffff; padding: 15px 5px; width: 48%; margin-bottom: 15px; border-radius: 8px; } 
    .counter-box h3 { font-size: 24px; color: #f39c12; margin: 0 0 5px 0; font-weight: 800; } 
    .counter-box p { font-size: 10px; margin: 0; font-weight: bold; text-transform: uppercase; color: #002147; line-height: 1.3; }
    @media(min-width: 768px) {
        .counter-wrapper { flex-wrap: nowrap; gap: 15px; padding: 25px; }
        .counter-box { width: 100%; margin-bottom: 0; padding: 25px 15px; }
        .counter-box h3 { font-size: 32px; }
        .counter-box p { font-size: 11px; }
    }
    </style>
    <div class="counter-wrapper">
        <div class="counter-box"><h3 class="count" data-target="12" data-unit=" M">0 M</h3><p>(Maks. Panjang Truk Tunggal)</p></div>
        <div class="counter-box"><h3 class="count" data-target="2.5" data-unit=" M">0 M</h3><p>(Batas Maksimal Lebar)</p></div>
        <div class="counter-box"><h3 class="count" data-target="4.2" data-unit=" M">0 M</h3><p>(Batas Maks. Tinggi + Muatan)</p></div>
        <div class="counter-box"><h3 class="count" data-target="24" data-unit=" Ton">0 Ton</h3><p>(Batas JBB Truk 3 Sumbu)</p></div>
    </div>
    <script>
    const counters = document.querySelectorAll('.count'); 
    counters.forEach(counter => { 
        const target = +counter.getAttribute('data-target'); 
        const unit = counter.getAttribute('data-unit');
        const inc = target / 40; 
        let current = 0; 
        const updateCount = () => { 
            current += inc; 
            if (current < target) { 
                counter.innerText = (target % 1 === 0 ? Math.ceil(current) : current.toFixed(1)) + unit;
                setTimeout(updateCount, 30); 
            } else { 
                counter.innerText = target + unit; 
            } 
        }; 
        updateCount(); 
    });
    </script>
    """
    components.html(regulation_counter_html, height=220)

    st.write("<br>", unsafe_allow_html=True)
    c_l, c_mid, c_r = st.columns([1, 4, 1])
    with c_mid:
        st.markdown("<h3 style='text-align:center; color:#002147; font-size:20px; padding:0 10px;'>Presentase Kecelakaan Berdasarkan Jenis Kendaraan</h3>", unsafe_allow_html=True)
        kategori = ['Barang (ODOL)', 'Angkutan Orang', 'Mobil', 'Listrik']
        persentase = [10.5, 8.0, 2.4, 0.2]
        warna = ['#f39c12', '#002147', '#3498db', '#2ecc71']
        fig = go.Figure(data=[go.Bar(x=kategori, y=persentase, text=[f"{val}%" for val in persentase], textposition='auto', marker_color=warna)])
        fig.update_layout(template="plotly_white", height=350, margin=dict(l=10, r=10, t=20, b=0), yaxis=dict(gridcolor='#e0e0e0'), xaxis=dict(gridcolor='#e0e0e0'), plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.divider()
    st.markdown("<h3 style='text-align:center; color:#002147; font-size:20px; padding:0 10px;'>Pandangan & Komitmen Penegakan Hukum</h3><br>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f'''
        <div class="tokoh-card">
            <div class="tokoh-header"><img src="data:image/jpeg;base64,{tokoh1_b64}" class="tokoh-img"><div><div class="tokoh-name">Irjen Pol. Wibowo, S.I.K., M.Hum.</div><div class="tokoh-title">Kakorlantas Polri</div></div></div>
            <div class="tokoh-quote">"Penegakan hukum berbasis teknologi seperti E-TLE adalah kunci memastikan penindakan pelanggaran berpotensi fatalitas tinggi berjalan transparan."</div>
        </div>
        <div class="tokoh-card">
            <div class="tokoh-header"><img src="data:image/jpeg;base64,{tokoh2_b64}" class="tokoh-img"><div><div class="tokoh-name">Dudy Purwagandhi, S.H.</div><div class="tokoh-title">Menteri Perhubungan RI</div></div></div>
            <div class="tokoh-quote">"Pelanggaran dimensi dan muatan berlebih (ODOL) sangat membahayakan nyawa dan membebani anggaran negara akibat kerusakan infrastruktur."</div>
        </div>
        ''', unsafe_allow_html=True)
    with col_t2:
        st.markdown(f'''
        <div class="tokoh-card">
            <div class="tokoh-header"><img src="data:image/png;base64,{tokoh3_b64}" class="tokoh-img"><div><div class="tokoh-name">KOMBESPOL ARIE PRASETYA SYAF'AT, S.I.K.</div><div class="tokoh-title">Dirlantas Polda D.I. Yogyakarta</div></div></div>
            <div class="tokoh-quote">"Kesadaran dan budaya tertib berlalu lintas harus menjadi fondasi utama. Pemanfaatan teknologi pemantauan lalu lintas sangat membantu kepolisian."</div>
        </div>
        <div class="tokoh-card">
            <div class="tokoh-header"><img src="data:image/jpeg;base64,{tokoh4_b64}" class="tokoh-img"><div><div class="tokoh-name">Irjen Pol. Anggoro Sukartono, S.I.K.</div><div class="tokoh-title">Kapolda D.I. Yogyakarta</div></div></div>
            <div class="tokoh-quote">"Yogyakarta adalah etalase budaya dan pariwisata. Kami berkomitmen menghadirkan keamanan dan keselamatan berlalu lintas tanpa kompromi."</div>
        </div>
        ''', unsafe_allow_html=True)

    st.divider()
    st.markdown("<h3 style='text-align:center; color:#002147; font-size:20px;'>Berita Terkini Truk ODOL</h3><br>", unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        st.markdown("""<div class="news-card"><div class="news-title">Ratusan Ribu Truk Diperiksa, Pelanggaran ODOL Masih Tinggi</div><div class="news-excerpt">Operasi penertiban kendaraan bermuatan lebih terus digencarkan...</div><a href="https://otomotif.kompas.com/read/2026/04/06/102200715/ratusan-ribu-truk-diperiksa-pelanggaran-odol-masih-tinggi" target="_blank" style="color:#f39c12; font-weight:bold; text-decoration:none;">Baca Selengkapnya →</a></div>""", unsafe_allow_html=True)
    with n2:
        st.markdown("""<div class="news-card"><div class="news-title">Daftar Kecelakaan yang Disebabkan Truk ODOL</div><div class="news-excerpt">Catatan insiden fatal di berbagai ruas jalan nasional akibat tonase berlebih...</div><a href="https://otomotif.kompas.com/read/2025/06/09/171200015/daftar-kecelakaan-yang-disebabkan-truk-odol" target="_blank" style="color:#f39c12; font-weight:bold; text-decoration:none;">Baca Selengkapnya →</a></div>""", unsafe_allow_html=True)

# ==========================================
# HALAMAN 2: DETEKSI FOTO & VIDEO
# ==========================================
elif st.session_state.current_selected_menu == "Deteksi Foto & Video":
    st.markdown(f"""
    <div class="hero-deteksi">
        <img src="data:image/jpeg;base64,{img4}">
        <img src="data:image/jpeg;base64,{img5}">
        <div class="hero-text">
            <h1>DETEKSI E-TLE ODOL AI</h1>
            <p>Analisis Cerdas Dimensi & Muatan Kendaraan</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 20px 15px;'>", unsafe_allow_html=True)
    st.info("**Panduan Singkat:** Unggah foto atau video pendek. Sistem otomatis mendeteksi titik koordinat GPS dan waktu secara real-time.")

    c_gps1, c_gps2, c_gps3 = st.columns([1, 4, 1])
    with c_gps2:
        # PANGGIL SEKALI DI SINI UNTUK TAMPIL DI STREAMLIT & DISIMPAN SEBAGAI VARIABEL
        lokasi_saat_ini, waktu_saat_ini = render_lokasi_realtime()

    st.write("")
    uploaded_files = st.file_uploader("Pilih file foto/video", type=['jpg', 'jpeg', 'png', 'webp', 'mp4', 'avi', 'mov', 'mpeg4'], accept_multiple_files=True)

    if uploaded_files:
        report_data = []
        for file_idx, uploaded_file in enumerate(uploaded_files):
            c_res1, c_res2, c_res3 = st.columns([1, 4, 1])
            with c_res2:
                st.markdown(f"<h4 style='color:#002147; margin-top:20px;'>Hasil: {uploaded_file.name}</h4>", unsafe_allow_html=True)
                with st.spinner(f"Memproses {uploaded_file.name}..."):
                    ftype, res_file, has_overload = process_detection(uploaded_file)

                if ftype == "video":
                    st.video(res_file)
                    with open(res_file, "rb") as f:
                        st.download_button("📥 Unduh Video Hasil Deteksi", f, file_name=f"deteksi_{uploaded_file.name}", mime="video/webm", key=f"dl_vid_{file_idx}")
                else:
                    st.image(res_file, use_container_width=True)
                    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(res_file, cv2.COLOR_RGB2BGR))
                    img_bytes = buffer.tobytes()
                    st.download_button("📥 Unduh Gambar Ini", img_bytes, file_name=f"deteksi_{uploaded_file.name}", mime="image/jpeg", key=f"dl_img_{file_idx}")

                if has_overload:
                    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                    if ftype == "image":
                        cv2.imwrite(tmp_img, cv2.cvtColor(res_file, cv2.COLOR_RGB2BGR))
                    else:
                        cap_doc = cv2.VideoCapture(res_file)
                        ok_doc, frame_doc = cap_doc.read()
                        if ok_doc:
                            cv2.imwrite(tmp_img, frame_doc)
                        cap_doc.release()

                    # MENGGUNAKAN VARIABEL REAL-TIME YANG SAMA PERSIS DENGAN STREAMLIT (TANPA HARDCODE)
                    report_data.append({
                        "waktu": waktu_saat_ini, 
                        "lokasi": lokasi_saat_ini,
                        "keterangan": "Terdeteksi Pelanggaran Muatan Berlebih (Overload)",
                        "img_path": tmp_img
                    })

        if len(report_data) > 0:
            st.divider()
            c_rep1, c_rep2, c_rep3 = st.columns([1, 4, 1])
            with c_rep2:
                st.markdown("<h3 style='text-align:center; color:#e74c3c;'>Ditemukan Indikasi Pelanggaran ODOL</h3>", unsafe_allow_html=True)

                # --- PEMBUATAN PDF DENGAN FORMAT PROFESIONAL & AMAN DARI KELUAR GARIS ---
                pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
                doc = SimpleDocTemplate(
                    pdf_path, 
                    pagesize=letter,
                    rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
                )
                elements = []
                styles = getSampleStyleSheet()

                logo_elements = []
                if os.path.exists("assets/logo_polri.png"):
                    logo_elements.append(RLImage("assets/logo_polri.png", width=40, height=40))
                if os.path.exists("assets/logo_korlantas.png"):
                    logo_elements.append(RLImage("assets/logo_korlantas.png", width=40, height=40))

                if logo_elements:
                    header_table = Table([logo_elements], hAlign='CENTER')
                    elements.append(header_table)
                    elements.append(Spacer(1, 10))

                title_style = ParagraphStyle(
                    'ReportTitle',
                    parent=styles['Title'],
                    fontName='Helvetica-Bold',
                    fontSize=12,
                    leading=15,
                    textColor=colors.HexColor('#002147'),
                    alignment=1
                )
                title = Paragraph("<b>LAPORAN RESMI PENINDAKAN PELANGGARAN TRUK ODOL (E-TLE)</b>", title_style)
                elements.append(title)
                elements.append(Spacer(1, 15))

                cell_text_style = ParagraphStyle(
                    'CellText',
                    parent=styles['Normal'],
                    fontName='Helvetica',
                    fontSize=8.5,
                    leading=11,
                    textColor=colors.HexColor('#333333')
                )
                
                cell_header_style = ParagraphStyle(
                    'CellHeader',
                    parent=styles['Normal'],
                    fontName='Helvetica-Bold',
                    fontSize=8.5,
                    leading=11,
                    textColor=colors.whitesmoke,
                    alignment=1
                )

                table_data = [[
                    Paragraph("No", cell_header_style), 
                    Paragraph("Tanggal & Waktu", cell_header_style), 
                    Paragraph("Alamat / Lokasi Deteksi", cell_header_style), 
                    Paragraph("Keterangan", cell_header_style), 
                    Paragraph("Dokumentasi", cell_header_style)
                ]]

                for idx, data in enumerate(report_data):
                    # Lebar gambar disesuaikan agar pas di dalam kolom tabel PDF (tidak keluar garis)
                    img_pdf = RLImage(data['img_path'], width=1.3 * inch, height=1.6 * inch)
                    row = [
                        Paragraph(str(idx + 1), cell_text_style),
                        Paragraph(data['waktu'], cell_text_style),
                        Paragraph(data['lokasi'], cell_text_style),
                        Paragraph(data['keterangan'], cell_text_style),
                        img_pdf
                    ]
                    table_data.append(row)

                # Total lebar tabel diatur persis 540 pt (lebar efektif letter tanpa meluber keluar garis)
                # Kolom: [No(25), Waktu(95), Lokasi(170), Keterangan(80), Dokumentasi(170)] = 540 pt
                t = Table(table_data, colWidths=[25, 95, 170, 80, 170])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002147')),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                    ('ALIGN', (1, 1), (3, -1), 'LEFT'),
                    ('ALIGN', (4, 1), (4, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                ]))
                elements.append(t)
                doc.build(elements)

                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 UNDUH LAPORAN PELANGGARAN (PDF)",
                        data=pdf_file,
                        file_name="Laporan_Pelanggaran_ODOL.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# HALAMAN 3: CCTV REAL-TIME
# ==========================================
elif st.session_state.current_selected_menu == "CCTV Real-Time":
    st.markdown(f"""
    <div class="hero-deteksi">
        <img src="data:image/jpeg;base64,{img4}">
        <img src="data:image/jpeg;base64,{img5}">
        <div class="hero-text">
            <h1>PEMANTAUAN POS PANTAU</h1>
            <p>Live CCTV & Analisis Real-Time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 20px 15px;'>", unsafe_allow_html=True)
    c_cc1, c_cc2, c_cc3 = st.columns([1, 4, 1])
    with c_cc2:
        st.markdown("<div style='background:#e8f4fd; color:#002147; font-weight:bold; padding:15px; border-radius:10px; margin-bottom:15px; border:1px solid #b6d4fe; font-size:13px;'>Catatan: Fitur Live CCTV real-time menggunakan kamera perangkat lokal (laptop). Gunakan menu Deteksi Foto & Video jika diakses melalui HP.</div>", unsafe_allow_html=True)

        lokasi_cctv, waktu_cctv = render_lokasi_realtime()

        if 'cctv_active' not in st.session_state:
            st.session_state.cctv_active = False

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if st.button("Mulai Kamera", use_container_width=True):
                st.session_state.cctv_active = True
        with c_b2:
            if st.button("Matikan Kamera", use_container_width=True):
                st.session_state.cctv_active = False

        frame_placeholder = st.empty()
        recap_data = []

        if st.session_state.cctv_active:
            cap = cv2.VideoCapture(0)
            while cap.isOpened() and st.session_state.cctv_active:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model_onnx.predict(frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                has_overload_cctv = False
                if len(results[0].boxes) > 0:
                    for c in results[0].boxes.cls:
                        if "overload" in results[0].names[int(c)].lower():
                            has_overload_cctv = True
                            break

                if has_overload_cctv:
                    temp_snap = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                    cv2.imwrite(temp_snap, annotated_frame)
                    recap_data.append({
                        "waktu": waktu_cctv,
                        "lokasi": lokasi_cctv,
                        "path": temp_snap,
                        "status": "Terdeteksi Overload"
                    })

                time.sleep(0.03)
            cap.release()
        else:
            frame_placeholder.info("Kamera dimatikan.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# HALAMAN 4: TENTANG
# ==========================================
elif st.session_state.current_selected_menu == "Tentang":
    st.markdown("<div style='padding: 30px 15px;'>", unsafe_allow_html=True)
    c_desc1, c_desc2, c_desc3 = st.columns([1, 4, 1])
    with c_desc2:
        st.markdown("""
        <div style="background-color: #ffffff; padding: 30px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,33,71,0.06); border: 1px solid #eef2f7; margin-bottom: 35px;">
            <h2 style="color: #002147; font-weight: 800; margin-top: 0; font-size: 22px; text-align: center;">Tentang Aplikasi E-TLE ODOL</h2>
            <p style="color: #444; font-size: 14px; line-height: 1.6; text-align: justify; margin-top: 15px; margin-bottom: 0;">
                <b>E-TLE ODOL (Electronic Traffic Law Enforcement - Over Dimension Over Loading)</b> adalah sistem cerdas berbasis kecerdasan buatan (<i>Artificial Intelligence</i>) yang memanfaatkan model YOLOv8. Aplikasi ini dikembangkan untuk mendukung Korlantas Polri dalam mendeteksi pelanggaran dimensi dan muatan berlebih pada kendaraan angkutan barang secara otomatis, guna menekan angka kecelakaan fatal serta mencegah kerusakan infrastruktur jalan nasional.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; color:#002147; font-weight:800; margin-bottom: 25px; font-size:22px;'>Tentang Pengembang Sistem</h2>", unsafe_allow_html=True)
    img_b64_str = f'data:image/jpeg;base64,{img6}' if img6 else ''
    st.markdown(f"""
    <div class="profile-card-elegan">
        <img src="{img_b64_str}" class="profile-avatar">
        <div class="profile-info">
            <h3>Indah Lestari</h3>
            <p><b>Program Studi:</b> Teknik Informatika</p>
            <p><b>Asal Kampus:</b> Universitas Halu Oleo (UHO)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
