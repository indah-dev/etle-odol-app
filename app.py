import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.graph_objects as go
import base64
import os
import glob
import random
import time
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

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
            if not ret: break
            
            results = model_onnx.predict(frame, conf=0.25, verbose=False)
            frame_plotted = results[0].plot()
            out.write(frame_plotted)
            frameCount += 1
            
        cap.release()
        out.release()
        return "video", temp_out
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = model_onnx.predict(img_array, conf=0.25, verbose=False)
        res_plotted = results[0].plot()
        
        return "image", cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# --- 2. CSS CUSTOM TEMA TERANG + KUSTOM TOMBOL UPLOAD ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9 !important; color: #333333 !important; }
    .block-container { padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 0 !important; padding-right: 0 !important; max-width: 100% !important; }
    header { display: none !important; } 
    .nav-wrapper { position: absolute; top: 15px; width: 100%; z-index: 9999; display: flex; justify-content: center; }

    iframe {
        border-radius: 30px !important;
        border: none !important;
        background-color: transparent !important;
    }
    div[data-testid="stFrame"] {
        border-radius: 30px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #f39c12 !important;
        color: #002147 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #e67e22 !important;
        color: #ffffff !important;
    }

    /* TOMBOL MULAI (HIJAU) & MATIKAN (MERAH) */
    div.stButton > button:first-child {
        background-color: #27ae60 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #219653 !important;
    }

    div.stButton:nth-of-type(2) > button, 
    div.row-widget.stButton:nth-child(2) button {
        background-color: #e74c3c !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }
    div.stButton:nth-of-type(2) > button:hover, 
    div.row-widget.stButton:nth-child(2) button:hover {
        background-color: #c0392b !important;
    }

    /* HERO BERANDA */
    .hero { position: relative; width: 100vw; height: 100vh; background-color: #002147; overflow: hidden; }
    .hero img { position: absolute; width: 100%; height: 100%; object-fit: cover; opacity: 0; animation: fade3 6s infinite; }
    .hero img:nth-child(1) { animation-delay: 0s; }
    .hero img:nth-child(2) { animation-delay: 2s; }
    .hero img:nth-child(3) { animation-delay: 4s; }
    @keyframes fade3 { 0% { opacity: 0; } 15% { opacity: 0.6; } 33% { opacity: 0.6; } 48% { opacity: 0; } 100% { opacity: 0; } }

    /* HERO DETEKSI & CCTV (2 Gambar) */
    .hero-deteksi { position: relative; width: 100vw; height: 100vh; background-color: #002147; overflow: hidden; }
    .hero-deteksi img { position: absolute; width: 100%; height: 100%; object-fit: cover; opacity: 0; animation: fade2 8s infinite; }
    .hero-deteksi img:nth-child(1) { animation-delay: 0s; }
    .hero-deteksi img:nth-child(2) { animation-delay: 4s; }
    @keyframes fade2 { 
        0% { opacity: 0; } 
        10% { opacity: 0.7; } 
        40% { opacity: 0.7; } 
        50% { opacity: 0; } 
        100% { opacity: 0; } 
    }

    .hero-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: white; z-index: 10; width: 100%; }
    .hero-text h1 { font-size: 70px; font-weight: 800; text-transform: uppercase; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    .hero-text p { font-size: 24px; border-top: 3px solid #f39c12; display: inline-block; padding-top: 10px; margin-top: 10px; color: #f39c12; font-weight: bold; }

    /* KARTU TOKOH */
    .tokoh-card { background-color: #ffffff; padding: 22px; border-radius: 10px; border-left: 5px solid #f39c12; box-shadow: 0 4px 12px rgba(0,0,0,0.05); height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
    .tokoh-header { display: flex; align-items: center; gap: 15px; margin-bottom: 12px; }
    .tokoh-img { width: 65px; height: 65px; border-radius: 50%; object-fit: cover; border: 2px solid #f39c12; }
    .tokoh-name { font-weight: bold; color: #002147; font-size: 15px; line-height: 1.3; }
    .tokoh-title { font-size: 12px; color: #666; margin-top: 2px; }
    .tokoh-quote { font-size: 13px; color: #444; font-style: italic; line-height: 1.5; }

    /* BERITA */
    .news-card { background-color: #ffffff; padding: 25px; border-radius: 8px; border-left: 5px solid #f39c12; box-shadow: 0 4px 12px rgba(0,0,0,0.06); height: 100%; }
    .news-title { color: #002147; font-weight: bold; font-size: 17px; margin-bottom: 10px; }
    .news-excerpt { color: #666666; font-size: 14px; margin-bottom: 15px; }

    /* BANNER MODEL YOLOv8 */
    .single-model-banner { background: linear-gradient(135deg, #002147 0%, #1e3c72 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(0,33,71,0.15); margin-bottom: 30px; }
    .single-model-banner h2 { color: #f39c12; font-size: 26px; font-weight: 800; margin: 0 0 8px 0; }
    .single-model-banner p { color: #e0e0e0; font-size: 14px; margin: 0; }

    /* PROFIL ELEGAN */
    .profile-card-elegan { background-color: #ffffff; border-radius: 16px; padding: 35px; box-shadow: 0 10px 30px rgba(0,33,71,0.06); border: 1px solid #eef2f7; display: flex; align-items: center; gap: 35px; }
    .profile-avatar { width: 180px; height: 180px; border-radius: 50%; object-fit: cover; border: 4px solid #002147; box-shadow: 0 6px 15px rgba(0,0,0,0.1); }
    .profile-info h3 { color: #002147; font-size: 28px; font-weight: 800; margin: 0 0 10px 0; }
    .profile-info p { color: #555; font-size: 18px; margin: 6px 0; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. NAVBAR DENGAN STATE MANAGEMENT ---
st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)

col_nav_left, col_nav_mid, col_nav_right = st.columns([1.5, 6, 1.5])

with col_nav_left:
    if logo_polri and logo_korlantas:
        st.markdown(f'''
            <div style="display:flex; gap:15px; align-items:center; justify-content:flex-end; padding-right:15px; height:100%;">
                <img src="data:image/png;base64,{logo_polri}" width="75" style="object-fit: contain;">
                <img src="data:image/png;base64,{logo_korlantas}" width="75" style="object-fit: contain;">
            </div>
        ''', unsafe_allow_html=True)

with col_nav_mid:
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
            "container": {"background-color": "rgba(0, 33, 71, 1)", "padding": "10px", "border-radius": "30px", "width": "100%", "box-shadow": "0 4px 10px rgba(0,0,0,0.15)", "margin": "0"},
            "nav-link": {"font-size": "15px", "color": "white", "font-weight": "600", "padding": "10px", "border-radius": "20px"},
            "nav-link-selected": {"background-color": "#f39c12", "color": "#002147", "font-weight": "bold", "border-radius": "20px"}
        }
    )
    
    if selected != st.session_state.current_selected_menu:
        st.session_state.current_selected_menu = selected
        st.rerun()

with col_nav_right:
    if logo_hut:
        st.markdown(f'''
            <div style="display:flex; align-items:center; justify-content:flex-start; padding-left:15px; height:100%;">
                <img src="data:image/png;base64,{logo_hut}" width="75" style="object-fit: contain;">
            </div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

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
        body { margin: 0; padding: 0; background-color: #f4f6f9; } 
        .counter-wrapper { display: flex; justify-content: space-around; background: #ffffff; padding: 30px 0; font-family: sans-serif; color: #333; box-shadow: 0 4px 6px rgba(0,0,0,0.02); } 
        .counter-box { text-align: center; border: 1px solid #e0e0e0; background-color: #ffffff; padding: 25px 15px; width: 23%; border-radius: 8px; } 
        .counter-box h3 { font-size: 38px; color: #f39c12; margin: 0 0 10px 0; font-weight: 800; } 
        .counter-box p { font-size: 12px; margin: 0; font-weight: bold; text-transform: uppercase; color: #002147; line-height: 1.4; }
    </style>
    <div class="counter-wrapper">
        <div class="counter-box">
            <h3 class="count" data-target="12" data-unit=" M">0 M</h3><p>(Maks. Panjang Truk Tunggal<br>PP No. 55/2012)</p>
        </div>
        <div class="counter-box">
            <h3 class="count" data-target="2.5" data-unit=" M">0 M</h3><p>(Batas Maksimal Lebar Kendaraan<br>PP No. 55/2012)</p>
        </div>
        <div class="counter-box">
            <h3 class="count" data-target="4.2" data-unit=" M">0 M</h3><p>(Batas Maks. Tinggi + Muatan<br>PP No. 55/2012)</p>
        </div>
        <div class="counter-box">
            <h3 class="count" data-target="24" data-unit=" Ton">0 Ton</h3><p>(Batas JBB Truk 3 Sumbu / Tronton<br>SE Menhub & UU 22/2009)</p>
        </div>
    </div>
    <script>
        const counters = document.querySelectorAll('.count'); 
        const speed = 40; 
        
        counters.forEach(counter => { 
            const target = +counter.getAttribute('data-target'); 
            const unit = counter.getAttribute('data-unit');
            const inc = target / speed; 
            let current = 0; 
            
            const updateCount = () => { 
                current += inc; 
                if (current < target) { 
                    if (target % 1 === 0) {
                        counter.innerText = Math.ceil(current) + unit;
                    } else {
                        counter.innerText = current.toFixed(1) + unit;
                    }
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
    c_l, c_mid, c_r = st.columns([1, 2, 1])
    with c_mid:
        st.markdown("<h3 style='text-align:center; color:#002147;'>Presentase Kecelakaan Berdasarkan Jenis Kendaraan</h3>", unsafe_allow_html=True)
        kategori = ['Angkutan Barang (ODOL)', 'Angkutan Orang', 'Mobil Penumpang', 'Kendaraan Listrik']
        persentase = [10.5, 8.0, 2.4, 0.2]
        warna = ['#f39c12', '#002147', '#3498db', '#2ecc71']
        
        fig = go.Figure(data=[go.Bar(x=kategori, y=persentase, text=[f"{val}%" for val in persentase], textposition='auto', marker_color=warna)])
        fig.update_layout(
            template="plotly_white", 
            height=400, 
            margin=dict(l=0,r=0,t=30,b=0), 
            yaxis=dict(title="Persentase (%)", gridcolor='#e0e0e0'),
            xaxis=dict(gridcolor='#e0e0e0'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        st.markdown("""
            <p style='text-align:center; font-size:14px; color:#666;'>
                Sumber: <a href='https://www.instagram.com/p/DL9w3-HJ1x6/?utm_source=ig_web_copy_link&stkn=NTc4MTIwNjQ2YQ==' target='_blank' style='color:#f39c12; font-weight:bold; text-decoration:none;'><i>Kemenkoinfra 2025 - Keselamatan Jalan Untuk Indonesia</i></a>
            </p>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<h3 style='text-align:center; color:#002147;'>Pandangan & Komitmen Penegakan Keselamatan Jalan</h3><br>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"""
            <div class="tokoh-card">
                <div>
                    <div class="tokoh-header">
                        <img src="data:image/jpeg;base64,{tokoh1_b64}" class="tokoh-img">
                        <div>
                            <div class="tokoh-name">Irjen Pol. Wibowo, S.I.K., M.Hum.</div>
                            <div class="tokoh-title">Kepala Korps Lalu Lintas (Kakorlantas) Polri</div>
                        </div>
                    </div>
                    <div class="tokoh-quote">"Korlantas Polri akan terus mendorong transformasi digital. Penegakan hukum berbasis teknologi seperti E-TLE adalah kunci untuk meminimalisir interaksi langsung dan memastikan penindakan pelanggaran yang berpotensi memicu fatalitas tinggi berjalan secara transparan."</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="tokoh-card">
                <div>
                    <div class="tokoh-header">
                        <img src="data:image/jpeg;base64,{tokoh2_b64}" class="tokoh-img">
                        <div>
                            <div class="tokoh-name">Dudy Purwagandhi, S.H.</div>
                            <div class="tokoh-title">Menteri Perhubungan Republik Indonesia</div>
                        </div>
                    </div>
                    <div class="tokoh-quote">"Keselamatan transportasi adalah harga mati. Pelanggaran dimensi dan muatan berlebih (ODOL) bukan hanya membahayakan nyawa masyarakat pengguna jalan, tetapi juga sangat membebani anggaran negara akibat kerusakan infrastruktur jalan."</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
            <div class="tokoh-card">
                <div>
                    <div class="tokoh-header">
                        <img src="data:image/png;base64,{tokoh3_b64}" class="tokoh-img">
                        <div>
                            <div class="tokoh-name">KOMBESPOL ARIE PRASETYA SYAF'AT, S.I.K., M.H.</div>
                            <div class="tokoh-title">Dirlantas Polda D.I. Yogyakarta</div>
                        </div>
                    </div>
                    <div class="tokoh-quote">"Kesadaran dan budaya tertib berlalu lintas harus menjadi fondasi utama. Pemanfaatan teknologi pemantauan lalu lintas sangat membantu kepolisian untuk menekan angka kecelakaan angkutan barang."</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="tokoh-card">
                <div>
                    <div class="tokoh-header">
                        <img src="data:image/jpeg;base64,{tokoh4_b64}" class="tokoh-img">
                        <div>
                            <div class="tokoh-name">Inspektur Jenderal Polisi Anggoro Sukartono, S.I.K.</div>
                            <div class="tokoh-title">Kapolda D.I. Yogyakarta</div>
                        </div>
                    </div>
                    <div class="tokoh-quote">"Yogyakarta adalah etalase budaya dan pariwisata Indonesia. Kami berkomitmen untuk menghadirkan keamanan, kenyamanan, dan keselamatan berlalu lintas bagi seluruh warga maupun wisatawan tanpa kompromi."</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("<br><br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<h3 style='text-align:center; color:#002147;'>Berita Terkini Truk ODOL</h3><br>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1: st.markdown("""<div class="news-card"><div class="news-title">Ratusan Ribu Truk Diperiksa, Pelanggaran ODOL Masih Tinggi</div><div class="news-excerpt">Operasi penertiban kendaraan bermuatan lebih terus digencarkan...</div><a href="https://otomotif.kompas.com/read/2026/04/06/102200715/ratusan-ribu-truk-diperiksa-pelanggaran-odol-masih-tinggi" target="_blank" style="color:#f39c12; text-decoration:none; font-weight:bold;">Baca Selengkapnya →</a></div>""", unsafe_allow_html=True)
    with n2: st.markdown("""<div class="news-card"><div class="news-title">Daftar Kecelakaan yang Disebabkan Truk ODOL</div><div class="news-excerpt">Catatan insiden fatal di berbagai ruas jalan nasional dan tol yang berakar dari hilangnya kendali...</div><a href="https://otomotif.kompas.com/read/2025/06/09/171200015/daftar-kecelakaan-yang-disebabkan-truk-odol" target="_blank" style="color:#f39c12; text-decoration:none; font-weight:bold;">Baca Selengkapnya →</a></div>""", unsafe_allow_html=True)
    with n3: st.markdown("""<div class="news-card"><div class="news-title">Kecelakaan Bus dan Truk di Tol Malang Lagi-lagi Karena ODOL</div><div class="news-excerpt">Tabrakan parah yang melibatkan transportasi publik kembali terjadi akibat rem blong tonase berlebih...</div><a href="https://otomotif.kompas.com/read/2024/12/24/132100515/kecelakaan-bus-dan-truk-di-tol-malang-lagi-lagi-karena-truk-odol" target="_blank" style="color:#f39c12; text-decoration:none; font-weight:bold;">Baca Selengkapnya →</a></div>""", unsafe_allow_html=True)
    st.write("<br><br>", unsafe_allow_html=True)

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
                <p>Analisis Cerdas Dimensi & Muatan Kendaraan Barang Berbasis Komputer</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 50px 40px 20px 40px;'>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #ffffff; border-radius: 16px; padding: 35px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid #eaeaea; margin-bottom: 35px;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px; border-bottom: 2px solid #f4f6f9; padding-bottom: 15px;">
                <div style="background: #002147; color: #f39c12; width: 45px; height: 45px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold;">i</div>
                <div>
                    <h3 style="margin: 0; color: #002147; font-size: 20px; font-weight: 800;">Panduan & Cara Penggunaan Sistem Deteksi</h3>
                    <p style="margin: 3px 0 0 0; color: #666; font-size: 13px;">Kenali cara kerja aplikasi cerdas pengawas truk bermuatan lebih (ODOL)</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_info1, col_info2 = st.columns(2, gap="medium")
    with col_info1:
        st.info("**Apa Tujuan Sistem Ini?**\n\nAplikasi ini dirancang untuk membantu petugas kepolisian mengenali truk yang membawa muatan berlebih atau dimensi tidak wajar (ODOL) secara otomatis lewat foto atau video, guna mencegah kecelakaan fatal di jalan raya.")
    with col_info2:
        st.success("**Apa Hasilnya Nanti?**\n\nSistem akan memunculkan kotak penanda (*bounding box*) berwarna pada kendaraan di foto/video beserta label jenis truk dan tingkat akurasi kecerdasan buatannya secara instan.")

    st.write("<br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #002147; font-weight: 800;'>Langkah-Langkah Mudah Penggunaan:</h4>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #f8fafc; border-left: 4px solid #f39c12; padding: 15px 20px; border-radius: 8px; margin-bottom: 12px;">
            <b style="color: #002147;">1. Siapkan File Uji</b><br>
            <span style="color: #555; font-size: 14px;">Pilih foto (format JPG, PNG) atau video pendek (format MP4) yang memperlihatkan tampak samping atau depan truk di jalan.</span>
        </div>
        <div style="background: #f8fafc; border-left: 4px solid #f39c12; padding: 15px 20px; border-radius: 8px; margin-bottom: 12px;">
            <b style="color: #002147;">2. Unggah (Upload) ke Sistem</b><br>
            <span style="color: #555; font-size: 14px;">Tekan tombol unggah file di bawah, lalu pilih file dari perangkat komputer atau HP Anda.</span>
        </div>
        <div style="background: #f8fafc; border-left: 4px solid #f39c12; padding: 15px 20px; border-radius: 8px; margin-bottom: 25px;">
            <b style="color: #002147;">3. Lihat Hasil Deteksi Cerdas</b><br>
            <span style="color: #555; font-size: 14px;">Tunggu beberapa detik saat kecerdasan buatan memindai gambar, lalu lihat hasil kotak penanda pelanggaran yang muncul secara otomatis.</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="single-model-banner">
            <h2>Didukung oleh Model YOLOv8 Custom ONNX</h2>
            <p>Model kecerdasan buatan terlatih khusus dengan tingkat akurasi tinggi dan kecepatan pemrosesan real-time.</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; color:#002147;'>Unggah File Foto / Video Kendaraan</h3>", unsafe_allow_html=True)
    c_up1, c_up2, c_up3 = st.columns([1, 2, 1])
    with c_up2:
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'webp', 'mp4', 'avi', 'mov', 'mpeg4'])

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > 50:
            st.error("Ukuran file melebihi batas maksimal 50 MB.")
        else:
            st.divider()
            c_res1, c_res2, c_res3 = st.columns([1, 2, 1])
            with c_res2:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                is_video_file = file_ext in ['mp4', 'avi', 'mov', 'mkv', 'mpeg4']
                
                status_msg = "Video sedang diproses..." if is_video_file else "Foto sedang diproses..."
                
                with st.spinner(status_msg):
                    ftype, res_file = process_detection(uploaded_file)
                    
                if ftype == "video":
                    st.video(res_file)
                else:
                    st.image(res_file, use_container_width=True)

            st.divider()
            
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# HALAMAN 3: CCTV REAL-TIME (BEBAS WARNING Kuning)
# ==========================================
elif st.session_state.current_selected_menu == "CCTV Real-Time":
    st.markdown(f"""
        <div class="hero-deteksi">
            <img src="data:image/jpeg;base64,{img4}">
            <img src="data:image/jpeg;base64,{img5}">
            <div class="hero-text">
                <h1>PEMANTAUAN POS PANTAU</h1>
                <p>Live CCTV & Analisis Pelanggaran Kendaraan Barang Real-Time</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 50px 40px 20px 40px;'>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #ffffff; border-radius: 16px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid #eaeaea; margin-bottom: 30px; text-align: center;">
            <h3 style="color: #002147; font-weight: 800; margin-top: 0;">Panduan Live Streaming Pos Pantau</h3>
            <p style="color: #555; font-size: 15px; margin-bottom: 0; line-height: 1.6;">
                <b>Arahkan kamera ke objek kendaraan atau jalan raya.</b> Tekan tombol <b>Mulai Kamera</b> di bawah untuk mengaktifkan pemantauan real-time, dan tekan tombol <b>Matikan Kamera</b> kapan saja jika ingin menghentikannya.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if 'cctv_active' not in st.session_state:
        st.session_state.cctv_active = False

    c_b1, c_b2, c_b3 = st.columns([1, 1, 1])
    with c_b1:
        if st.button("Mulai Kamera", use_container_width=True):
            st.session_state.cctv_active = True
    with c_b2:
        if st.button("Matikan Kamera", use_container_width=True):
            st.session_state.cctv_active = False

    st.write("<br>", unsafe_allow_html=True)
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
            
            if len(results[0].boxes) > 0:
                temp_snap = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                cv2.imwrite(temp_snap, annotated_frame)
                recap_data.append({"waktu": time.strftime("%H:%M:%S"), "path": temp_snap, "status": "Kendaraan Terdeteksi"})

            time.sleep(0.03)

        cap.release()
    else:
        frame_placeholder.info("Kamera sedang dimatikan. Klik 'Mulai Kamera' untuk mengaktifkan kembali.")

    if recap_data and not st.session_state.cctv_active:
        st.success("Sesi pemantauan selesai. Laporan PDF siap diunduh!")
        pdf_recap_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
        doc_r = SimpleDocTemplate(pdf_recap_path, pagesize=letter)
        story_r = []
        styles_r = getSampleStyleSheet()
        
        story_r.append(Paragraph("REKAPITULASI PENINDAKAN CCTV REAL-TIME", styles_r['Title']))
        story_r.append(Paragraph("Korps Lalu Lintas Polri - E-TLE System", styles_r['Normal']))
        story_r.append(Spacer(1, 15))
        
        for item in recap_data[-5:]:
            story_r.append(Paragraph(f"<b>Waktu:</b> {item['waktu']} | <b>Status:</b> {item['status']}", styles_r['Heading2']))
            try:
                story_r.append(RLImage(item['path'], width=280, height=200))
            except:
                pass
            story_r.append(Spacer(1, 10))
        
        doc_r.build(story_r)
        
        with open(pdf_recap_path, "rb") as f:
            st.download_button("Klik untuk Unduh Rekap Laporan PDF", f, file_name="Rekap_CCTV_ETLE.pdf", mime="application/pdf")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# HALAMAN 4: TENTANG
# ==========================================
elif st.session_state.current_selected_menu == "Tentang":
    st.markdown("<div style='padding-top: 15px; padding-bottom: 30px; padding-left: 80px; padding-right: 80px;'>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align:center; color:#002147; font-weight:800; margin-bottom: 20px;'>Tentang Pengembang Sistem</h1>", unsafe_allow_html=True)
    
    c_col1, c_col2, c_col3 = st.columns([1, 4, 1])
    with c_col2:
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