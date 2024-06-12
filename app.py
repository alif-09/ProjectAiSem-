import streamlit as st
import cv2
import base64
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from keras.models import load_model

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(layout="wide")

# Kelas untuk memproses frame video
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = "No sign recognized yet"
        self.description = "Description will be here."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Logika pengenalan rambu lalu lintas
        self.result_text = "Example Sign"  # Ganti dengan hasil pengenalan sebenarnya
        self.description = "This is an example description for the recognized traffic sign."  # Ganti dengan deskripsi rambu sebenarnya

        # Menampilkan hasil pada frame video
        cv2.putText(img, self.result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

    def get_result(self):
        return self.result_text

    def get_description(self):
        return self.description

# Inisialisasi state untuk video processor
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None

# Fungsi untuk menampilkan custom CSS
def load_css():
    css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #0e1117;
            color: #f4f4f4;
        }
        .container {
            height: 80vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 40px;
            text-align: center;
        }
        .intro {
            margin: 0;
            padding: 0;
        }
        .intro p {
            width: 70%;
            margin: 0 auto;
        }
        .intro button {
            color: #fff;
            text-align: center;
            vertical-align: middle;
            cursor: pointer;
            background-color: #ff4b4b;
            border: 1px solid white;
            padding: .375rem .75rem;
            line-height: 1.5;
            border-radius: .25rem;
            margin-top: 2rem;
        }
        button:hover {
            background-color: #ff2e2e;
        }
        #traffic-signs {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .marquee {
            width: 100%;
            overflow: hidden;
            position: relative;
            margin-bottom: 20px;
        }
        .marquee-content {
            display: flex;
            width: 200%;
            animation: marquee 30s linear infinite;
        }
        .marquee img {
            width: 50px;
            margin: 10px;
        }
        .reverse .marquee-content {
            animation-direction: reverse;
        }
        .centered-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 60%;
            margin: 0 auto;
        }
        @keyframes marquee {
            0% {
                transform: translateX(0);
            }
            100% {
                transform: translateX(-50%);
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Fungsi untuk mengonversi gambar ke base64
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Fungsi untuk menampilkan custom HTML
def load_html():
    row1_images = ''.join([f'<img src="data:image/png;base64,{img_to_base64(f"static/images/{i}.png")}" alt="Traffic Sign">' for i in range(1, 22)])
    row2_images = ''.join([f'<img src="data:image/png;base64,{img_to_base64(f"static/images/{i}.png")}" alt="Traffic Sign">' for i in range(22, 43)])

    # Menggabungkan baris gambar untuk infinite loop
    row1_images += row1_images
    row2_images += row2_images

    html = f"""
    <div class="container"> 
        <section class="intro">
            <h1>Traffic Sign Recognition</h1>
            <p>Welcome to our website, a center for traffic sign recognition. We utilize advanced AI technology to quickly and accurately identify and interpret various types and shapes of traffic signs. Our system is designed to learn and adapt to different types of traffic signs, allowing us to provide fast and precise information. To try our application, you can press the button below. Thank you.</p>
            <button class="scroll-button" onclick="scrollToSection()">Try it now</button>
        </section>
        <section id="traffic-signs">
            <div class="marquee">
                <div class="marquee-content">
                    <!-- Baris pertama gambar -->
                    {row1_images}
                </div>
            </div>
            <div class="marquee reverse">
                <div class="marquee-content">
                    <!-- Baris kedua gambar -->
                    {row2_images}
                </div>
            </div>
        </section>
    </div>
    <script>
        function scrollToSection() {{
            document.getElementById('video_feed_section').scrollIntoView({{ behavior: 'smooth' }});
        }}
    </script>
    """
    # Menampilkan HTML dengan gambar
    st.markdown(html, unsafe_allow_html=True)

# Memuat dan menampilkan custom CSS dan HTML
load_css()
load_html()

# Layout untuk video feed dan hasil pengenalan
st.markdown('<div id="video_feed_section" class="centered-content">', unsafe_allow_html=True)

cols = st.columns([1, 2, 1])  # Kolom pertama dan ketiga untuk memberikan jarak, kolom kedua untuk konten

with cols[1]:  # Mengatur agar konten berada di tengah
    # st.columns(3)[1].header("hello world")
    st.markdown("<h4 style='text-align: center; color: white;'>Recognition Test</h4>", unsafe_allow_html=True)
    ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor)
    if ctx.video_processor:
        st.session_state.video_processor = ctx.video_processor

    st.markdown("<h4 style='text-align: center; color: white;'>Recognition Results</h4>", unsafe_allow_html=True)
    if st.session_state.video_processor:
        result = st.session_state.video_processor.get_result()
        description = st.session_state.video_processor.get_description()
        st.write(f"**Nama Rambu:** {result}")
        st.write(f"**Penjelasan:** {description}")
    else:
        st.write("Belum ada hasil pengenalan")

st.markdown('</div>', unsafe_allow_html=True)
