import streamlit as st
from ultralytics import YOLO
import cv2
import base64
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from streamlit_option_menu import option_menu
import threading

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(
    layout="wide",
    page_title="Traffuck"
)

# model = YOLO('best10.pt')

class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons", 
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles", 
    "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left", 
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", 
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", 
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", 
    "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

selected = option_menu(
    menu_title=None,  
    options=["Home", "Projects", "Contact"],  
    icons=["house", "book", "envelope"], 
    menu_icon="cast",
    default_index=0, 
    orientation="horizontal",
)

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
            <button>Try it now</button>
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
    """
    # Menampilkan HTML dengan gambar
    st.markdown(html, unsafe_allow_html=True)

# Layout untuk halaman Home
if selected == "Home":
    load_css()
    load_html()

# Layout untuk halaman Projects
if selected == "Projects":
    st.markdown('<div id="video_feed_section" class="centered-content">', unsafe_allow_html=True)

    cols = st.columns([1, 2, 1])  # Kolom pertama dan ketiga untuk memberikan jarak, kolom kedua untuk konten

    with cols[1]:  # Mengatur agar konten berada di tengah
        st.markdown("<h4 style='text-align: center; color: white;'>Recognition Test</h4>", unsafe_allow_html=True)
        
        # Dropdown untuk memilih jenis inputan data
        input_type = st.selectbox("Pilih jenis input:", ["Camera", "File/Gambar"])
        
        if input_type == "Camera":
            ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor)
            if ctx.video_processor:
                st.session_state.video_processor = ctx.video_processor
        elif input_type == "File/Gambar":
            uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                st.image(img, channels="BGR")
                
                # Logika pengenalan rambu lalu lintas pada gambar
                st.session_state.video_processor = VideoProcessor()
                st.session_state.video_processor.result_text = "Example Sign"  # Ganti dengan hasil pengenalan sebenarnya
                st.session_state.video_processor.description = "This is an example description for the recognized traffic sign."  # Ganti dengan deskripsi rambu sebenarnya

        st.markdown("<h4 style='text-align: center; color: white;'>Recognition Results</h4>", unsafe_allow_html=True)
        if st.session_state.video_processor:
            result = st.session_state.video_processor.get_result()
            description = st.session_state.video_processor.get_description()
            st.write(f"**Nama Rambu:** {result}")
            st.write(f"**Penjelasan:** {description}")
        else:
            st.write("Belum ada hasil pengenalan")

    st.markdown('</div>', unsafe_allow_html=True)

# Layout untuk halaman Contact
if selected == "Contact":
    st.markdown("<h4 style='text-align: center; color: white;'>Contact Information</h4>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Berikut adalah detail informasi dari kami selaku pembuat aplikasi ini</h6>", unsafe_allow_html=True)
    
    team_members = [
        {"name": "Rio", "email": "rio22003@mail.unpad.ac.id", "image": "static/images/angga.png"},
        {"name": "Angga", "email": "angga22004@mail.unpad.ac.id", "image": "static/images/angga.png"},
        {"name": "Alif", "email": "alif22001@mail.unpad.ac.id", "image": "static/images/angga.png"},
        {"name": "Giast", "email": "giast22001@mail.unpad.ac.id", "image": "static/images/angga.png"},
        {"name": "Danendra", "email": "danen22001@mail.unpad.ac.id", "image": "static/images/angga.png"},
    ]

    cols = st.columns(len(team_members))

    for i, member in enumerate(team_members):
        image_base64 = img_to_base64(member["image"])
        with cols[i]:
            st.markdown(f"<div style='text-align: center'><img src='data:image/png;base64,{image_base64}' width='100'></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; margin: 0.5rem 0 '> {member['name']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; margin: 0 '> {member['email']}</p>", unsafe_allow_html=True)




