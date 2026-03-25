import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# Load Model (use lightweight)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # change to yolov8n.pt for faster testing

model = load_model()

st.title("🪖 Smart Helmet Detection System")

# -----------------------------
# 📸 IMAGE DETECTION
# -----------------------------
st.header("📸 Image Detection")

uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(img)

    results = model(img_np)

    annotated = results[0].plot()

    # 🔢 Count objects
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0

    st.write(f"👥 Total Detections: {count}")

    st.image(annotated, caption="Detection Result")

# -----------------------------
# 🎥 LIVE CAMERA
# -----------------------------
st.header("🎥 Live Camera Detection")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize (important for speed)
        img = cv2.resize(img, (640, 480))

        results = model(img)

        annotated = results[0].plot()

        # 🔢 Count
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0

        # 🧠 Add text
        cv2.putText(
            annotated,
            f"Detections: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return annotated

webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    async_processing=True
)

# -----------------------------
# 📹 VIDEO / CCTV FILE
# -----------------------------
st.header("📹 CCTV / Video Detection")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if video_file:
    tfile = open("temp.mp4", "wb")
    tfile.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame)

        annotated = results[0].plot()

        # 🔢 Count
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0

        cv2.putText(
            annotated,
            f"Detections: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        stframe.image(annotated, channels="BGR")

    cap.release()
