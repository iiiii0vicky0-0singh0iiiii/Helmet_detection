import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🪖 Helmet Detection App")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.header("📸 Image Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    results = model(img_np)
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Image")

# -----------------------------
# LIVE VIDEO DETECTION
# -----------------------------
st.header("🎥 Live Camera Detection")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run detection
        results = model(img)

        # Draw boxes
        annotated = results[0].plot()

        return annotated

webrtc_streamer(
    key="helmet-detection",
    video_processor_factory=VideoProcessor
)
