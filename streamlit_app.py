import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # or yolov8n.pt for testing

model = load_model()

st.title("🪖 Smart Helmet Detection System")

# -----------------------------
# Create folder for captures
# -----------------------------
if not os.path.exists("captures"):
    os.makedirs("captures")

# -----------------------------
# 🔥 DRAW BOXES FUNCTION
# -----------------------------
def draw_boxes(results, frame):
    boxes = results[0].boxes

    helmet_count = 0
    no_helmet_count = 0

    if boxes is None:
        return frame, helmet_count, no_helmet_count

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = model.names[cls_id]

        if label.lower() == "no_helmet":
            color = (0, 0, 255)
            no_helmet_count += 1
        else:
            color = (0, 255, 0)
            helmet_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame, helmet_count, no_helmet_count

# -----------------------------
# 📸 IMAGE DETECTION
# -----------------------------
st.header("📸 Image Detection")

uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(img)

    results = model(img_np)
    frame, h, nh = draw_boxes(results, img_np)

    st.write(f"🟢 Helmet: {h} | 🔴 No Helmet: {nh}")
    st.image(frame)

# -----------------------------
# 🎥 LIVE CAMERA
# -----------------------------
st.header("🎥 Live Camera Detection")

capture_flag = st.button("📸 Capture Image")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(img, (640, 480))

        results = model(img)
        frame_out, h, nh = draw_boxes(results, img)

        # Show count
        cv2.putText(frame_out, f"H:{h} NH:{nh}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 🔥 CAPTURE IMAGE
        if capture_flag:
            filename = f"captures/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame_out)

        return frame_out

webrtc_streamer(
    key="helmet",
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
        frame, h, nh = draw_boxes(results, frame)

        cv2.putText(frame, f"H:{h} NH:{nh}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
