import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Helmet Detection Dashboard", layout="wide")

st.title("🪖 Helmet Detection Dashboard")

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
option = st.sidebar.selectbox(
    "Choose Mode",
    ("Image Upload", "Live Camera")
)

# ---------------- IMAGE UPLOAD ----------------
if option == "Image Upload":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        # Run detection
        results = model(img_np)
        annotated = results[0].plot()

        # Count detections
        helmet = 0
        no_helmet = 0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:
                helmet += 1
            else:
                no_helmet += 1

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_np, caption="Original Image")

        with col2:
            st.image(annotated, caption="Detected Image")

        # Show counts
        st.success(f"🟢 Helmet: {helmet}")
        st.error(f"🔴 No Helmet: {no_helmet}")

        # Download button
        st.download_button(
            "📥 Download Result",
            data=cv2.imencode('.jpg', annotated)[1].tobytes(),
            file_name="result.jpg"
        )

# ---------------- LIVE CAMERA ----------------
elif option == "Live Camera":

    run = st.checkbox("Start Camera")
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        stframe.image(annotated, channels="BGR")

    cap.release()
