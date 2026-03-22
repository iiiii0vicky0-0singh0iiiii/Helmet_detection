import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load model
model = YOLO("best.pt")

st.title("🪖 Helmet Detection Dashboard")

option = st.sidebar.selectbox(
    "Choose Mode",
    ("Image Upload", "Video Upload", "Live Camera")
)

# ---------------- IMAGE ----------------
if option == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        results = model(img_np)
        annotated = results[0].plot()

        st.image(annotated, caption="Detection Result")

# ---------------- VIDEO ----------------
elif option == "Video Upload":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR")

        cap.release()

# ---------------- LIVE CAMERA ----------------
elif option == "Live Camera":
    run = st.checkbox("Start Camera")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        stframe.image(annotated, channels="BGR")

    cap.release()