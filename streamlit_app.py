import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Helmet Detection Dashboard", layout="wide")

st.title("🪖 Helmet Detection Dashboard")

# ---------------- LOAD MODEL (CACHE) ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")
st.sidebar.write("Upload an image to detect helmets")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # Run detection
    results = model(img_np)
    annotated = results[0].plot()

    # ---------------- COUNT ----------------
    helmet = 0
    no_helmet = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            helmet += 1
        else:
            no_helmet += 1

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_np, caption="Original Image", use_container_width=True)

    with col2:
        st.image(annotated, caption="Detected Image", use_container_width=True)

    # ---------------- STATS ----------------
    st.success(f"🟢 Helmet: {helmet}")
    st.error(f"🔴 No Helmet: {no_helmet}")

    # ---------------- DOWNLOAD BUTTON ----------------
    result_image = Image.fromarray(annotated)
    buf = io.BytesIO()
    result_image.save(buf, format="JPEG")

    st.download_button(
        label="📥 Download Result Image",
        data=buf.getvalue(),
        file_name="helmet_detection_result.jpg",
        mime="image/jpeg"
    )
