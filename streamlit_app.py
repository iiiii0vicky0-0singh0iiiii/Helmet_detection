import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Helmet Detection Dashboard",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Helmet Detection Dashboard")
st.write("Upload an image to detect people wearing or not wearing helmets.")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("best.pt")

    if not model_path.exists():
        st.error("❌ Model file 'best.pt' was not found.")
        st.stop()

    return YOLO(str(model_path))


model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

st.sidebar.info(
    "Upload an image containing workers to detect helmet usage."
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image and force RGB
    image = Image.open(uploaded_file).convert("RGB")

    img_np = np.array(image)

    # --------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------
    with st.spinner("🔍 Detecting helmets..."):

        results = model.predict(
            source=img_np,
            conf=confidence,
            verbose=False
        )

    result = results[0]

    # YOLO annotated image
    annotated = result.plot()

    # --------------------------------------------------
    # COUNT DETECTIONS
    # --------------------------------------------------
    helmet = 0
    no_helmet = 0

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())

            class_name = model.names[class_id].lower()

            if class_name in ["helmet", "hardhat", "hard_hat"]:
                helmet += 1

            elif class_name in [
                "no helmet",
                "no_helmet",
                "no-helmet",
                "no hardhat",
                "nohardhat"
            ]:
                no_helmet += 1

    # --------------------------------------------------
    # SHOW IMAGES
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Detection Result")

        # Ultralytics plot() returns BGR
        st.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    st.subheader("📊 Detection Summary")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "🟢 Helmet",
            helmet
        )

    with metric2:
        st.metric(
            "🔴 No Helmet",
            no_helmet
        )

    with metric3:
        st.metric(
            "👷 Total Detected",
            helmet + no_helmet
        )

    # Warning
    if no_helmet > 0:
        st.error(
            f"⚠️ Safety Alert: {no_helmet} person(s) detected without helmet!"
        )

    elif helmet > 0:
        st.success(
            "✅ All detected workers are wearing helmets."
        )

    else:
        st.warning(
            "⚠️ No helmet-related objects were detected."
        )
