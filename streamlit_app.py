import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# Load Model (only once)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🪖 Helmet Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    
    # Load image safely
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy (FIX)
    img_np = np.array(img)

    # DEBUG (optional)
    st.write("Image Shape:", img_np.shape)

    # Run model (FIXED)
    results = model(img_np)

    # Plot result
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_column_width=True)
