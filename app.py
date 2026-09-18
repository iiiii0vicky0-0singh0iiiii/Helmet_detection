"""Streamlit application for helmet detection with a custom YOLO model."""

from __future__ import annotations

import io
import hashlib
import os
import tempfile
from pathlib import Path

# Keep model/framework caches in Streamlit Cloud's temporary storage.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "helmet-detection-cache"
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_CACHE_ROOT / "ultralytics"))

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = ("best.pt", "yolov11nbest.pt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(
    page_title="HelmetGuard AI",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[YOLO, Path]:
    """Load the first available local YOLO model and cache it between reruns."""
    for filename in MODEL_CANDIDATES:
        model_path = APP_DIR / filename
        if model_path.is_file():
            return YOLO(str(model_path)), model_path
    expected = " or ".join(MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model not found. Add {expected} to the repository root.")


@st.cache_data(show_spinner=False)
def load_sample(path: str) -> bytes:
    """Read a bundled sample once and cache its bytes."""
    return Path(path).read_bytes()


def available_samples() -> list[Path]:
    sample_dir = APP_DIR / "images"
    if not sample_dir.is_dir():
        return []
    return sorted(
        path for path in sample_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_rgb_image(image_bytes: bytes) -> Image.Image:
    """Open uploaded bytes and normalize all supported image modes to RGB."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        return image.convert("RGB")


def run_detection(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    iou: float,
    image_size: int,
) -> tuple[Image.Image, pd.DataFrame, dict[str, int]]:
    """Run YOLO and return an annotated image, detection rows, and class totals."""
    result = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device="cpu",
        verbose=False,
    )[0]

    # Ultralytics plots in BGR; PIL/Streamlit expect RGB.
    plotted_bgr = result.plot()
    annotated = Image.fromarray(plotted_bgr[:, :, ::-1])

    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(result.names[class_id])
            score = float(box.conf.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            counts[class_name] = counts.get(class_name, 0) + 1
            rows.append(
                {
                    "Class": class_name,
                    "Confidence": round(score, 3),
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1),
                }
            )

    columns = ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
    return annotated, pd.DataFrame(rows, columns=columns), counts


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.markdown(
    """
    <style>
        .stApp { background: #f6f8fb; }
        .block-container { max-width: 1200px; padding-top: 2.2rem; }
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            color: white;
            background: linear-gradient(120deg, #071b2e 0%, #0c4a5a 55%, #0f766e 100%);
            box-shadow: 0 18px 45px rgba(7, 27, 46, .14);
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0 0 .45rem; font-size: clamp(2rem, 5vw, 3.3rem); }
        .hero p { margin: 0; color: #d9f5ee; font-size: 1.05rem; max-width: 720px; }
        .eyebrow { color: #7dd3c7; font-weight: 700; letter-spacing: .12em; font-size: .75rem; }
        div[data-testid="stMetric"] {
            background: white; border: 1px solid #e3e8ef; border-radius: 14px; padding: 1rem;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 16px; border: 1.5px dashed #79a9a2; background: #fbfefd;
        }
        .status-ok { color: #0f766e; font-weight: 650; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">WORKPLACE SAFETY · COMPUTER VISION</div>
        <h1>HelmetGuard AI</h1>
        <p>Upload a workplace image and instantly identify people with and without safety helmets.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    model, model_path = load_model()
except Exception as exc:
    st.error(f"The detection model could not be loaded: {exc}")
    st.info("Place `yolov11nbest.pt` in the same folder as `app.py`, then restart the app.")
    st.stop()

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Higher values show fewer, more certain detections.",
    )
    iou = st.slider(
        "Overlap threshold (IoU)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Controls when overlapping boxes are treated as duplicates.",
    )
    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768],
        value=640,
        help="Larger sizes can improve small-object detection but take longer.",
    )
    st.divider()
    st.markdown(
        f'<span class="status-ok">● Model ready</span><br><small>{model_path.name}</small>',
        unsafe_allow_html=True,
    )
    st.caption("Inference runs on CPU on Streamlit Community Cloud. The first prediction may take longer.")

st.subheader("Choose an image")
input_tab, sample_tab = st.tabs(["Upload image", "Try a sample"])

selected_bytes: bytes | None = None
selected_name = "image"
with input_tab:
    uploaded_file = st.file_uploader(
        "Drop a JPG, PNG, or WEBP image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        selected_bytes = uploaded_file.getvalue()
        selected_name = Path(uploaded_file.name).stem

with sample_tab:
    samples = available_samples()
    if samples:
        sample_choice = st.selectbox(
            "Sample image",
            samples,
            format_func=lambda path: path.stem.replace("-", " "),
        )
        use_sample = st.button("Use this sample", use_container_width=True)
        if use_sample:
            st.session_state["sample_path"] = str(sample_choice)
    else:
        st.info("No sample images are included in this repository.")

# An uploaded image takes precedence; otherwise remember the last selected sample.
if selected_bytes is None and st.session_state.get("sample_path"):
    sample_path = Path(st.session_state["sample_path"])
    if sample_path.is_file():
        selected_bytes = load_sample(str(sample_path))
        selected_name = sample_path.stem

if selected_bytes is None:
    st.info("Upload an image above or choose a sample to begin.")
    st.stop()

try:
    source_image = open_rgb_image(selected_bytes)
except Exception:
    st.error("This image could not be read. Please try a valid JPG, PNG, or WEBP file.")
    st.stop()

input_signature = hashlib.sha256(selected_bytes).hexdigest()
input_signature += f":{confidence}:{iou}:{image_size}"

preview_col, action_col = st.columns([1.55, 1], gap="large")
with preview_col:
    st.image(source_image, caption="Input image", use_container_width=True)
with action_col:
    st.markdown("### Ready to inspect")
    st.write(f"Image size: **{source_image.width} × {source_image.height} px**")
    st.write(f"Confidence threshold: **{confidence:.0%}**")
    detect_clicked = st.button(
        "Run helmet detection",
        type="primary",
        use_container_width=True,
    )
    st.caption("Images are processed only for this session and are not saved by the app.")

if detect_clicked:
    with st.spinner("Inspecting the image… the first run can take up to a minute."):
        try:
            annotated_image, detections, class_counts = run_detection(
                model, source_image, confidence, iou, image_size
            )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.stop()
    st.session_state["result"] = {
        "annotated_png": image_to_png_bytes(annotated_image),
        "detections": detections,
        "counts": class_counts,
        "source_name": selected_name,
        "input_signature": input_signature,
    }

result_data = st.session_state.get("result")
if result_data and result_data.get("input_signature") == input_signature:
    st.divider()
    st.subheader("Detection result")
    detections = result_data["detections"]
    counts = result_data["counts"]

    metric_columns = st.columns(3)
    metric_columns[0].metric("Total detections", len(detections))
    metric_columns[1].metric("With helmet", counts.get("accept-Helmet-", 0))
    metric_columns[2].metric("Without helmet", counts.get("non-Helmet-", 0))

    st.image(result_data["annotated_png"], use_container_width=True)

    download_col, csv_col = st.columns(2)
    with download_col:
        st.download_button(
            "Download annotated image",
            data=result_data["annotated_png"],
            file_name=f"{result_data['source_name']}_helmet_detection.png",
            mime="image/png",
            use_container_width=True,
        )
    with csv_col:
        st.download_button(
            "Download detections as CSV",
            data=detections.to_csv(index=False).encode("utf-8"),
            file_name=f"{result_data['source_name']}_detections.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=detections.empty,
        )

    if detections.empty:
        st.warning("No helmets or non-helmeted people were detected at this confidence threshold.")
    else:
        with st.expander("View detection details"):
            st.dataframe(
                detections,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="%.3f"
                    )
                },
            )
elif result_data:
    st.caption("The image or detection settings changed. Run detection again to refresh the result.")

st.divider()
st.caption("Built with Streamlit and Ultralytics YOLO · Predictions should support, not replace, workplace safety checks.")
