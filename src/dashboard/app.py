"""Streamlit dashboard for satellite image analysis with Folium map integration."""

import base64
import io
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` imports work
# when launched via `streamlit run` (which doesn't add the project root).
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import folium  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from src.export.geotiff_writer import CLASS_COLORS  # noqa: E402
from src.models.change_detector import ChangeDetector  # noqa: E402
from src.models.classifier import LandUseClassifier  # noqa: E402
from src.utils.config import config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

st.set_page_config(
    page_title="Satellite Image Analysis",
    page_icon="\U0001f6f0\ufe0f",
    layout="wide",
)

DATA_DIR = _project_root / "data"
SAMPLE_DIR = DATA_DIR / "sample"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Image library
# ---------------------------------------------------------------------------


def scan_available_images() -> list[Path]:
    """Recursively find all image files under data/."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        f for f in DATA_DIR.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    )


def image_label(path: Path) -> str:
    """Short display label relative to data/."""
    return str(path.relative_to(DATA_DIR))


def load_image_from_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image from disk as (display HWC uint8, tensor CHW float32)."""
    img = Image.open(path).convert("RGB")
    image_np = np.array(img)
    image_tensor = image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image_np, image_tensor


def preprocess_image(uploaded_file: io.BytesIO) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess an uploaded file into (display HWC uint8, tensor CHW float32)."""
    img = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(img)
    image_tensor = image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image_np, image_tensor


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model() -> LandUseClassifier:
    """Load and cache the classification model."""
    model = LandUseClassifier(
        num_classes=config.get("data.num_classes", 10),
        architecture=config.get("model.architecture", "resnet50"),
        pretrained=False,
    )

    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


def classify_image(
    model: LandUseClassifier,
    image_tensor: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Run classification on a preprocessed image tensor."""
    with torch.no_grad():
        tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    probs_np = np.array(probs[0].tolist(), dtype=np.float32)
    return pred_class, confidence, probs_np


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def create_folium_map(
    center: tuple[float, float] = (51.5, 10.5),
    zoom: int = 6,
) -> folium.Map:
    """Create a base Folium map with satellite tile layer."""
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def add_classification_overlay(
    m: folium.Map,
    classification_mask: np.ndarray,
    bounds: tuple[float, float, float, float],
    class_names: list[str],
) -> folium.Map:
    """Overlay a colour-coded classification on the Folium map."""
    height, width = classification_mask.shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    for class_idx, class_name in enumerate(class_names):
        mask = classification_mask == class_idx
        color = CLASS_COLORS.get(class_name, (128, 128, 128))
        overlay[mask, :3] = color
        overlay[mask, 3] = 180

    img = Image.fromarray(overlay, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_b64}",
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.6,
        name="Classification",
    ).add_to(m)
    return m


def create_class_distribution_chart(
    probs: np.ndarray,
    class_names: list[str],
) -> go.Figure:
    """Create a horizontal bar chart of class probabilities."""
    df = pd.DataFrame({"Class": class_names, "Probability": probs * 100})
    df = df.sort_values("Probability", ascending=True)
    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        title="Classification Probabilities",
        labels={"Probability": "Probability (%)"},
    )
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------------------
# Sidebar — shared across all modes
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, list[Path]]:
    """Render sidebar controls and return (mode, available_images)."""
    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single Image Classification", "Change Detection", "Time Series"],
    )

    st.sidebar.header("Image Library")
    available = scan_available_images()

    if available:
        st.sidebar.caption(f"{len(available)} image(s) found in data/")
    else:
        st.sidebar.warning("No images in data/. Upload below or add files to data/sample/.")

    st.sidebar.subheader("Add to library")
    uploaded = st.sidebar.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="sidebar_upload",
    )
    if uploaded:
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SAMPLE_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        st.sidebar.success(f"Saved: {image_label(save_path)}")
        st.rerun()

    return mode, available


# ---------------------------------------------------------------------------
# Mode pages
# ---------------------------------------------------------------------------


def page_classification(images: list[Path], class_names: list[str]) -> None:
    """Single Image Classification page."""
    st.header("Single Image Classification")

    if not images:
        st.info("No images available. Upload images via the sidebar.")
        return

    labels = [image_label(p) for p in images]
    selected_label = st.selectbox("Select an image", labels)
    selected_path = images[labels.index(selected_label)]

    model = load_model()
    image_np, image_tensor = load_image_from_path(selected_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_np, caption=selected_label)

    with col2:
        st.subheader("Classification Result")
        with st.spinner("Classifying..."):
            pred_class, confidence, probs = classify_image(model, image_tensor)
        predicted_label = class_names[pred_class]
        st.success(f"**Predicted Class:** {predicted_label}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.subheader("Class Probabilities")
    fig = create_class_distribution_chart(probs, class_names)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Map View")
    m = create_folium_map()
    st_folium(m, width=None, height=400)

    st.subheader("Export Results")
    summary = {
        "class": predicted_label,
        "confidence": confidence,
        **{f"prob_{name}": prob for name, prob in zip(class_names, probs, strict=False)},
    }
    df = pd.DataFrame([summary])
    csv = df.to_csv(index=False)
    st.download_button("Download CSV Summary", csv, "classification_result.csv", "text/csv")


def page_change_detection(images: list[Path], class_names: list[str]) -> None:
    """Change Detection page."""
    st.header("Change Detection")
    st.markdown("Select two images of the same area at different times")

    if len(images) < 2:
        st.info("Need at least 2 images. Upload images via the sidebar.")
        return

    labels = [image_label(p) for p in images]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time 1 (Before)")
        sel1 = st.selectbox("Before image", labels, index=0, key="cd_before")
    with col2:
        st.subheader("Time 2 (After)")
        default_after = min(1, len(labels) - 1)
        sel2 = st.selectbox("After image", labels, index=default_after, key="cd_after")

    path1 = images[labels.index(sel1)]
    path2 = images[labels.index(sel2)]

    img1_np, img1_tensor = load_image_from_path(path1)
    img2_np, img2_tensor = load_image_from_path(path2)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img1_np, caption=f"Before: {sel1}")
    with c2:
        st.image(img2_np, caption=f"After: {sel2}")

    if st.button("Detect Changes"):
        with st.spinner("Analyzing changes..."):
            detector = ChangeDetector()
            result = detector.detect_changes(img1_tensor, img2_tensor)

        st.subheader("Change Detection Results")
        st.metric("Changed Area", f"{result.change_percentage:.1f}%")
        if result.change_type:
            st.info(f"Detected change type: **{result.change_type}**")
        st.image(result.change_mask * 255, caption="Change Mask (white = change)", clamp=True)


def page_time_series(images: list[Path], class_names: list[str]) -> None:
    """Time Series Analysis page."""
    st.header("Time Series Analysis")

    if len(images) < 2:
        st.info(
            "Upload multiple images from different time periods "
            "to track land use changes over time."
        )
        return

    labels = [image_label(p) for p in images]
    selected = st.multiselect("Select images (in chronological order)", labels, default=labels[:3])

    if selected:
        cols = st.columns(min(len(selected), 4))
        for i, label in enumerate(selected):
            path = images[labels.index(label)]
            img_np, _ = load_image_from_path(path)
            with cols[i % len(cols)]:
                st.image(img_np, caption=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit application entry point."""
    st.title("Satellite Image Analysis Platform")
    st.markdown("Land use classification and change detection from satellite imagery")

    mode, available_images = render_sidebar()
    class_names = config.get("data.classes")

    if mode == "Single Image Classification":
        page_classification(available_images, class_names)
    elif mode == "Change Detection":
        page_change_detection(available_images, class_names)
    elif mode == "Time Series":
        page_time_series(available_images, class_names)


if __name__ == "__main__":
    main()
