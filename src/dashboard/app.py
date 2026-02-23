"""Streamlit dashboard for satellite image analysis with Folium map integration."""

import base64
import io

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from streamlit_folium import st_folium

from src.export.geotiff_writer import CLASS_COLORS
from src.models.change_detector import ChangeDetector
from src.models.classifier import LandUseClassifier
from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(
    page_title="Satellite Image Analysis",
    page_icon="\U0001f6f0\ufe0f",
    layout="wide",
)


@st.cache_resource
def load_model() -> LandUseClassifier:
    """Load and cache the classification model."""
    from pathlib import Path

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


def preprocess_image(uploaded_file: io.BytesIO) -> tuple[np.ndarray, np.ndarray]:
    """Convert an uploaded file to display and tensor arrays.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Tuple of ``(display_array_HWC, tensor_array_CHW)``.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_tensor = image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image_np, image_tensor


def classify_image(
    model: LandUseClassifier,
    image_tensor: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Run classification on a preprocessed image tensor.

    Args:
        model: Loaded classifier.
        image_tensor: Array of shape ``(C, H, W)`` in ``[0, 1]``.

    Returns:
        Tuple of ``(predicted_class_index, confidence, all_probabilities)``.
    """
    with torch.no_grad():
        tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    probs_np = np.array(probs[0].tolist(), dtype=np.float32)
    return pred_class, confidence, probs_np


def create_folium_map(
    center: tuple[float, float] = (51.5, 10.5),
    zoom: int = 6,
) -> folium.Map:
    """Create a base Folium map with satellite tile layer.

    Args:
        center: Map centre ``(lat, lon)``.
        zoom: Initial zoom level.

    Returns:
        Folium :class:`Map`.
    """
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
    """Overlay a colour-coded classification on the Folium map.

    Args:
        m: Folium map to modify.
        classification_mask: 2-D integer class indices.
        bounds: ``(west, south, east, north)`` geographic bounds.
        class_names: Ordered class name list.

    Returns:
        The modified Folium map.
    """
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
    """Create a horizontal bar chart of class probabilities.

    Args:
        probs: Probability array of length ``num_classes``.
        class_names: Ordered class name list.

    Returns:
        Plotly :class:`Figure`.
    """
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


def main() -> None:
    """Streamlit application entry point."""
    st.title("Satellite Image Analysis Platform")
    st.markdown("Land use classification and change detection from satellite imagery")

    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single Image Classification", "Change Detection", "Time Series"],
    )

    model = load_model()
    class_names = config.get("data.classes")

    if mode == "Single Image Classification":
        st.header("Upload Satellite Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
        )

        if uploaded_file:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                image_np, image_tensor = preprocess_image(uploaded_file)
                st.image(image_np, caption="Uploaded Image")

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
            st.download_button(
                "Download CSV Summary",
                csv,
                "classification_result.csv",
                "text/csv",
            )

    elif mode == "Change Detection":
        st.header("Change Detection")
        st.markdown("Upload two images of the same area at different times")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Time 1 (Before)")
            image1 = st.file_uploader("First image", type=["tif", "jpg", "png"], key="img1")
        with col2:
            st.subheader("Time 2 (After)")
            image2 = st.file_uploader("Second image", type=["tif", "jpg", "png"], key="img2")

        if image1 and image2:
            img1_np, img1_tensor = preprocess_image(image1)
            img2_np, img2_tensor = preprocess_image(image2)

            c1, c2 = st.columns(2)
            with c1:
                st.image(img1_np, caption="Before")
            with c2:
                st.image(img2_np, caption="After")

            if st.button("Detect Changes"):
                with st.spinner("Analyzing changes..."):
                    detector = ChangeDetector()
                    result = detector.detect_changes(img1_tensor, img2_tensor)

                st.subheader("Change Detection Results")
                st.metric("Changed Area", f"{result.change_percentage:.1f}%")
                if result.change_type:
                    st.info(f"Detected change type: **{result.change_type}**")
                st.image(
                    result.change_mask * 255, caption="Change Mask (white = change)", clamp=True
                )

    elif mode == "Time Series":
        st.header("Time Series Analysis")
        st.info(
            "Upload multiple images from different time periods "
            "to track land use changes over time."
        )


if __name__ == "__main__":
    main()
