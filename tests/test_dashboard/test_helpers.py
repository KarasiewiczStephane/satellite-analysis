"""Tests for dashboard helper functions (non-Streamlit logic)."""

import io

import folium
import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image

from src.dashboard.app import (
    classify_image,
    create_class_distribution_chart,
    create_folium_map,
    preprocess_image,
)
from src.models.classifier import LandUseClassifier


@pytest.fixture()
def sample_upload():
    """Simulate a Streamlit uploaded file."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture()
def model():
    return LandUseClassifier(num_classes=3, architecture="resnet50", pretrained=False)


class TestPreprocessImage:
    def test_shapes(self, sample_upload) -> None:
        display, tensor = preprocess_image(sample_upload)
        assert display.shape == (64, 64, 3)
        assert tensor.shape == (3, 64, 64)
        assert tensor.max() <= 1.0


class TestClassifyImage:
    def test_returns_valid(self, model, sample_upload) -> None:
        _, tensor = preprocess_image(sample_upload)
        pred, conf, probs = classify_image(model, tensor)
        assert 0 <= pred < 3
        assert 0 <= conf <= 1
        assert probs.shape == (3,)


class TestCreateFoliumMap:
    def test_returns_map(self) -> None:
        m = create_folium_map()
        assert isinstance(m, folium.Map)


class TestClassDistributionChart:
    def test_returns_figure(self) -> None:
        probs = np.array([0.5, 0.3, 0.2])
        fig = create_class_distribution_chart(probs, ["A", "B", "C"])
        assert isinstance(fig, go.Figure)
