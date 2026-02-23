"""Shared pytest fixtures for the satellite analysis test suite."""

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from src.utils.config import Config


@pytest.fixture()
def sample_image():
    """Create a random 3-band image array (C, H, W)."""
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture()
def sample_classification_mask():
    """Create a random classification mask (H, W)."""
    return np.random.randint(0, 10, size=(64, 64), dtype=np.uint8)


@pytest.fixture()
def sample_data_dir(tmp_path):
    """Create a temporary EuroSAT-style directory with JPEG images."""
    classes = ["Forest", "Residential", "River"]
    for class_name in classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(class_dir / f"img_{i}.jpg")
    return tmp_path


@pytest.fixture()
def sample_geotiff(tmp_path):
    """Create a small 4-band GeoTIFF for testing."""
    path = tmp_path / "test_image.tif"
    data = np.random.randint(0, 10000, (4, 32, 32), dtype=np.uint16)
    transform = from_bounds(10.0, 48.0, 10.1, 48.1, 32, 32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=4,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture()
def mock_config(tmp_path):
    """Load a minimal mock config for testing."""
    content = """\
data:
  classes:
    - Forest
    - Residential
    - River
  num_classes: 3
model:
  architecture: resnet50
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 2
export:
  crs: "EPSG:4326"
change_detection:
  sensitivity_threshold: 0.3
  min_change_area: 100
logging:
  level: INFO
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)

    instance = Config()
    instance.reset()
    instance.load(config_path)
    return instance
