"""Tests for GeoJSON and GeoTIFF export writers."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Point

from src.export.geojson_writer import (
    classification_to_polygons,
    create_geojson_feature_collection,
    export_geojson,
)
from src.export.geotiff_writer import (
    export_change_detection_geotiff,
    export_classification_geotiff,
)
from src.utils.config import Config


@pytest.fixture()
def _mock_config(tmp_path):
    content = """\
data:
  classes:
    - Forest
    - Residential
    - River
  num_classes: 3
export:
  crs: "EPSG:4326"
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content)
    instance = Config()
    instance.reset()
    instance.load(cfg_path)


@pytest.fixture()
def classification_mask():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :] = 0  # Forest
    mask[16:, :16] = 1  # Residential
    mask[16:, 16:] = 2  # River
    return mask


class TestGeoJSONExport:
    @pytest.mark.usefixtures("_mock_config")
    def test_classification_to_polygons(self, classification_mask) -> None:
        transform = Affine.identity()
        gdf = classification_to_polygons(
            classification_mask, transform, "EPSG:4326", ["Forest", "Residential", "River"]
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    @pytest.mark.usefixtures("_mock_config")
    def test_export_creates_file(self, classification_mask, tmp_path) -> None:
        output = tmp_path / "test.geojson"
        transform = Affine.identity()
        export_geojson(
            classification_mask,
            transform,
            "EPSG:4326",
            output,
            class_names=["Forest", "Residential", "River"],
        )
        assert output.exists()

    @pytest.mark.usefixtures("_mock_config")
    def test_export_valid_geojson(self, classification_mask, tmp_path) -> None:
        output = tmp_path / "test.geojson"
        transform = Affine.identity()
        export_geojson(
            classification_mask,
            transform,
            "EPSG:4326",
            output,
            class_names=["Forest", "Residential", "River"],
        )
        with open(output) as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"

    def test_create_feature_collection(self) -> None:
        results = [
            {"geometry": Point(0, 0).buffer(1), "class": "Forest", "confidence": 0.9},
            {"geometry": Point(1, 1).buffer(1), "class": "River"},
        ]
        fc = create_geojson_feature_collection(results)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2


class TestGeoTIFFExport:
    @pytest.mark.usefixtures("_mock_config")
    def test_export_rgb_creates_file(self, classification_mask, tmp_path) -> None:
        output = tmp_path / "test.tif"
        export_classification_geotiff(
            classification_mask,
            output,
            class_names=["Forest", "Residential", "River"],
        )
        assert output.exists()

        with rasterio.open(output) as src:
            assert src.count == 3
            assert src.height == 32
            assert src.width == 32

    @pytest.mark.usefixtures("_mock_config")
    def test_export_single_band(self, classification_mask, tmp_path) -> None:
        output = tmp_path / "test_single.tif"
        export_classification_geotiff(
            classification_mask,
            output,
            class_names=["Forest", "Residential", "River"],
            add_color_overlay=False,
        )
        with rasterio.open(output) as src:
            assert src.count == 1

    @pytest.mark.usefixtures("_mock_config")
    def test_change_detection_geotiff(self, tmp_path) -> None:
        mask = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
        output = tmp_path / "change.tif"
        export_change_detection_geotiff(mask, output)
        assert output.exists()

        with rasterio.open(output) as src:
            assert src.count == 3

    @pytest.mark.usefixtures("_mock_config")
    def test_change_with_probability(self, tmp_path) -> None:
        mask = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
        prob = np.random.rand(32, 32).astype(np.float32)
        output = tmp_path / "change_prob.tif"
        export_change_detection_geotiff(mask, output, probability_map=prob)
        assert output.exists()
