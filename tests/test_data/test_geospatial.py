"""Tests for satellite image loading and geospatial metadata extraction."""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src.data.geospatial import SatelliteImage, extract_geospatial_metadata, load_satellite_image


@pytest.fixture()
def sample_geotiff(tmp_path):
    """Create a small GeoTIFF file for testing."""
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


class TestSatelliteImage:
    def test_properties(self) -> None:
        data = np.zeros((3, 64, 128), dtype=np.float32)
        img = SatelliteImage(
            data=data,
            crs=None,
            transform=None,
            bounds=None,
            metadata={},
            path="dummy.tif",
        )
        assert img.height == 64
        assert img.width == 128
        assert img.num_bands == 3


class TestLoadSatelliteImage:
    def test_loads_all_bands(self, sample_geotiff) -> None:
        img = load_satellite_image(sample_geotiff)
        assert img.num_bands == 4
        assert img.height == 32
        assert img.width == 32
        assert img.data.dtype == np.float32

    def test_loads_specific_bands(self, sample_geotiff) -> None:
        img = load_satellite_image(sample_geotiff, bands=[1, 3])
        assert img.num_bands == 2

    def test_crs_and_transform(self, sample_geotiff) -> None:
        img = load_satellite_image(sample_geotiff)
        assert img.crs is not None
        assert img.crs == CRS.from_epsg(4326)
        assert img.transform is not None

    def test_bounds(self, sample_geotiff) -> None:
        img = load_satellite_image(sample_geotiff)
        assert img.bounds is not None
        left, bottom, right, top = img.bounds
        assert left < right
        assert bottom < top

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_satellite_image("/nonexistent/image.tif")


class TestExtractGeospatialMetadata:
    def test_metadata_keys(self, sample_geotiff) -> None:
        img = load_satellite_image(sample_geotiff)
        meta = extract_geospatial_metadata(img)

        assert "crs" in meta
        assert "transform" in meta
        assert "bounds" in meta
        assert "shape" in meta
        assert meta["shape"]["bands"] == 4
        assert meta["shape"]["height"] == 32
        assert meta["shape"]["width"] == 32

    def test_metadata_with_none_crs(self) -> None:
        data = np.zeros((1, 10, 10), dtype=np.float32)
        img = SatelliteImage(
            data=data,
            crs=None,
            transform=None,
            bounds=None,
            metadata={
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": None,
                "count": 1,
                "descriptions": None,
            },
            path="dummy.tif",
        )
        meta = extract_geospatial_metadata(img)
        assert meta["crs"] is None
        assert meta["transform"] is None
        assert meta["bounds"] is None
