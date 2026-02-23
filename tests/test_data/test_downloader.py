"""Tests for the EuroSAT downloader and sample dataset creator."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.data.downloader import DownloadProgressBar, create_sample_dataset
from src.utils.config import Config


@pytest.fixture()
def _mock_config(tmp_path):
    """Load a mock config for testing."""
    content = """\
data:
  eurosat_url: "https://example.com/EuroSAT.zip"
  raw_dir: "data/raw"
  classes:
    - Forest
    - Residential
    - River
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content)
    instance = Config()
    instance.reset()
    instance.load(cfg_path)


@pytest.fixture()
def sample_source(tmp_path):
    """Create a fake EuroSAT directory with tiny GeoTIFF images."""
    source = tmp_path / "eurosat"
    for cls in ("Forest", "Residential", "River"):
        cls_dir = source / cls
        cls_dir.mkdir(parents=True)
        for i in range(5):
            path = cls_dir / f"{cls}_{i}.tif"
            data = np.random.randint(0, 255, (3, 8, 8), dtype=np.uint8)
            transform = from_bounds(0, 0, 1, 1, 8, 8)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=3,
                dtype="uint8",
                transform=transform,
                crs="EPSG:4326",
            ) as dst:
                dst.write(data)
    return source


class TestDownloadProgressBar:
    def test_update_to_sets_total(self) -> None:
        bar = DownloadProgressBar(unit="B", unit_scale=True, disable=True)
        bar.update_to(b=1, bsize=100, tsize=1000)
        assert bar.total == 1000


class TestCreateSampleDataset:
    @pytest.mark.usefixtures("_mock_config")
    def test_creates_sample_dirs(self, sample_source, tmp_path) -> None:
        sample_dir = tmp_path / "sample"
        create_sample_dataset(sample_source, sample_dir, samples_per_class=2)

        for cls in ("Forest", "Residential", "River"):
            cls_dir = sample_dir / cls
            assert cls_dir.exists()
            assert len(list(cls_dir.glob("*.tif"))) == 2

    @pytest.mark.usefixtures("_mock_config")
    def test_handles_missing_class(self, sample_source, tmp_path) -> None:
        sample_dir = tmp_path / "sample"
        # Remove one class
        import shutil

        shutil.rmtree(sample_source / "River")
        create_sample_dataset(sample_source, sample_dir, samples_per_class=2)
        assert (
            not (sample_dir / "River").exists() or len(list((sample_dir / "River").glob("*"))) == 0
        )
