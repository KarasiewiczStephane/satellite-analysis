"""Tests for area calculation and statistical summaries."""

import json
from datetime import datetime

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.analysis.statistics import (
    LandUseStatistics,
    calculate_class_areas,
    calculate_pixel_area,
    calculate_statistics,
    export_statistics_csv,
    generate_summary_report,
    track_changes_over_time,
)
from src.utils.config import Config

CLASSES = ["Forest", "Residential", "River"]


@pytest.fixture()
def _mock_config(tmp_path):
    content = """\
data:
  classes:
    - Forest
    - Residential
    - River
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
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 0
    mask[50:, :50] = 1
    mask[50:, 50:] = 2
    return mask


@pytest.fixture()
def transform_10m():
    """10-meter pixel projected CRS."""
    return Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)


class TestCalculatePixelArea:
    def test_projected_crs(self, transform_10m) -> None:
        area = calculate_pixel_area(transform_10m, CRS.from_epsg(32632))
        assert area == pytest.approx(100.0)

    def test_geographic_crs(self) -> None:
        transform = Affine(0.001, 0.0, 0.0, 0.0, -0.001, 0.0)
        area = calculate_pixel_area(transform, "EPSG:4326")
        assert area > 0


class TestCalculateClassAreas:
    @pytest.mark.usefixtures("_mock_config")
    def test_areas_sum_to_total(self, classification_mask, transform_10m) -> None:
        areas = calculate_class_areas(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES
        )
        total = sum(areas.values())
        expected = classification_mask.size * 100.0
        assert total == pytest.approx(expected)

    @pytest.mark.usefixtures("_mock_config")
    def test_correct_class_names(self, classification_mask, transform_10m) -> None:
        areas = calculate_class_areas(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES
        )
        assert set(areas.keys()) == set(CLASSES)


class TestCalculateStatistics:
    @pytest.mark.usefixtures("_mock_config")
    def test_distribution_sums_to_100(self, classification_mask, transform_10m) -> None:
        stats = calculate_statistics(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES
        )
        total_pct = sum(stats.class_distribution.values())
        assert total_pct == pytest.approx(100.0)

    @pytest.mark.usefixtures("_mock_config")
    def test_returns_dataclass(self, classification_mask, transform_10m) -> None:
        stats = calculate_statistics(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES
        )
        assert isinstance(stats, LandUseStatistics)
        assert stats.total_area > 0

    @pytest.mark.usefixtures("_mock_config")
    def test_with_timestamp(self, classification_mask, transform_10m) -> None:
        ts = datetime(2024, 6, 15)
        stats = calculate_statistics(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES, timestamp=ts
        )
        assert stats.timestamp == ts


class TestTrackChangesOverTime:
    def test_single_entry(self) -> None:
        stats = LandUseStatistics(
            class_distribution={"Forest": 50, "River": 50},
            class_areas={"Forest": 500, "River": 500},
            total_area=1000,
            timestamp=datetime(2024, 1, 1),
            metadata={},
        )
        df = track_changes_over_time([stats])
        assert len(df) == 1

    def test_change_rates(self) -> None:
        s1 = LandUseStatistics(
            class_distribution={"Forest": 60, "River": 40},
            class_areas={"Forest": 600, "River": 400},
            total_area=1000,
            timestamp=datetime(2024, 1, 1),
            metadata={},
        )
        s2 = LandUseStatistics(
            class_distribution={"Forest": 50, "River": 50},
            class_areas={"Forest": 500, "River": 500},
            total_area=1000,
            timestamp=datetime(2024, 6, 1),
            metadata={},
        )
        df = track_changes_over_time([s1, s2])
        assert "Forest_pct_change" in df.columns


class TestGenerateSummaryReport:
    @pytest.mark.usefixtures("_mock_config")
    def test_report_structure(self, classification_mask, transform_10m) -> None:
        stats = calculate_statistics(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES
        )
        report = generate_summary_report(stats)
        assert "summary" in report
        assert "class_breakdown" in report
        assert "dominant_class" in report

    @pytest.mark.usefixtures("_mock_config")
    def test_save_to_file(self, classification_mask, transform_10m, tmp_path) -> None:
        stats = calculate_statistics(
            classification_mask, transform_10m, CRS.from_epsg(32632), CLASSES
        )
        path = tmp_path / "report.json"
        generate_summary_report(stats, output_path=path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["dominant_class"] == "Forest"


class TestExportStatisticsCSV:
    def test_creates_csv(self, tmp_path) -> None:
        stats = LandUseStatistics(
            class_distribution={"Forest": 100},
            class_areas={"Forest": 1000},
            total_area=1000,
            timestamp=datetime(2024, 1, 1),
            metadata={},
        )
        path = tmp_path / "stats.csv"
        export_statistics_csv([stats], path)
        assert path.exists()
