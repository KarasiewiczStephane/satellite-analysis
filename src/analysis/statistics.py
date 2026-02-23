"""Area calculation, class statistics, and temporal change tracking."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LandUseStatistics:
    """Container for land-use statistics of a classified area.

    Attributes:
        class_distribution: Percentage of each class.
        class_areas: Area in square metres for each class.
        total_area: Total classified area in square metres.
        timestamp: Acquisition timestamp (optional).
        metadata: Auxiliary information.
    """

    class_distribution: dict[str, float]
    class_areas: dict[str, float]
    total_area: float
    timestamp: datetime | None
    metadata: dict[str, Any]


def calculate_pixel_area(transform: Affine, crs: CRS | str) -> float:
    """Estimate the area of a single pixel in square metres.

    Args:
        transform: Affine geo-transform.
        crs: Coordinate reference system.

    Returns:
        Pixel area in m².
    """
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    crs_obj = CRS.from_string(str(crs)) if isinstance(crs, str) else crs

    if crs_obj and crs_obj.is_geographic:
        meters_per_degree = 111_320
        pixel_area = (pixel_width * meters_per_degree) * (pixel_height * meters_per_degree)
    else:
        pixel_area = pixel_width * pixel_height

    return pixel_area


def calculate_class_areas(
    classification_mask: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Calculate the area per class in square metres.

    Args:
        classification_mask: 2-D integer class index array.
        transform: Affine geo-transform.
        crs: Coordinate reference system.
        class_names: Ordered class name list.

    Returns:
        Mapping of class name to area in m².
    """
    class_names = class_names or config.get("data.classes")
    pixel_area = calculate_pixel_area(transform, crs)

    class_areas: dict[str, float] = {}
    unique_classes, counts = np.unique(classification_mask, return_counts=True)

    for class_idx, count in zip(unique_classes, counts, strict=False):
        if int(class_idx) < len(class_names):
            class_name = class_names[int(class_idx)]
            class_areas[class_name] = float(count) * pixel_area

    return class_areas


def calculate_statistics(
    classification_mask: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    class_names: list[str] | None = None,
    timestamp: datetime | None = None,
) -> LandUseStatistics:
    """Compute comprehensive land-use statistics from a classification mask.

    Args:
        classification_mask: 2-D integer class index array.
        transform: Affine geo-transform.
        crs: Coordinate reference system.
        class_names: Ordered class name list.
        timestamp: Optional acquisition timestamp.

    Returns:
        Populated :class:`LandUseStatistics`.
    """
    class_names = class_names or config.get("data.classes")

    class_areas = calculate_class_areas(classification_mask, transform, crs, class_names)
    total_area = sum(class_areas.values())

    class_distribution = {
        name: (area / total_area * 100) if total_area > 0 else 0
        for name, area in class_areas.items()
    }

    return LandUseStatistics(
        class_distribution=class_distribution,
        class_areas=class_areas,
        total_area=total_area,
        timestamp=timestamp,
        metadata={
            "pixel_count": int(classification_mask.size),
            "crs": str(crs),
        },
    )


def track_changes_over_time(
    statistics_series: list[LandUseStatistics],
) -> pd.DataFrame:
    """Build a DataFrame tracking class areas across time periods.

    Args:
        statistics_series: Chronologically ordered statistics.

    Returns:
        DataFrame with one row per time step and columns for each class.
    """
    records = []

    for stats in statistics_series:
        record: dict[str, Any] = {
            "timestamp": stats.timestamp,
            "total_area_m2": stats.total_area,
        }
        for name, pct in stats.class_distribution.items():
            record[f"{name}_pct"] = pct
        for name, area in stats.class_areas.items():
            record[f"{name}_m2"] = area
        records.append(record)

    df = pd.DataFrame(records)

    if len(df) > 1 and "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        for col in df.columns:
            if col.endswith("_pct") or col.endswith("_m2"):
                df[f"{col}_change"] = df[col].diff()

    return df


def generate_summary_report(
    statistics: LandUseStatistics,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a human-readable summary report dictionary.

    Args:
        statistics: Computed land-use statistics.
        output_path: Optional JSON file path to persist the report.

    Returns:
        Report dictionary.
    """
    report: dict[str, Any] = {
        "summary": {
            "total_area_km2": statistics.total_area / 1e6,
            "total_area_hectares": statistics.total_area / 1e4,
            "timestamp": statistics.timestamp.isoformat() if statistics.timestamp else None,
        },
        "class_breakdown": [
            {
                "class": name,
                "area_m2": statistics.class_areas.get(name, 0),
                "area_hectares": statistics.class_areas.get(name, 0) / 1e4,
                "percentage": statistics.class_distribution.get(name, 0),
            }
            for name in statistics.class_distribution
        ],
        "dominant_class": max(
            statistics.class_distribution,
            key=statistics.class_distribution.get,  # type: ignore[arg-type]
        ),
        "metadata": statistics.metadata,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Summary report saved to %s", output_path)

    return report


def export_statistics_csv(
    statistics_list: list[LandUseStatistics],
    output_path: str | Path,
) -> Path:
    """Export a list of statistics to a CSV file.

    Args:
        statistics_list: One or more statistics snapshots.
        output_path: Destination file path.

    Returns:
        Path to the written CSV.
    """
    df = track_changes_over_time(statistics_list)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Statistics exported to %s", output_path)
    return output_path
