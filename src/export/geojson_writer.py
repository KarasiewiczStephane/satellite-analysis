"""Export classification results as GeoJSON vector polygons."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from shapely.geometry import mapping, shape

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def classification_to_polygons(
    classification_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: CRS | str,
    class_names: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Convert a classification raster into vector polygons.

    Args:
        classification_mask: 2-D array of integer class indices.
        transform: Affine transform mapping pixel to CRS coordinates.
        crs: Coordinate reference system.
        class_names: Ordered class name list.

    Returns:
        GeoDataFrame with ``class`` and ``geometry`` columns.
    """
    class_names = class_names or config.get("data.classes")

    geometries: list[Any] = []
    classes: list[str] = []

    unique_classes = np.unique(classification_mask)

    for class_idx in unique_classes:
        mask = (classification_mask == class_idx).astype(np.uint8)
        shapes = features.shapes(mask, transform=transform)

        for geom, value in shapes:
            if value == 1:
                geometries.append(shape(geom))
                name = (
                    class_names[int(class_idx)]
                    if int(class_idx) < len(class_names)
                    else f"class_{class_idx}"
                )
                classes.append(name)

    gdf = gpd.GeoDataFrame(
        {"class": classes, "geometry": geometries},
        crs=crs,
    )

    if not gdf.empty:
        gdf = gdf.dissolve(by="class").reset_index()

    return gdf


def export_geojson(
    classification_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: CRS | str,
    output_path: str | Path,
    class_names: list[str] | None = None,
    simplify_tolerance: float | None = None,
) -> Path:
    """Export classification results as a GeoJSON file.

    Args:
        classification_mask: 2-D class index array.
        transform: Affine transform.
        crs: Coordinate reference system.
        output_path: Destination file path.
        class_names: Ordered class name list.
        simplify_tolerance: Optional geometry simplification tolerance.

    Returns:
        Path to the written GeoJSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = classification_to_polygons(classification_mask, transform, crs, class_names)

    if simplify_tolerance and not gdf.empty:
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance)

    gdf.to_file(output_path, driver="GeoJSON")
    logger.info("Exported GeoJSON to %s", output_path)

    return output_path


def create_geojson_feature_collection(
    classification_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection from a list of classification dicts.

    Each dict should contain ``geometry`` (Shapely), ``class``, and optional
    ``confidence``, ``area_m2``, ``timestamp`` keys.

    Args:
        classification_results: List of result dictionaries.

    Returns:
        GeoJSON FeatureCollection dictionary.
    """
    geojson_features = []

    for result in classification_results:
        feature = {
            "type": "Feature",
            "geometry": mapping(result["geometry"]),
            "properties": {
                "class": result["class"],
                "confidence": result.get("confidence"),
                "area_m2": result.get("area_m2"),
                "timestamp": result.get("timestamp"),
            },
        }
        geojson_features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": geojson_features,
    }
