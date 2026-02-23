"""Export classification and change detection results as GeoTIFF rasters."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "AnnualCrop": (255, 255, 0),
    "Forest": (0, 128, 0),
    "HerbaceousVegetation": (144, 238, 144),
    "Highway": (128, 128, 128),
    "Industrial": (255, 0, 0),
    "Pasture": (173, 255, 47),
    "PermanentCrop": (255, 165, 0),
    "Residential": (165, 42, 42),
    "River": (0, 0, 255),
    "SeaLake": (0, 191, 255),
}


def export_classification_geotiff(
    classification_mask: np.ndarray,
    output_path: str | Path,
    transform: Affine | None = None,
    crs: CRS | str | None = None,
    class_names: list[str] | None = None,
    add_color_overlay: bool = True,
) -> Path:
    """Write a classification mask as a colour-coded GeoTIFF.

    Args:
        classification_mask: 2-D integer class index array.
        output_path: Destination file path.
        transform: Affine transform. Defaults to identity.
        crs: Coordinate reference system. Defaults to config value.
        class_names: Ordered class name list.
        add_color_overlay: Write a 3-band RGB overlay instead of raw indices.

    Returns:
        Path to the written GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crs = crs or config.get("export.crs", "EPSG:4326")
    class_names = class_names or config.get("data.classes")
    height, width = classification_mask.shape

    if transform is None:
        transform = Affine.identity()

    if add_color_overlay:
        rgb = np.zeros((3, height, width), dtype=np.uint8)
        for class_idx, class_name in enumerate(class_names):
            mask = classification_mask == class_idx
            color = CLASS_COLORS.get(class_name, (128, 128, 128))
            for band_idx, c in enumerate(color):
                rgb[band_idx][mask] = c

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(rgb)
    else:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(classification_mask.astype(np.uint8), 1)

    logger.info("Exported GeoTIFF to %s", output_path)
    return output_path


def export_change_detection_geotiff(
    change_mask: np.ndarray,
    output_path: str | Path,
    transform: Affine | None = None,
    crs: CRS | str | None = None,
    probability_map: np.ndarray | None = None,
) -> Path:
    """Write change detection results as a red/green GeoTIFF.

    Args:
        change_mask: Binary change mask ``(H, W)``.
        output_path: Destination file path.
        transform: Affine transform.
        crs: Coordinate reference system.
        probability_map: Optional probability map to modulate intensity.

    Returns:
        Path to the written GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crs = crs or config.get("export.crs", "EPSG:4326")
    height, width = change_mask.shape

    if transform is None:
        transform = Affine.identity()

    rgb = np.zeros((3, height, width), dtype=np.uint8)
    rgb[0][change_mask == 1] = 255
    rgb[1][change_mask == 0] = 255

    if probability_map is not None:
        intensity = (probability_map * 255).astype(np.uint8)
        rgb[0] = np.where(change_mask == 1, intensity, 0)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(rgb)

    logger.info("Exported change detection GeoTIFF to %s", output_path)
    return output_path
