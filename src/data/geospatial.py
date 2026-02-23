"""Satellite image loading with rasterio and geospatial metadata handling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SatelliteImage:
    """Container for a satellite image with geospatial metadata.

    Attributes:
        data: Pixel values with shape ``(bands, height, width)``.
        crs: Coordinate reference system.
        transform: Affine transform mapping pixel to world coordinates.
        bounds: Geographic bounding box ``(left, bottom, right, top)``.
        metadata: Additional rasterio metadata.
        path: Source file path.
    """

    data: np.ndarray
    crs: CRS | None
    transform: Affine | None
    bounds: tuple[float, float, float, float] | None
    metadata: dict[str, Any]
    path: Path

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.data.shape[2]

    @property
    def num_bands(self) -> int:
        """Number of spectral bands."""
        return self.data.shape[0]


def load_satellite_image(
    path: str | Path,
    bands: list[int] | None = None,
) -> SatelliteImage:
    """Load a satellite image using rasterio with full geospatial metadata.

    Args:
        path: Path to a GeoTIFF or other rasterio-supported file.
        bands: 1-indexed band numbers to read. ``None`` reads all bands.

    Returns:
        Populated :class:`SatelliteImage` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be opened.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with rasterio.open(path) as src:
        if bands is None:
            data = src.read()
        else:
            data = src.read(bands)

        return SatelliteImage(
            data=data.astype(np.float32),
            crs=src.crs,
            transform=src.transform,
            bounds=src.bounds,
            metadata={
                "driver": src.driver,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "count": src.count,
                "descriptions": src.descriptions,
            },
            path=path,
        )


def extract_geospatial_metadata(image: SatelliteImage) -> dict[str, Any]:
    """Extract a summary dictionary of geospatial metadata.

    Args:
        image: Loaded satellite image.

    Returns:
        Dictionary containing CRS, transform, bounds, and shape information.
    """
    return {
        "crs": str(image.crs) if image.crs else None,
        "transform": list(image.transform)[:6] if image.transform else None,
        "bounds": {
            "left": image.bounds[0],
            "bottom": image.bounds[1],
            "right": image.bounds[2],
            "top": image.bounds[3],
        }
        if image.bounds
        else None,
        "shape": {
            "bands": image.num_bands,
            "height": image.height,
            "width": image.width,
        },
        **image.metadata,
    }
