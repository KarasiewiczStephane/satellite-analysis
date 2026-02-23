"""Data preprocessing, dataset class, and stratified splitting for EuroSAT."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.geospatial import load_satellite_image
from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT satellite images.

    Supports both multi-band GeoTIFF and standard JPEG/PNG formats.

    Args:
        image_paths: List of image file paths.
        labels: Corresponding integer class labels.
        transform: Optional callable applied to the numpy array before
            conversion to a tensor.
        bands: 1-indexed band numbers to read from GeoTIFF files.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Callable | None = None,
        bands: list[int] | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.bands = bands or [1, 2, 3]
        self.classes = config.get("data.classes")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a ``(image_tensor, label)`` pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of image tensor ``(C, H, W)`` and integer label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        if image_path.suffix.lower() in (".tif", ".tiff"):
            sat_img = load_satellite_image(image_path, self.bands)
            image = sat_img.data
        else:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            image = np.array(img).transpose(2, 0, 1).astype(np.float32)

        image = self._normalize(image)

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), label

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Normalize each band to the ``[0, 1]`` range.

        Args:
            image: Array of shape ``(C, H, W)``.

        Returns:
            Normalized array with the same shape.
        """
        image = image.copy()
        for i in range(image.shape[0]):
            band = image[i]
            min_val, max_val = band.min(), band.max()
            if max_val > min_val:
                image[i] = (band - min_val) / (max_val - min_val)
            else:
                image[i] = np.zeros_like(band)
        return image


def load_eurosat_paths(
    data_dir: str | Path,
) -> tuple[list[Path], list[int], list[str]]:
    """Scan the EuroSAT directory for images and their class labels.

    Args:
        data_dir: Root directory containing one sub-folder per class.

    Returns:
        Tuple of ``(image_paths, labels, class_names)``.
    """
    data_dir = Path(data_dir)
    classes = config.get("data.classes")

    image_paths: list[Path] = []
    labels: list[int] = []

    for class_idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_dir)
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
                image_paths.append(img_path)
                labels.append(class_idx)

    logger.info("Loaded %d images from %d classes", len(image_paths), len(classes))
    return image_paths, labels, classes


def create_data_splits(
    image_paths: list[Path],
    labels: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, tuple[list[Path], list[int]]]:
    """Create stratified train / validation / test splits.

    Args:
        image_paths: All image file paths.
        labels: Corresponding integer labels.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        ``(paths, labels)`` tuples.
    """
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state,
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_state,
    )

    logger.info(
        "Split sizes - Train: %d, Val: %d, Test: %d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }
