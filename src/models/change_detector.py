"""Two-image change detection with difference maps and binary classification."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.classifier import LandUseClassifier
from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ChangeDetectionResult:
    """Container for change detection outputs.

    Attributes:
        change_mask: Binary mask ``(H, W)`` — 1 where change is detected.
        change_probability: Probability map ``(H, W)``.
        difference_map: Raw per-band differences ``(C, H, W)``.
        change_type: Semantic change category (e.g. ``"urbanization"``).
        change_percentage: Fraction of pixels classified as changed.
    """

    change_mask: np.ndarray
    change_probability: np.ndarray
    difference_map: np.ndarray
    change_type: str | None
    change_percentage: float


class DifferenceModule(nn.Module):
    """Compute difference features between two temporally separated images.

    Args:
        method: Differencing strategy (``"absolute"``, ``"squared"``, or
            ``"learned"``).
    """

    def __init__(self, method: str = "absolute") -> None:
        super().__init__()
        self.method = method

        if method == "learned":
            self.conv = nn.Sequential(
                nn.Conv2d(6, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute difference features.

        Args:
            img1: First image ``(B, C, H, W)``.
            img2: Second image ``(B, C, H, W)``.

        Returns:
            Difference feature tensor.
        """
        if self.method == "absolute":
            return torch.abs(img1 - img2)
        elif self.method == "squared":
            return (img1 - img2) ** 2
        elif self.method == "learned":
            concat = torch.cat([img1, img2], dim=1)
            return self.conv(concat)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ChangeClassifier(nn.Module):
    """Binary classifier that predicts change / no-change from a difference map.

    Args:
        in_channels: Number of input channels (matching the difference map).
        hidden_dim: Hidden feature dimension.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, diff_features: torch.Tensor) -> torch.Tensor:
        """Predict change probability logit.

        Args:
            diff_features: Difference feature tensor ``(B, C, H, W)``.

        Returns:
            Logit tensor of shape ``(B, 1)``.
        """
        features = self.encoder(diff_features)
        return self.classifier(features)


CHANGE_TYPES = [
    ("urbanization", ["Forest", "Pasture", "HerbaceousVegetation"], ["Residential", "Industrial"]),
    ("deforestation", ["Forest"], ["Pasture", "AnnualCrop", "HerbaceousVegetation"]),
    ("flooding", ["Residential", "Pasture", "Forest"], ["River", "SeaLake"]),
    (
        "agricultural_expansion",
        ["Forest", "HerbaceousVegetation"],
        ["AnnualCrop", "PermanentCrop"],
    ),
]


def categorize_change(class_from: str, class_to: str) -> str:
    """Determine the semantic change type between two land-use classes.

    Args:
        class_from: Land-use class at time 1.
        class_to: Land-use class at time 2.

    Returns:
        Change type string (e.g. ``"urbanization"``), or ``"other"``.
    """
    for change_type, from_classes, to_classes in CHANGE_TYPES:
        if class_from in from_classes and class_to in to_classes:
            return change_type
    return "other"


class ChangeDetector:
    """End-to-end change detection pipeline.

    Args:
        classifier: Optional trained land-use classifier.
        change_classifier: Binary change classifier network.
        sensitivity_threshold: Pixel difference threshold for change masking.
        device: Compute device.
    """

    def __init__(
        self,
        classifier: LandUseClassifier | None = None,
        change_classifier: ChangeClassifier | None = None,
        sensitivity_threshold: float | None = None,
        device: str = "auto",
    ) -> None:
        self.land_use_classifier = classifier
        self.change_classifier = change_classifier or ChangeClassifier()
        self.difference_module = DifferenceModule(method="absolute")

        self.sensitivity = sensitivity_threshold or config.get(
            "change_detection.sensitivity_threshold", 0.3
        )

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.change_classifier.to(self.device)
        if self.land_use_classifier:
            self.land_use_classifier.to(self.device)

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute pixel-wise absolute difference between two images.

        Args:
            img1: First image array ``(C, H, W)``.
            img2: Second image array ``(C, H, W)``.

        Returns:
            Difference array ``(C, H, W)``.
        """
        return np.abs(img1.astype(np.float32) - img2.astype(np.float32))

    def generate_change_mask(
        self,
        difference_map: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Threshold the difference map into a binary change mask.

        Args:
            difference_map: Difference array ``(C, H, W)``.
            threshold: Override for the default sensitivity threshold.

        Returns:
            Binary mask ``(H, W)`` with dtype ``uint8``.
        """
        threshold = threshold or self.sensitivity
        avg_diff = np.mean(difference_map, axis=0)
        return (avg_diff > threshold).astype(np.uint8)

    @torch.no_grad()
    def detect_changes(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        class1: str | None = None,
        class2: str | None = None,
    ) -> ChangeDetectionResult:
        """Run the full change detection pipeline on two images.

        Args:
            image1: Image at time 1 ``(C, H, W)``.
            image2: Image at time 2 ``(C, H, W)``.
            class1: Optional known land-use class at time 1.
            class2: Optional known land-use class at time 2.

        Returns:
            Populated :class:`ChangeDetectionResult`.
        """
        self.change_classifier.eval()

        diff_map = self.compute_difference_map(image1, image2)
        change_mask = self.generate_change_mask(diff_map)

        diff_tensor = torch.tensor(diff_map, dtype=torch.float32).unsqueeze(0).to(self.device)
        change_logit = self.change_classifier(diff_tensor)
        change_prob_scalar = torch.sigmoid(change_logit).item()

        change_prob_map = change_mask.astype(np.float32) * change_prob_scalar

        change_type = None
        if class1 and class2 and class1 != class2:
            change_type = categorize_change(class1, class2)

        change_percentage = float(np.mean(change_mask) * 100)

        return ChangeDetectionResult(
            change_mask=change_mask,
            change_probability=change_prob_map,
            difference_map=diff_map,
            change_type=change_type,
            change_percentage=change_percentage,
        )


def generate_synthetic_change_pairs(
    dataset_dir: Path,
    num_pairs: int = 1000,
    classes: list[str] | None = None,
) -> list[tuple[Path, Path, bool, str | None]]:
    """Generate synthetic image pairs for change detection training.

    Strategy: images from different classes simulate change; same-class pairs
    simulate no-change.

    Args:
        dataset_dir: Root directory with one sub-folder per class.
        num_pairs: Number of pairs to generate.
        classes: Class names. Defaults to the configured class list.

    Returns:
        List of ``(path1, path2, is_change, change_label)`` tuples.
    """
    import random

    classes = classes or config.get("data.classes")
    pairs: list[tuple[Path, Path, bool, str | None]] = []

    images_by_class: dict[str, list[Path]] = {}
    for class_name in classes:
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            images_by_class[class_name] = list(class_dir.glob("*"))

    available_classes = [c for c in images_by_class if len(images_by_class[c]) >= 2]

    for _ in range(num_pairs):
        if not available_classes:
            break
        if random.random() < 0.5:
            class_name = random.choice(available_classes)
            img1, img2 = random.sample(images_by_class[class_name], 2)
            pairs.append((img1, img2, False, None))
        else:
            if len(available_classes) < 2:
                continue
            class1, class2 = random.sample(available_classes, 2)
            img1 = random.choice(images_by_class[class1])
            img2 = random.choice(images_by_class[class2])
            pairs.append((img1, img2, True, f"{class1}_to_{class2}"))

    logger.info("Generated %d synthetic change pairs", len(pairs))
    return pairs
