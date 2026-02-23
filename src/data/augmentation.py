"""Data augmentation transforms for satellite imagery.

All transforms operate on numpy arrays with shape ``(C, H, W)`` and values
in ``[0, 1]``.
"""

import random
from collections.abc import Callable

import numpy as np


class Compose:
    """Chain multiple transforms sequentially.

    Args:
        transforms: List of callable transforms.
    """

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip:
    """Flip the image horizontally with probability *p*.

    Args:
        p: Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.flip(image, axis=2).copy()
        return image


class RandomVerticalFlip:
    """Flip the image vertically with probability *p*.

    Args:
        p: Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.flip(image, axis=1).copy()
        return image


class RandomRotation90:
    """Rotate image by a random multiple of 90 degrees."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        k = random.randint(0, 3)
        return np.rot90(image, k, axes=(1, 2)).copy()


class RandomCrop:
    """Randomly crop the image to the given size.

    Args:
        size: Target spatial dimension (square crop).
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        _, h, w = image.shape
        if h <= self.size or w <= self.size:
            return image

        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        return image[:, top : top + self.size, left : left + self.size].copy()


class SpectralAugmentation:
    """Apply per-band brightness and contrast jitter.

    Args:
        brightness_range: Maximum absolute brightness shift.
        contrast_range: Maximum absolute contrast shift.
    """

    def __init__(
        self,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.copy()
        for i in range(image.shape[0]):
            brightness = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            contrast = 1 + random.uniform(-self.contrast_range, self.contrast_range)

            mean = image[i].mean()
            image[i] = (image[i] - mean) * contrast + mean * brightness
            image[i] = np.clip(image[i], 0, 1)
        return image


def get_train_transforms() -> Compose:
    """Return the default training augmentation pipeline."""
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(),
            SpectralAugmentation(brightness_range=0.1, contrast_range=0.1),
        ]
    )


def get_val_transforms() -> Compose:
    """Return the validation/test transform pipeline (identity)."""
    return Compose([])
