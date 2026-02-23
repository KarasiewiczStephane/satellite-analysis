"""Tests for EuroSAT dataset, preprocessing, and augmentation."""

import numpy as np
import pytest
from PIL import Image

from src.data.augmentation import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation90,
    RandomVerticalFlip,
    SpectralAugmentation,
    get_train_transforms,
    get_val_transforms,
)
from src.data.preprocessor import EuroSATDataset, create_data_splits, load_eurosat_paths
from src.utils.config import Config


@pytest.fixture()
def _mock_config(tmp_path):
    content = """\
data:
  classes:
    - Forest
    - Residential
    - River
  num_classes: 3
model:
  learning_rate: 0.001
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content)
    instance = Config()
    instance.reset()
    instance.load(cfg_path)


@pytest.fixture()
def sample_data_dir(tmp_path):
    """Create a fake image directory with JPEG images."""
    for cls_name in ("Forest", "Residential", "River"):
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir()
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(cls_dir / f"img_{i}.jpg")
    return tmp_path


@pytest.fixture()
def sample_image():
    return np.random.rand(3, 64, 64).astype(np.float32)


class TestEuroSATDataset:
    @pytest.mark.usefixtures("_mock_config")
    def test_dataset_length(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        dataset = EuroSATDataset(paths, labels)
        assert len(dataset) == 30

    @pytest.mark.usefixtures("_mock_config")
    def test_dataset_getitem_shape(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        dataset = EuroSATDataset(paths, labels)
        image, label = dataset[0]
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)

    @pytest.mark.usefixtures("_mock_config")
    def test_dataset_normalization(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        dataset = EuroSATDataset(paths, labels)
        image, _ = dataset[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    @pytest.mark.usefixtures("_mock_config")
    def test_dataset_with_transform(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        transform = get_train_transforms()
        dataset = EuroSATDataset(paths, labels, transform=transform)
        image, label = dataset[0]
        assert image.shape[0] == 3


class TestLoadEurosatPaths:
    @pytest.mark.usefixtures("_mock_config")
    def test_returns_correct_counts(self, sample_data_dir) -> None:
        paths, labels, classes = load_eurosat_paths(sample_data_dir)
        assert len(paths) == 30
        assert len(labels) == 30
        assert len(classes) == 3


class TestDataSplits:
    @pytest.mark.usefixtures("_mock_config")
    def test_split_preserves_total(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        splits = create_data_splits(paths, labels)
        total = sum(len(s[0]) for s in splits.values())
        assert total == len(paths)

    @pytest.mark.usefixtures("_mock_config")
    def test_custom_ratios(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        splits = create_data_splits(paths, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert len(splits["train"][0]) > 0
        assert len(splits["val"][0]) > 0
        assert len(splits["test"][0]) > 0

    @pytest.mark.usefixtures("_mock_config")
    def test_stratification(self, sample_data_dir) -> None:
        paths, labels, _ = load_eurosat_paths(sample_data_dir)
        splits = create_data_splits(paths, labels)
        for _, (_, split_labels) in splits.items():
            unique = set(split_labels)
            assert len(unique) == 3


class TestAugmentations:
    def test_horizontal_flip_deterministic(self, sample_image) -> None:
        transform = RandomHorizontalFlip(p=1.0)
        flipped = transform(sample_image)
        expected = np.flip(sample_image, axis=2)
        np.testing.assert_array_equal(flipped, expected)

    def test_vertical_flip_deterministic(self, sample_image) -> None:
        transform = RandomVerticalFlip(p=1.0)
        flipped = transform(sample_image)
        expected = np.flip(sample_image, axis=1)
        np.testing.assert_array_equal(flipped, expected)

    def test_rotation_shape(self, sample_image) -> None:
        transform = RandomRotation90()
        result = transform(sample_image)
        assert result.shape[0] == 3

    def test_random_crop(self, sample_image) -> None:
        transform = RandomCrop(size=32)
        result = transform(sample_image)
        assert result.shape == (3, 32, 32)

    def test_random_crop_skip_small(self) -> None:
        small = np.random.rand(3, 16, 16).astype(np.float32)
        transform = RandomCrop(size=32)
        result = transform(small)
        assert result.shape == (3, 16, 16)

    def test_spectral_augmentation_range(self, sample_image) -> None:
        transform = SpectralAugmentation()
        result = transform(sample_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_compose(self, sample_image) -> None:
        transform = Compose([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)])
        result = transform(sample_image)
        assert result.shape == sample_image.shape

    def test_get_val_transforms_identity(self, sample_image) -> None:
        transform = get_val_transforms()
        result = transform(sample_image)
        np.testing.assert_array_equal(result, sample_image)
