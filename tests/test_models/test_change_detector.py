"""Tests for the change detection pipeline."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.change_detector import (
    ChangeClassifier,
    ChangeDetectionResult,
    ChangeDetector,
    DifferenceModule,
    categorize_change,
    generate_synthetic_change_pairs,
)
from src.utils.config import Config


@pytest.fixture()
def _mock_config(tmp_path):
    content = """\
change_detection:
  sensitivity_threshold: 0.3
  min_change_area: 100
data:
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
def pair_images():
    img1 = np.random.rand(3, 32, 32).astype(np.float32)
    img2 = np.random.rand(3, 32, 32).astype(np.float32)
    return img1, img2


class TestDifferenceModule:
    def test_absolute(self) -> None:
        module = DifferenceModule(method="absolute")
        t1 = torch.randn(1, 3, 16, 16)
        t2 = torch.randn(1, 3, 16, 16)
        out = module(t1, t2)
        assert out.shape == (1, 3, 16, 16)
        assert (out >= 0).all()

    def test_squared(self) -> None:
        module = DifferenceModule(method="squared")
        t1 = torch.randn(1, 3, 16, 16)
        t2 = torch.randn(1, 3, 16, 16)
        out = module(t1, t2)
        assert out.shape == (1, 3, 16, 16)
        assert (out >= 0).all()

    def test_learned(self) -> None:
        module = DifferenceModule(method="learned")
        t1 = torch.randn(1, 3, 16, 16)
        t2 = torch.randn(1, 3, 16, 16)
        out = module(t1, t2)
        assert out.shape == (1, 16, 16, 16)

    def test_invalid_method(self) -> None:
        module = DifferenceModule(method="invalid")
        with pytest.raises(ValueError):
            module(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))


class TestChangeClassifier:
    def test_forward_shape(self) -> None:
        clf = ChangeClassifier(in_channels=3, hidden_dim=32)
        x = torch.randn(2, 3, 16, 16)
        out = clf(x)
        assert out.shape == (2, 1)


class TestCategorizeChange:
    def test_urbanization(self) -> None:
        assert categorize_change("Forest", "Residential") == "urbanization"

    def test_deforestation(self) -> None:
        assert categorize_change("Forest", "AnnualCrop") == "deforestation"

    def test_flooding(self) -> None:
        assert categorize_change("Residential", "River") == "flooding"

    def test_other(self) -> None:
        assert categorize_change("River", "Highway") == "other"


class TestChangeDetector:
    @pytest.mark.usefixtures("_mock_config")
    def test_compute_difference_map(self, pair_images) -> None:
        detector = ChangeDetector(device="cpu")
        diff = detector.compute_difference_map(*pair_images)
        assert diff.shape == (3, 32, 32)
        assert (diff >= 0).all()

    @pytest.mark.usefixtures("_mock_config")
    def test_generate_change_mask(self, pair_images) -> None:
        detector = ChangeDetector(device="cpu")
        diff = detector.compute_difference_map(*pair_images)
        mask = detector.generate_change_mask(diff)
        assert mask.shape == (32, 32)
        assert set(np.unique(mask)).issubset({0, 1})

    @pytest.mark.usefixtures("_mock_config")
    def test_detect_changes(self, pair_images) -> None:
        detector = ChangeDetector(device="cpu")
        result = detector.detect_changes(*pair_images)
        assert isinstance(result, ChangeDetectionResult)
        assert 0 <= result.change_percentage <= 100

    @pytest.mark.usefixtures("_mock_config")
    def test_detect_with_classes(self, pair_images) -> None:
        detector = ChangeDetector(device="cpu")
        result = detector.detect_changes(*pair_images, class1="Forest", class2="Residential")
        assert result.change_type == "urbanization"


class TestSyntheticPairs:
    @pytest.mark.usefixtures("_mock_config")
    def test_generate_pairs(self, tmp_path) -> None:
        for cls in ("Forest", "Residential", "River"):
            d = tmp_path / cls
            d.mkdir()
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                img.save(d / f"img_{i}.jpg")

        pairs = generate_synthetic_change_pairs(tmp_path, num_pairs=20)
        assert len(pairs) > 0
        assert all(len(p) == 4 for p in pairs)
