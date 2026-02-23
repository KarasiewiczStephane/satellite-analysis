"""Tests for the land-use classification models."""

import pytest
import torch

from src.models.classifier import LandUseClassifier


class TestResNet50:
    def test_output_shape(self) -> None:
        model = LandUseClassifier(num_classes=10, architecture="resnet50", pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == (2, 10)

    def test_custom_input_channels(self) -> None:
        model = LandUseClassifier(
            num_classes=10, architecture="resnet50", in_channels=4, pretrained=False
        )
        x = torch.randn(2, 4, 64, 64)
        output = model(x)
        assert output.shape == (2, 10)

    def test_get_features(self) -> None:
        model = LandUseClassifier(num_classes=10, architecture="resnet50", pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        features = model.get_features(x)
        assert features.dim() == 2
        assert features.shape[0] == 2
        assert features.shape[1] == 2048


class TestEfficientNetB0:
    def test_output_shape(self) -> None:
        model = LandUseClassifier(num_classes=10, architecture="efficientnet_b0", pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == (2, 10)

    def test_custom_classes(self) -> None:
        model = LandUseClassifier(num_classes=5, architecture="efficientnet_b0", pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        output = model(x)
        assert output.shape == (1, 5)


class TestClassifierMisc:
    def test_invalid_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            LandUseClassifier(architecture="invalid_arch")

    def test_forward_differentiable(self) -> None:
        model = LandUseClassifier(num_classes=10, architecture="resnet50", pretrained=False)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
