"""CNN classifiers for satellite land-use classification.

Supports ResNet-50 and EfficientNet-B0 with optional pretrained weights and
configurable input channels (e.g. 4-band RGB+NIR).
"""

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class LandUseClassifier(nn.Module):
    """CNN backbone with a classification head for land-use prediction.

    Args:
        num_classes: Number of output classes.
        architecture: Backbone identifier (``"resnet50"`` or ``"efficientnet_b0"``).
        pretrained: Whether to load ImageNet pretrained weights.
        in_channels: Number of input spectral bands.
    """

    def __init__(
        self,
        num_classes: int = 10,
        architecture: Literal["resnet50", "efficientnet_b0"] = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)

            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, num_classes),
            )

        elif architecture == "efficientnet_b0":
            if TIMM_AVAILABLE:
                self.backbone = timm.create_model(
                    "efficientnet_b0",
                    pretrained=pretrained,
                    num_classes=num_classes,
                    in_chans=in_channels,
                )
            else:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.backbone = models.efficientnet_b0(weights=weights)

                if in_channels != 3:
                    self.backbone.features[0][0] = nn.Conv2d(
                        in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
                    )

                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, num_classes),
                )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the backbone.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vectors before the classification head.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor of shape ``(B, feature_dim)``.
        """
        if self.architecture == "resnet50":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)
        else:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)
