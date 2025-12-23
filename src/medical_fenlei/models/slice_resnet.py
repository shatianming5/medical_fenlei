from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SliceMeanResNet(nn.Module):
    """
    Apply a 2D ResNet to each slice and mean-pool logits across slices.

    Input:  (B, K, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, *, num_classes: int = 6, in_channels: int = 1, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        if in_channels != 3:
            old = self.backbone.conv1
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained and old.weight.shape[1] == 3 and in_channels == 1:
                with torch.no_grad():
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            self.backbone.conv1 = new

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, k, c, h, w = x.shape
        x = x.view(b * k, c, h, w)
        logits = self.backbone(x)
        logits = logits.view(b, k, -1).mean(dim=1)
        return logits
