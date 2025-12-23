from __future__ import annotations

import torch
import torch.nn as nn


class ResNet10_3D(nn.Module):
    """
    MONAI 3D ResNet10 wrapper.

    Input:  (B, K, C, H, W) where C=1
    Output: (B, num_classes)
    """

    def __init__(self, *, num_classes: int = 6, in_channels: int = 1) -> None:
        super().__init__()
        from monai.networks.nets import resnet10

        self.net = resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, K, C, H, W) -> (B, C, K, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.net(x)

