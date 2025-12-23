from __future__ import annotations

import torch
import torch.nn as nn


class DualResNet10_3D(nn.Module):
    """
    Dual-output classifier with a shared MONAI 3D ResNet10 backbone.

    Input:  (B, 2, C, D, H, W) where 2=(left,right)
    Output: (B, 2, num_classes)
    """

    def __init__(self, *, num_classes: int = 6, in_channels: int = 1) -> None:
        super().__init__()
        from monai.networks.nets import resnet10

        self.net = resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"expected x shape (B,2,C,D,H,W), got {tuple(x.shape)}")
        b, s, c, d, h, w = x.shape
        x2 = x.reshape(b * s, c, d, h, w).contiguous()
        y2 = self.net(x2)  # (B*2, num_classes)
        y = y2.reshape(b, s, -1)
        return y

