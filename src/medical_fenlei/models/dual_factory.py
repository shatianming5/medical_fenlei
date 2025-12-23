from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class DualInputWrapper(nn.Module):
    """
    Wrap a single-input model to accept dual inputs.

    Input:  (B, 2, C, D, H, W)
    Output: (B, 2, num_classes)

    Notes:
      - If the underlying model returns a tuple/list, we take the first item.
      - If the underlying model returns a segmentation-like tensor
        (B, C, D, H, W), we global-average pool to (B, C).
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"expected x shape (B,2,C,D,H,W), got {tuple(x.shape)}")
        b, s, c, d, h, w = x.shape
        x2 = x.reshape(b * s, c, d, h, w).contiguous()

        y2 = self.base(x2)
        if isinstance(y2, (tuple, list)):
            y2 = y2[0]
        if not torch.is_tensor(y2):
            raise TypeError(f"model output must be a Tensor, got {type(y2)}")

        if y2.ndim == 5:
            y2 = y2.mean(dim=(2, 3, 4))
        if y2.ndim != 2:
            raise ValueError(f"expected base output (N,C) or (N,C,D,H,W), got {tuple(y2.shape)}")

        return y2.reshape(b, s, -1)


@dataclass(frozen=True)
class DualModelSpec:
    name: str
    kwargs: dict[str, Any]


_RESNET_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d$")


def make_dual_model(
    name: str,
    *,
    num_classes: int,
    in_channels: int,
    img_size: tuple[int, int, int],
    vit_patch_size: tuple[int, int, int] = (4, 16, 16),
    vit_hidden_size: int = 768,
    vit_mlp_dim: int = 3072,
    vit_num_layers: int = 12,
    vit_num_heads: int = 12,
    unet_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    unet_strides: tuple[int, ...] = (2, 2, 2, 2),
    unet_num_res_units: int = 2,
) -> tuple[nn.Module, DualModelSpec]:
    """
    Factory for dual-output models backed by MONAI networks.

    Supported:
      - dual_resnet{10,18,34,50,101,152,200}_3d
      - dual_unet_3d
      - dual_vit_3d
    """
    from monai.networks import nets

    m = _RESNET_RE.match(name)
    if m:
        depth = m.group("depth")
        fn_name = f"resnet{depth}"
        if not hasattr(nets, fn_name):
            raise ValueError(f"unsupported resnet depth: {depth}")
        fn = getattr(nets, fn_name)
        base = fn(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        return DualInputWrapper(base), DualModelSpec(name=name, kwargs={})

    if name == "dual_unet_3d":
        stride_prod = 1
        for s in unet_strides:
            stride_prod *= int(s)
        for dim in img_size:
            if int(dim) % stride_prod != 0:
                raise ValueError(
                    f"UNet requires each dim in img_size {img_size} divisible by prod(strides)={stride_prod} "
                    f"(strides={unet_strides})"
                )
        base = nets.UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=unet_channels,
            strides=unet_strides,
            num_res_units=int(unet_num_res_units),
        )
        return (
            DualInputWrapper(base),
            DualModelSpec(
                name=name,
                kwargs={
                    "unet_channels": tuple(int(x) for x in unet_channels),
                    "unet_strides": tuple(int(x) for x in unet_strides),
                    "unet_num_res_units": int(unet_num_res_units),
                },
            ),
        )

    if name == "dual_vit_3d":
        # ViT uses img_size/patch_size to build positional embeddings.
        if any(s <= 0 for s in img_size):
            raise ValueError(f"invalid img_size: {img_size}")
        if any(p <= 0 for p in vit_patch_size):
            raise ValueError(f"invalid vit_patch_size: {vit_patch_size}")
        for dim, p in zip(img_size, vit_patch_size):
            if int(dim) % int(p) != 0:
                raise ValueError(f"ViT requires img_size {img_size} divisible by patch_size {vit_patch_size}")
        base = nets.ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=vit_patch_size,
            hidden_size=int(vit_hidden_size),
            mlp_dim=int(vit_mlp_dim),
            num_layers=int(vit_num_layers),
            num_heads=int(vit_num_heads),
            classification=True,
            num_classes=num_classes,
            spatial_dims=3,
        )
        return (
            DualInputWrapper(base),
            DualModelSpec(
                name=name,
                kwargs={
                    "img_size": tuple(int(x) for x in img_size),
                    "vit_patch_size": tuple(int(x) for x in vit_patch_size),
                    "vit_hidden_size": int(vit_hidden_size),
                    "vit_mlp_dim": int(vit_mlp_dim),
                    "vit_num_layers": int(vit_num_layers),
                    "vit_num_heads": int(vit_num_heads),
                },
            ),
        )

    raise ValueError(f"unknown model: {name}")
