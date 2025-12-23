from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


BackboneName = Literal["resnet18", "resnet34", "resnet50"]


@dataclass(frozen=True)
class SliceAttentionSpec:
    backbone: str
    embed_dim: int
    attn_hidden: int
    dropout: float
    out_dim: int


def _make_resnet(backbone: BackboneName, *, in_channels: int) -> tuple[nn.Module, int]:
    if backbone == "resnet18":
        net = resnet18(weights=None)
        embed_dim = 512
    elif backbone == "resnet34":
        net = resnet34(weights=None)
        embed_dim = 512
    elif backbone == "resnet50":
        net = resnet50(weights=None)
        embed_dim = 2048
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    if int(in_channels) != 3:
        old = net.conv1
        net.conv1 = nn.Conv2d(
            int(in_channels),
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
    net.fc = nn.Identity()
    return net, int(embed_dim)


class SliceAttentionResNet(nn.Module):
    """
    2D slice encoder + attention pooling along z.

    Input:  (B, K, C, H, W)
    Output: logits (B, out_dim)
    """

    def __init__(
        self,
        *,
        backbone: BackboneName = "resnet18",
        in_channels: int = 1,
        attn_hidden: int = 128,
        dropout: float = 0.1,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.backbone, embed_dim = _make_resnet(backbone, in_channels=int(in_channels))
        self.embed_dim = int(embed_dim)
        self.attn = nn.Sequential(
            nn.Linear(self.embed_dim, int(attn_hidden)),
            nn.Tanh(),
            nn.Linear(int(attn_hidden), 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.embed_dim, int(out_dim)),
        )
        self.spec = SliceAttentionSpec(
            backbone=str(backbone),
            embed_dim=int(self.embed_dim),
            attn_hidden=int(attn_hidden),
            dropout=float(dropout),
            out_dim=int(out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
        return_embedding: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"expected x (B,K,C,H,W), got shape={tuple(x.shape)}")
        b, k, c, h, w = x.shape
        z = x.reshape(b * k, c, h, w)
        emb = self.backbone(z)  # (B*K, D)
        emb = emb.reshape(b, k, self.embed_dim)
        scores = self.attn(emb).squeeze(-1)  # (B,K)
        wts = torch.softmax(scores, dim=1)
        pooled = (wts.unsqueeze(-1) * emb).sum(dim=1)
        logits = self.head(pooled)

        if not return_attention and not return_embedding:
            return logits

        out: dict[str, torch.Tensor] = {"logits": logits}
        if return_attention:
            out["attention"] = wts
        if return_embedding:
            out["embedding"] = pooled
        return out

