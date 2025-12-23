from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


BackboneName = Literal["resnet18", "resnet34", "resnet50"]
AggregatorName = Literal["attention", "mean", "transformer"]


@dataclass(frozen=True)
class SliceAttentionSpec:
    backbone: str
    embed_dim: int
    aggregator: str
    attn_hidden: int
    dropout: float
    out_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
    transformer_max_len: int


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
    2D slice encoder + pooling along z (mean/attention/transformer).

    Input:  (B, K, C, H, W)
    Output: logits (B, out_dim)
    """

    def __init__(
        self,
        *,
        backbone: BackboneName = "resnet18",
        in_channels: int = 1,
        aggregator: AggregatorName = "attention",
        attn_hidden: int = 128,
        dropout: float = 0.1,
        out_dim: int = 1,
        transformer_layers: int = 0,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 0,
        transformer_dropout: float = 0.1,
        transformer_max_len: int = 256,
    ) -> None:
        super().__init__()
        self.backbone, embed_dim = _make_resnet(backbone, in_channels=int(in_channels))
        self.embed_dim = int(embed_dim)
        self.aggregator = str(aggregator)

        if self.aggregator not in ("attention", "mean", "transformer"):
            raise ValueError(f"unknown aggregator: {aggregator}")

        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ff_dim = int(transformer_ff_dim) if int(transformer_ff_dim) > 0 else int(self.embed_dim) * 4
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_max_len = int(transformer_max_len)

        if self.aggregator == "transformer":
            if self.transformer_layers <= 0:
                raise ValueError("transformer_layers must be >0 when aggregator='transformer'")
            if self.embed_dim % max(1, self.transformer_heads) != 0:
                raise ValueError(f"embed_dim {self.embed_dim} must be divisible by nhead {self.transformer_heads}")
            if self.transformer_max_len <= 0:
                raise ValueError("transformer_max_len must be >0")

            self.pos_emb = nn.Embedding(int(self.transformer_max_len), int(self.embed_dim))
            enc = nn.TransformerEncoderLayer(
                d_model=int(self.embed_dim),
                nhead=int(self.transformer_heads),
                dim_feedforward=int(self.transformer_ff_dim),
                dropout=float(self.transformer_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc, num_layers=int(self.transformer_layers))
        else:
            self.pos_emb = None
            self.transformer = None

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
            aggregator=str(self.aggregator),
            attn_hidden=int(attn_hidden),
            dropout=float(dropout),
            out_dim=int(out_dim),
            transformer_layers=int(self.transformer_layers),
            transformer_heads=int(self.transformer_heads),
            transformer_ff_dim=int(self.transformer_ff_dim),
            transformer_dropout=float(self.transformer_dropout),
            transformer_max_len=int(self.transformer_max_len),
        )

    @classmethod
    def from_spec(cls, spec: dict[str, Any], *, in_channels: int = 1) -> "SliceAttentionResNet":
        return cls(
            backbone=str(spec.get("backbone", "resnet18")),
            in_channels=int(in_channels),
            aggregator=str(spec.get("aggregator", "attention")),
            attn_hidden=int(spec.get("attn_hidden", 128)),
            dropout=float(spec.get("dropout", 0.2)),
            out_dim=int(spec.get("out_dim", 1)),
            transformer_layers=int(spec.get("transformer_layers", 0) or 0),
            transformer_heads=int(spec.get("transformer_heads", 8) or 8),
            transformer_ff_dim=int(spec.get("transformer_ff_dim", 0) or 0),
            transformer_dropout=float(spec.get("transformer_dropout", 0.1) or 0.1),
            transformer_max_len=int(spec.get("transformer_max_len", 256) or 256),
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

        if self.transformer is not None and self.pos_emb is not None:
            if k > int(self.transformer_max_len):
                raise ValueError(f"num_slices={k} exceeds transformer_max_len={self.transformer_max_len}")
            pos = torch.arange(k, device=emb.device, dtype=torch.long)
            emb = emb + self.pos_emb(pos)[None, :, :].to(dtype=emb.dtype)
            emb = self.transformer(emb)

        if self.aggregator == "mean":
            pooled = emb.mean(dim=1)
            wts = torch.full((b, k), 1.0 / float(max(1, k)), device=emb.device, dtype=emb.dtype)
        else:
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
