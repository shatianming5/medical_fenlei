from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn


AggregatorName = Literal["attention", "mean", "transformer"]


@dataclass(frozen=True)
class SliceAttentionUNetSpec:
    image_size: int
    unet_channels: tuple[int, ...]
    unet_strides: tuple[int, ...]
    unet_num_res_units: int
    unet_embed_dim: int
    aggregator: str
    attn_hidden: int
    dropout: float
    out_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
    transformer_max_len: int


class SliceAttentionUNet(nn.Module):
    """
    Per-slice 2D UNet encoder (MONAI) + pooling along z (mean/attention/transformer).

    Input:  (B, K, C, H, W)
    Output: logits (B, out_dim)
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 224,
        unet_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        unet_strides: tuple[int, ...] = (2, 2, 2, 2),
        unet_num_res_units: int = 2,
        unet_embed_dim: int = 128,
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
        from monai.networks import nets

        image_size = int(image_size)
        stride_prod = 1
        for s in unet_strides:
            stride_prod *= int(s)
        if image_size % stride_prod != 0:
            raise ValueError(f"UNet requires image_size={image_size} divisible by prod(strides)={stride_prod} (strides={unet_strides})")

        self.unet = nets.UNet(
            spatial_dims=2,
            in_channels=int(in_channels),
            out_channels=int(unet_embed_dim),
            channels=tuple(int(x) for x in unet_channels),
            strides=tuple(int(x) for x in unet_strides),
            num_res_units=int(unet_num_res_units),
        )

        self.embed_dim = int(unet_embed_dim)
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
        self.spec = SliceAttentionUNetSpec(
            image_size=int(image_size),
            unet_channels=tuple(int(x) for x in unet_channels),
            unet_strides=tuple(int(x) for x in unet_strides),
            unet_num_res_units=int(unet_num_res_units),
            unet_embed_dim=int(unet_embed_dim),
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
    def from_spec(cls, spec: dict[str, Any], *, in_channels: int = 1) -> "SliceAttentionUNet":
        def _parse_ints(v: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
            if v is None:
                return default
            if isinstance(v, (list, tuple)):
                return tuple(int(x) for x in v)
            if isinstance(v, str):
                parts = [p.strip() for p in v.split(",") if p.strip()]
                if parts:
                    return tuple(int(p) for p in parts)
            return default

        return cls(
            in_channels=int(in_channels),
            image_size=int(spec.get("image_size", 224)),
            unet_channels=_parse_ints(spec.get("unet_channels"), default=(16, 32, 64, 128, 256)),
            unet_strides=_parse_ints(spec.get("unet_strides"), default=(2, 2, 2, 2)),
            unet_num_res_units=int(spec.get("unet_num_res_units", 2)),
            unet_embed_dim=int(spec.get("unet_embed_dim", 128)),
            aggregator=str(spec.get("aggregator", "attention")),
            attn_hidden=int(spec.get("attn_hidden", 128)),
            dropout=float(spec.get("dropout", 0.1)),
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
        z = x.reshape(b * k, c, h, w).contiguous()
        y = self.unet(z)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.ndim != 4:
            raise ValueError(f"expected UNet output (N,C,H,W), got shape={tuple(y.shape)}")
        emb2 = y.mean(dim=(2, 3))  # (N,C)
        emb = emb2.reshape(b, k, self.embed_dim)

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
            scores = self.attn(emb).squeeze(-1)
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

