from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn


AggregatorName = Literal["attention", "mean", "transformer"]


@dataclass(frozen=True)
class SliceAttentionViTSpec:
    # Per-slice ViT encoder
    image_size: int
    patch_size: int
    vit_hidden_size: int
    vit_mlp_dim: int
    vit_num_layers: int
    vit_num_heads: int
    vit_dropout: float
    # Slice pooling
    aggregator: str
    attn_hidden: int
    dropout: float
    out_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
    transformer_max_len: int


class _ViT2DEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        image_size = int(image_size)
        patch_size = int(patch_size)
        if image_size <= 0:
            raise ValueError(f"invalid image_size: {image_size}")
        if patch_size <= 0:
            raise ValueError(f"invalid patch_size: {patch_size}")
        if image_size % patch_size != 0:
            raise ValueError(f"ViT requires image_size={image_size} divisible by patch_size={patch_size}")

        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError(f"invalid hidden_size: {hidden_size}")
        num_heads = int(num_heads)
        if num_heads <= 0:
            raise ValueError(f"invalid num_heads: {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")

        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError(f"invalid num_layers: {num_layers}")

        mlp_dim = int(mlp_dim)
        if mlp_dim <= 0:
            mlp_dim = hidden_size * 4

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embed = nn.Conv2d(int(in_channels), hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        n = (image_size // patch_size) * (image_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + n, hidden_size))
        self.pos_drop = nn.Dropout(float(dropout))

        enc = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        self._init_params()

    def _init_params(self) -> None:
        # Conservative init; avoids extra deps.
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        else:
            nn.init.normal_(self.pos_emb, std=0.02)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected x (N,C,H,W), got shape={tuple(x.shape)}")
        n, _, h, w = x.shape
        if int(h) != int(self.image_size) or int(w) != int(self.image_size):
            raise ValueError(f"expected H=W={self.image_size}, got {(int(h), int(w))}")

        x = self.patch_embed(x)  # (N,D,Hp,Wp)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (N, P, D)
        cls = self.cls_token.expand(int(n), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_emb.to(dtype=x.dtype, device=x.device))
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]


class SliceAttentionViT(nn.Module):
    """
    Per-slice ViT encoder + pooling along z (mean/attention/transformer).

    Input:  (B, K, C, H, W)
    Output: logits (B, out_dim)
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 224,
        patch_size: int = 16,
        vit_hidden_size: int = 512,
        vit_mlp_dim: int = 2048,
        vit_num_layers: int = 8,
        vit_num_heads: int = 8,
        vit_dropout: float = 0.1,
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
        self.encoder = _ViT2DEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(patch_size),
            hidden_size=int(vit_hidden_size),
            mlp_dim=int(vit_mlp_dim),
            num_layers=int(vit_num_layers),
            num_heads=int(vit_num_heads),
            dropout=float(vit_dropout),
        )
        self.embed_dim = int(self.encoder.hidden_size)
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
        self.spec = SliceAttentionViTSpec(
            image_size=int(image_size),
            patch_size=int(patch_size),
            vit_hidden_size=int(vit_hidden_size),
            vit_mlp_dim=int(vit_mlp_dim),
            vit_num_layers=int(vit_num_layers),
            vit_num_heads=int(vit_num_heads),
            vit_dropout=float(vit_dropout),
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
    def from_spec(cls, spec: dict[str, Any], *, in_channels: int = 1) -> "SliceAttentionViT":
        return cls(
            in_channels=int(in_channels),
            image_size=int(spec.get("image_size", 224)),
            patch_size=int(spec.get("patch_size", 16)),
            vit_hidden_size=int(spec.get("vit_hidden_size", 512)),
            vit_mlp_dim=int(spec.get("vit_mlp_dim", 2048)),
            vit_num_layers=int(spec.get("vit_num_layers", 8)),
            vit_num_heads=int(spec.get("vit_num_heads", 8)),
            vit_dropout=float(spec.get("vit_dropout", 0.1)),
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
        emb = self.encoder(z).reshape(b, k, self.embed_dim)

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

