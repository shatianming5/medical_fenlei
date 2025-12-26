from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _choose_num_heads(embed_dim: int, *, prefer: int = 8) -> int:
    if int(embed_dim) <= 0:
        return 1
    for h in [int(prefer), 12, 10, 9, 8, 6, 5, 4, 3, 2, 1]:
        if h > 0 and int(embed_dim) % int(h) == 0:
            return int(h)
    return 1


class _TwoTokenSelfAttentionBlock(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(int(dim), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(int(dim))
        hidden = int(max(1, round(float(dim) * float(mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Linear(int(dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,2,dim)
        x1 = self.norm1(x)
        y, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class DualCrossEarAttention(nn.Module):
    """
    Shared backbone -> 2 ear tokens -> lightweight self-attention -> per-ear logits.

    Input:  (B,2,C,D,H,W)
    Output: (B,2,num_classes)
    """

    def __init__(
        self,
        extractor: nn.Module,
        *,
        feat_dim: int,
        num_classes: int,
        num_layers: int = 1,
        num_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.feat_dim = int(feat_dim)
        if num_heads is None:
            num_heads = _choose_num_heads(int(feat_dim), prefer=8)
        self.blocks = nn.ModuleList(
            [_TwoTokenSelfAttentionBlock(dim=int(feat_dim), num_heads=int(num_heads), dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        self.head = nn.Linear(int(feat_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"expected x shape (B,2,C,D,H,W), got {tuple(x.shape)}")
        b, s, c, d, h, w = x.shape
        x2 = x.reshape(b * s, c, d, h, w).contiguous()
        feat = self.extractor(x2)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat):
            raise TypeError(f"extractor output must be a Tensor, got {type(feat)}")
        if feat.ndim != 2:
            raise ValueError(f"expected extractor output (N,F), got {tuple(feat.shape)}")
        t = feat.reshape(b, s, -1)
        for blk in self.blocks:
            t = blk(t)
        return self.head(t)


class _ResNetStage4(nn.Module):
    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.resnet
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.act(x)
        if not r.no_max_pool:
            x = r.maxpool(x)
        x = r.layer1(x)
        x = r.layer2(x)
        x = r.layer3(x)
        x = r.layer4(x)
        return x


class DualResNetA3B(nn.Module):
    """
    A lightweight Asymmetry-Aware Attention Block (A3B) inspired by check.md:
      - extract per-ear feature maps
      - compute diff map |L-R|
      - depthwise 1x1x1 conv -> sigmoid mask
      - enhance features and classify per ear

    Input:  (B,2,C,D,H,W)
    Output: (B,2,num_classes)
    """

    def __init__(self, trunk: nn.Module, *, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.feat_dim = int(feat_dim)
        # depthwise 1x1x1 conv as a lightweight per-channel gate
        self.diff_conv = nn.Conv3d(int(feat_dim), int(feat_dim), kernel_size=1, groups=int(feat_dim), bias=True)
        self.head = nn.Linear(int(feat_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"expected x shape (B,2,C,D,H,W), got {tuple(x.shape)}")
        b, s, c, d, h, w = x.shape
        if s != 2:
            raise ValueError(f"expected 2 ears, got s={s}")

        x2 = x.reshape(b * s, c, d, h, w).contiguous()
        feat = self.trunk(x2)
        if feat.ndim != 5:
            raise ValueError(f"expected trunk output (N,C,D,H,W), got {tuple(feat.shape)}")
        feat = feat.reshape(b, s, feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4])
        left = feat[:, 0]
        right = feat[:, 1]

        diff = (left - right).abs()
        mask = torch.sigmoid(self.diff_conv(diff))
        left = left * (1.0 + mask)
        right = right * (1.0 + mask)

        left_vec = left.mean(dim=(2, 3, 4))
        right_vec = right.mean(dim=(2, 3, 4))
        out_left = self.head(left_vec)
        out_right = self.head(right_vec)
        return torch.stack([out_left, out_right], dim=1)


class DualResNetA3B4(nn.Module):
    """
    Multi-stage A3B (Difference Excitation) for ResNet backbones.

    Insert A3B after each ResNet stage (layer1..layer4), matching check.md's
    \"insert between every stage\" spirit.

    Input:  (B,2,C,D,H,W)
    Output: (B,2,num_classes)
    """

    def __init__(self, backbone: nn.Module, *, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        # Use per-stage lazy 1x1x1 convs to generate a single-channel asymmetry mask.
        # (LazyConv3d infers in_channels at first forward, so it works for both BasicBlock/Bottleneck variants.)
        self.a3b = nn.ModuleList([nn.LazyConv3d(1, kernel_size=1, bias=True) for _ in range(4)])
        self.head = nn.Linear(int(feat_dim), int(num_classes))

    def _apply_a3b(self, x: torch.Tensor, gate: nn.Module) -> torch.Tensor:
        # x: (B*2,C,D,H,W) where the first half is left and second half is right (after reshape).
        if x.ndim != 5:
            raise ValueError(f"expected (N,C,D,H,W), got {tuple(x.shape)}")
        n, c, d, h, w = x.shape
        if n % 2 != 0:
            raise ValueError(f"expected even batch for dual ears, got N={n}")
        b = n // 2
        x2 = x.reshape(b, 2, c, d, h, w)
        left = x2[:, 0]
        right = x2[:, 1]
        diff = (left - right).abs()
        mask = torch.sigmoid(gate(diff))  # (B,1,D,H,W)
        left = left * (1.0 + mask)
        right = right * (1.0 + mask)
        return torch.stack([left, right], dim=1).reshape(n, c, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"expected x shape (B,2,C,D,H,W), got {tuple(x.shape)}")
        b, s, c, d, h, w = x.shape
        if s != 2:
            raise ValueError(f"expected 2 ears, got s={s}")
        r = self.backbone

        x2 = x.reshape(b * s, c, d, h, w).contiguous()
        x2 = r.conv1(x2)
        x2 = r.bn1(x2)
        x2 = r.act(x2)
        if not r.no_max_pool:
            x2 = r.maxpool(x2)

        x2 = r.layer1(x2)
        x2 = self._apply_a3b(x2, self.a3b[0])
        x2 = r.layer2(x2)
        x2 = self._apply_a3b(x2, self.a3b[1])
        x2 = r.layer3(x2)
        x2 = self._apply_a3b(x2, self.a3b[2])
        x2 = r.layer4(x2)
        x2 = self._apply_a3b(x2, self.a3b[3])

        x2 = r.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if x2.shape[-1] != int(self.feat_dim):
            raise ValueError(f"feat_dim mismatch: expected {self.feat_dim}, got {int(x2.shape[-1])}")
        y2 = self.head(x2).reshape(b, s, -1)
        return y2


def _prompt_hash_prototypes(*, class_ids: list[int], dim: int, prompts_zh: dict[int, str] | None = None) -> torch.Tensor:
    """
    Build fixed-dim text prototypes from Chinese prompts using HashingVectorizer.

    This is a lightweight baseline placeholder for check.md's CMBERT prototypes:
    it produces deterministic vectors without extra model dependencies.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    from medical_fenlei.text_prompts import get_default_class_prompts_zh

    prompts = prompts_zh or get_default_class_prompts_zh()
    texts = [str(prompts.get(int(cid), "")) for cid in class_ids]
    vec = HashingVectorizer(
        n_features=int(dim),
        analyzer="char",
        ngram_range=(1, 2),
        lowercase=False,
        alternate_sign=False,
        norm=None,
    )
    mat = vec.transform(texts).toarray().astype("float32")
    t = torch.from_numpy(mat)
    return F.normalize(t, dim=-1)


def _prompt_hf_prototypes(
    *,
    class_ids: list[int],
    dim: int,
    model_name_or_path: str,
    pool: str,
    max_length: int,
    proj_seed: int,
    prompts_zh: dict[int, str] | None = None,
) -> torch.Tensor:
    """
    Build fixed-dim text prototypes from Chinese prompts using a HuggingFace model.

    This is closer to check.md's CMBERT requirement. The HF model output is
    projected to `dim` using a deterministic random projection.
    """
    from medical_fenlei.text_encoder import encode_texts_to_dim
    from medical_fenlei.text_prompts import get_default_class_prompts_zh

    prompts = prompts_zh or get_default_class_prompts_zh()
    texts = [str(prompts.get(int(cid), "")) for cid in class_ids]
    return encode_texts_to_dim(
        texts,
        dim=int(dim),
        encoder="hf",
        hf_model_name_or_path=str(model_name_or_path),
        hf_pool=str(pool),
        hf_max_length=int(max_length),
        proj_seed=int(proj_seed),
    )


class DualProtoCosineClassifier(nn.Module):
    """
    Prototype cosine classifier head for Dual 3D.

    extractor: single-input backbone that returns (N, feat_dim) when its fc/head is removed.
    """

    def __init__(
        self,
        extractor: nn.Module,
        *,
        feat_dim: int,
        num_classes: int,
        init: str = "prompt_hash",
        scale: float = 10.0,
        hf_model_name_or_path: str = "hfl/chinese-roberta-wwm-ext",
        hf_pool: str = "cls",
        hf_max_length: int = 256,
        hf_proj_seed: int = 42,
        prompts_zh: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self.extractor = DualInputWrapper(extractor)  # (B,2,feat_dim)
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))
        self.text_encoder = str(init).strip().lower()
        self.hf_model_name_or_path = str(hf_model_name_or_path)
        self.hf_pool = str(hf_pool)
        self.hf_max_length = int(hf_max_length)
        self.hf_proj_seed = int(hf_proj_seed)

        init_s = str(init).strip().lower()
        if init_s not in {"prompt_hash", "prompt_hf", "rand"}:
            raise ValueError(f"unknown proto init: {init!r} (expected prompt_hash|prompt_hf|rand)")
        if init_s == "prompt_hash":
            class_ids = list(range(int(num_classes)))
            p0 = _prompt_hash_prototypes(class_ids=class_ids, dim=int(feat_dim), prompts_zh=prompts_zh)
        elif init_s == "prompt_hf":
            class_ids = list(range(int(num_classes)))
            p0 = _prompt_hf_prototypes(
                class_ids=class_ids,
                dim=int(feat_dim),
                model_name_or_path=str(hf_model_name_or_path),
                pool=str(hf_pool),
                max_length=int(hf_max_length),
                proj_seed=int(hf_proj_seed),
                prompts_zh=prompts_zh,
            )
        else:
            g = torch.Generator()
            g.manual_seed(42)
            p0 = torch.randn(int(num_classes), int(feat_dim), generator=g)
            p0 = F.normalize(p0, dim=-1)

        self.prototypes = nn.Parameter(p0)
        # Anchor prototypes for check.md 4.3.2: constrain learned prototypes to stay
        # close to the text-derived initialization, and allow freezing missing tail
        # classes (their prototypes stay fixed as text prototypes).
        self.register_buffer("prototypes_init", p0.detach().clone(), persistent=True)
        self.register_buffer("frozen_mask", torch.zeros((int(num_classes),), dtype=torch.bool), persistent=True)

    def set_frozen_class_ids(self, class_ids: list[int] | tuple[int, ...] | set[int]) -> None:
        mask = torch.zeros_like(self.frozen_mask, dtype=torch.bool)
        for cid in class_ids:
            i = int(cid)
            if i < 0 or i >= int(self.num_classes):
                raise ValueError(f"frozen class id out of range: {i} (num_classes={self.num_classes})")
            mask[i] = True
        self.frozen_mask.copy_(mask)

    def get_frozen_class_ids(self) -> list[int]:
        return [int(i) for i, v in enumerate(self.frozen_mask.detach().cpu().tolist()) if bool(v)]

    def prototype_reg_loss(self, reduction: str = "mean") -> torch.Tensor:
        """
        L2 anchor regularization to keep prototypes close to initialization.

        By default, frozen classes are excluded (their prototypes are fixed).
        """
        proto = F.normalize(self.prototypes, dim=-1)
        init = self.prototypes_init

        if proto.shape != init.shape:
            raise ValueError(f"prototype/init shape mismatch: {tuple(proto.shape)} vs {tuple(init.shape)}")

        if self.frozen_mask.any():
            keep = ~self.frozen_mask
            if int(keep.sum().item()) <= 0:
                return torch.zeros((), device=proto.device, dtype=proto.dtype)
            proto = proto[keep]
            init = init[keep]

        per_class = (proto - init).pow(2).sum(dim=-1)
        reduction = str(reduction or "mean").strip().lower()
        if reduction in {"mean", "avg"}:
            return per_class.mean()
        if reduction in {"sum"}:
            return per_class.sum()
        raise ValueError(f"unknown reduction: {reduction!r} (expected mean|sum)")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)  # (B,2,feat_dim)

    def logits_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"expected feat (B,2,D), got {tuple(feat.shape)}")
        proto_raw = self.prototypes
        if self.frozen_mask.any():
            proto_raw = torch.where(self.frozen_mask[:, None], self.prototypes_init, proto_raw)
        feat = F.normalize(feat, dim=-1)
        proto = F.normalize(proto_raw, dim=-1)
        scale = self.logit_scale.clamp(min=1.0, max=100.0)
        logits = torch.einsum("bsd,cd->bsc", feat, proto) * scale
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.forward_features(x)
        logits = self.logits_from_features(feat)
        return logits, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits


class DualResNetA3B4ProtoCosineClassifier(nn.Module):
    """
    ResNet + multi-stage A3B (difference excitation) + prototype cosine head.

    This is the closest in-code approximation to check.md's "insert A3B between
    every stage + prototype classifier + vision-language contrastive pretraining".

    Input:  (B,2,C,D,H,W)
    Output: (B,2,num_classes)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feat_dim: int,
        num_classes: int,
        init: str = "prompt_hash",
        scale: float = 10.0,
        hf_model_name_or_path: str = "hfl/chinese-roberta-wwm-ext",
        hf_pool: str = "cls",
        hf_max_length: int = 256,
        hf_proj_seed: int = 42,
        prompts_zh: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

        # per-stage A3B gates (single-channel mask from |L-R|)
        self.a3b = nn.ModuleList([nn.LazyConv3d(1, kernel_size=1, bias=True) for _ in range(4)])

        init_s = str(init).strip().lower()
        if init_s not in {"prompt_hash", "prompt_hf", "rand"}:
            raise ValueError(f"unknown proto init: {init!r} (expected prompt_hash|prompt_hf|rand)")
        if init_s == "prompt_hash":
            class_ids = list(range(int(num_classes)))
            p0 = _prompt_hash_prototypes(class_ids=class_ids, dim=int(feat_dim), prompts_zh=prompts_zh)
        elif init_s == "prompt_hf":
            class_ids = list(range(int(num_classes)))
            p0 = _prompt_hf_prototypes(
                class_ids=class_ids,
                dim=int(feat_dim),
                model_name_or_path=str(hf_model_name_or_path),
                pool=str(hf_pool),
                max_length=int(hf_max_length),
                proj_seed=int(hf_proj_seed),
                prompts_zh=prompts_zh,
            )
        else:
            g = torch.Generator()
            g.manual_seed(42)
            p0 = torch.randn(int(num_classes), int(feat_dim), generator=g)
            p0 = F.normalize(p0, dim=-1)

        self.prototypes = nn.Parameter(p0)
        self.register_buffer("prototypes_init", p0.detach().clone(), persistent=True)
        self.register_buffer("frozen_mask", torch.zeros((int(num_classes),), dtype=torch.bool), persistent=True)

    def set_frozen_class_ids(self, class_ids: list[int] | tuple[int, ...] | set[int]) -> None:
        mask = torch.zeros_like(self.frozen_mask, dtype=torch.bool)
        for cid in class_ids:
            i = int(cid)
            if i < 0 or i >= int(self.num_classes):
                raise ValueError(f"frozen class id out of range: {i} (num_classes={self.num_classes})")
            mask[i] = True
        self.frozen_mask.copy_(mask)

    def get_frozen_class_ids(self) -> list[int]:
        return [int(i) for i, v in enumerate(self.frozen_mask.detach().cpu().tolist()) if bool(v)]

    def prototype_reg_loss(self, reduction: str = "mean") -> torch.Tensor:
        proto = F.normalize(self.prototypes, dim=-1)
        init = self.prototypes_init
        if proto.shape != init.shape:
            raise ValueError(f"prototype/init shape mismatch: {tuple(proto.shape)} vs {tuple(init.shape)}")

        if self.frozen_mask.any():
            keep = ~self.frozen_mask
            if int(keep.sum().item()) <= 0:
                return torch.zeros((), device=proto.device, dtype=proto.dtype)
            proto = proto[keep]
            init = init[keep]

        per_class = (proto - init).pow(2).sum(dim=-1)
        reduction = str(reduction or "mean").strip().lower()
        if reduction in {"mean", "avg"}:
            return per_class.mean()
        if reduction in {"sum"}:
            return per_class.sum()
        raise ValueError(f"unknown reduction: {reduction!r} (expected mean|sum)")

    def _apply_a3b(self, x: torch.Tensor, gate: nn.Module) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected (N,C,D,H,W), got {tuple(x.shape)}")
        n, c, d, h, w = x.shape
        if n % 2 != 0:
            raise ValueError(f"expected even batch for dual ears, got N={n}")
        b = n // 2
        x2 = x.reshape(b, 2, c, d, h, w)
        left = x2[:, 0]
        right = x2[:, 1]
        diff = (left - right).abs()
        mask = torch.sigmoid(gate(diff))  # (B,1,D,H,W)
        left = left * (1.0 + mask)
        right = right * (1.0 + mask)
        return torch.stack([left, right], dim=1).reshape(n, c, d, h, w)

    def _forward_backbone(self, x: torch.Tensor, *, return_map: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 6:
            raise ValueError(f"expected x shape (B,2,C,D,H,W), got {tuple(x.shape)}")
        b, s, c, d, h, w = x.shape
        if s != 2:
            raise ValueError(f"expected 2 ears, got s={s}")
        r = self.backbone

        x2 = x.reshape(b * s, c, d, h, w).contiguous()
        x2 = r.conv1(x2)
        x2 = r.bn1(x2)
        x2 = r.act(x2)
        if not r.no_max_pool:
            x2 = r.maxpool(x2)

        x2 = r.layer1(x2)
        x2 = self._apply_a3b(x2, self.a3b[0])
        x2 = r.layer2(x2)
        x2 = self._apply_a3b(x2, self.a3b[1])
        x2 = r.layer3(x2)
        x2 = self._apply_a3b(x2, self.a3b[2])
        x2 = r.layer4(x2)
        x2 = self._apply_a3b(x2, self.a3b[3])

        feat_map = None
        if return_map:
            feat_map = x2.reshape(b, s, x2.shape[1], x2.shape[2], x2.shape[3], x2.shape[4])

        x2 = r.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if x2.shape[-1] != int(self.feat_dim):
            raise ValueError(f"feat_dim mismatch: expected {self.feat_dim}, got {int(x2.shape[-1])}")
        return x2.reshape(b, s, -1), feat_map

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self._forward_backbone(x, return_map=False)
        return feat

    def forward_features_and_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat, fmap = self._forward_backbone(x, return_map=True)
        if fmap is None:
            raise RuntimeError("internal error: fmap is None when return_map=True")
        return feat, fmap

    def logits_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"expected feat (B,2,D), got {tuple(feat.shape)}")
        proto_raw = self.prototypes
        if self.frozen_mask.any():
            proto_raw = torch.where(self.frozen_mask[:, None], self.prototypes_init, proto_raw)
        feat = F.normalize(feat, dim=-1)
        proto = F.normalize(proto_raw, dim=-1)
        scale = self.logit_scale.clamp(min=1.0, max=100.0)
        logits = torch.einsum("bsd,cd->bsc", feat, proto) * scale
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.forward_features(x)
        logits = self.logits_from_features(feat)
        return logits, feat

    def forward_with_features_and_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, fmap = self.forward_features_and_maps(x)
        logits = self.logits_from_features(feat)
        return logits, feat, fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits


class _ViTTokenPoolClassifier(nn.Module):
    """
    Classification head on top of a non-classification ViT.

    MONAI ViT with classification=False returns (B, N_tokens, hidden_size).
    We pool tokens (mean) and apply a linear head to get (B, num_classes).
    """

    def __init__(self, vit: nn.Module, *, hidden_size: int, num_classes: int, pool: str) -> None:
        super().__init__()
        self.vit = vit
        self.pool = str(pool).strip().lower()
        self.head = nn.Linear(int(hidden_size), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.vit(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if not torch.is_tensor(y):
            raise TypeError(f"ViT output must be a Tensor, got {type(y)}")
        if y.ndim != 3:
            raise ValueError(f"expected ViT token tensor (B,N,C), got {tuple(y.shape)}")

        if self.pool == "mean":
            feat = y.mean(dim=1)
        elif self.pool == "cls":
            feat = y[:, 0]
        else:
            raise ValueError(f"unknown vit_pool: {self.pool!r} (expected cls|mean)")
        return self.head(feat)


class _ViTTokenPool(nn.Module):
    def __init__(self, vit: nn.Module, *, pool: str) -> None:
        super().__init__()
        self.vit = vit
        self.pool = str(pool).strip().lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.vit(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if not torch.is_tensor(y):
            raise TypeError(f"ViT output must be a Tensor, got {type(y)}")
        if y.ndim != 3:
            raise ValueError(f"expected ViT token tensor (B,N,C), got {tuple(y.shape)}")
        if self.pool == "mean":
            return y.mean(dim=1)
        if self.pool == "cls":
            return y[:, 0]
        raise ValueError(f"unknown vit_pool: {self.pool!r} (expected cls|mean)")


@dataclass(frozen=True)
class DualModelSpec:
    name: str
    kwargs: dict[str, Any]


_RESNET_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d$")
_RESNET_XATTN_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d_xattn$")
_RESNET_A3B_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d_a3b$")
_RESNET_A3B4_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d_a3b4$")
_RESNET_PROTO_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d_proto$")
_RESNET_A3B4_PROTO_RE = re.compile(r"^dual_resnet(?P<depth>\d+)_3d_a3b4_proto$")


def make_dual_model(
    name: str,
    *,
    num_classes: int,
    in_channels: int,
    img_size: tuple[int, int, int],
    vit_patch_size: tuple[int, int, int] = (4, 16, 16),
    vit_pool: str = "cls",
    vit_hidden_size: int = 768,
    vit_mlp_dim: int = 3072,
    vit_num_layers: int = 12,
    vit_num_heads: int = 12,
    unet_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    unet_strides: tuple[int, ...] = (2, 2, 2, 2),
    unet_num_res_units: int = 2,
    proto_init: str | None = None,
    proto_text_model: str = "hfl/chinese-roberta-wwm-ext",
    proto_text_pool: str = "cls",
    proto_text_max_length: int = 256,
    proto_text_proj_seed: int = 42,
    proto_prompts_zh: dict[int, str] | None = None,
) -> tuple[nn.Module, DualModelSpec]:
    """
    Factory for dual-output models backed by MONAI networks.

    Supported:
      - dual_resnet{10,18,34,50,101,152,200}_3d
      - dual_resnet{10,18,34,50,101,152,200}_3d_xattn
      - dual_resnet{10,18,34,50,101,152,200}_3d_a3b
      - dual_resnet{10,18,34,50,101,152,200}_3d_a3b4
      - dual_resnet{10,18,34,50,101,152,200}_3d_proto
      - dual_resnet{10,18,34,50,101,152,200}_3d_a3b4_proto
      - dual_unet_3d
      - dual_vit_3d
      - dual_vit_3d_xattn
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

    m = _RESNET_PROTO_RE.match(name)
    if m:
        depth = m.group("depth")
        fn_name = f"resnet{depth}"
        if not hasattr(nets, fn_name):
            raise ValueError(f"unsupported resnet depth: {depth}")
        fn = getattr(nets, fn_name)
        backbone = fn(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        if getattr(backbone, "fc", None) is None:
            raise ValueError("unexpected ResNet without fc")
        feat_dim = int(backbone.fc.in_features)
        backbone.fc = None
        from medical_fenlei.constants import CLASS_ID_TO_NAME

        default_init = "prompt_hash" if int(num_classes) == len(CLASS_ID_TO_NAME) else "rand"
        init_s = str(proto_init or "").strip().lower()
        if init_s in {"", "auto", "default"}:
            init_s = default_init
        model = DualProtoCosineClassifier(
            backbone,
            feat_dim=int(feat_dim),
            num_classes=int(num_classes),
            init=str(init_s),
            hf_model_name_or_path=str(proto_text_model),
            hf_pool=str(proto_text_pool),
            hf_max_length=int(proto_text_max_length),
            hf_proj_seed=int(proto_text_proj_seed),
            prompts_zh=proto_prompts_zh,
        )
        return model, DualModelSpec(
            name=name,
            kwargs={
                "feat_dim": int(feat_dim),
                "proto_init": str(init_s),
                "proto_text_model": str(proto_text_model),
                "proto_text_pool": str(proto_text_pool),
                "proto_text_max_length": int(proto_text_max_length),
                "proto_text_proj_seed": int(proto_text_proj_seed),
            },
        )

    m = _RESNET_A3B4_PROTO_RE.match(name)
    if m:
        depth = m.group("depth")
        fn_name = f"resnet{depth}"
        if not hasattr(nets, fn_name):
            raise ValueError(f"unsupported resnet depth: {depth}")
        fn = getattr(nets, fn_name)
        backbone = fn(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        if getattr(backbone, "fc", None) is None:
            raise ValueError("unexpected ResNet without fc")
        feat_dim = int(backbone.fc.in_features)
        backbone.fc = None
        from medical_fenlei.constants import CLASS_ID_TO_NAME

        default_init = "prompt_hash" if int(num_classes) == len(CLASS_ID_TO_NAME) else "rand"
        init_s = str(proto_init or "").strip().lower()
        if init_s in {"", "auto", "default"}:
            init_s = default_init
        model = DualResNetA3B4ProtoCosineClassifier(
            backbone,
            feat_dim=int(feat_dim),
            num_classes=int(num_classes),
            init=str(init_s),
            hf_model_name_or_path=str(proto_text_model),
            hf_pool=str(proto_text_pool),
            hf_max_length=int(proto_text_max_length),
            hf_proj_seed=int(proto_text_proj_seed),
            prompts_zh=proto_prompts_zh,
        )
        return model, DualModelSpec(
            name=name,
            kwargs={
                "feat_dim": int(feat_dim),
                "proto_init": str(init_s),
                "proto_text_model": str(proto_text_model),
                "proto_text_pool": str(proto_text_pool),
                "proto_text_max_length": int(proto_text_max_length),
                "proto_text_proj_seed": int(proto_text_proj_seed),
            },
        )

    m = _RESNET_XATTN_RE.match(name)
    if m:
        depth = m.group("depth")
        fn_name = f"resnet{depth}"
        if not hasattr(nets, fn_name):
            raise ValueError(f"unsupported resnet depth: {depth}")
        fn = getattr(nets, fn_name)
        backbone = fn(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        if getattr(backbone, "fc", None) is None:
            raise ValueError("unexpected ResNet without fc")
        feat_dim = int(backbone.fc.in_features)
        backbone.fc = None
        model = DualCrossEarAttention(backbone, feat_dim=int(feat_dim), num_classes=int(num_classes))
        return model, DualModelSpec(name=name, kwargs={"feat_dim": int(feat_dim)})

    m = _RESNET_A3B_RE.match(name)
    if m:
        depth = m.group("depth")
        fn_name = f"resnet{depth}"
        if not hasattr(nets, fn_name):
            raise ValueError(f"unsupported resnet depth: {depth}")
        fn = getattr(nets, fn_name)
        backbone = fn(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        if getattr(backbone, "fc", None) is None:
            raise ValueError("unexpected ResNet without fc")
        feat_dim = int(backbone.fc.in_features)
        trunk = _ResNetStage4(backbone)
        model = DualResNetA3B(trunk, feat_dim=int(feat_dim), num_classes=int(num_classes))
        return model, DualModelSpec(name=name, kwargs={"feat_dim": int(feat_dim)})

    m = _RESNET_A3B4_RE.match(name)
    if m:
        depth = m.group("depth")
        fn_name = f"resnet{depth}"
        if not hasattr(nets, fn_name):
            raise ValueError(f"unsupported resnet depth: {depth}")
        fn = getattr(nets, fn_name)
        backbone = fn(spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        if getattr(backbone, "fc", None) is None:
            raise ValueError("unexpected ResNet without fc")
        feat_dim = int(backbone.fc.in_features)
        backbone.fc = None
        model = DualResNetA3B4(backbone, feat_dim=int(feat_dim), num_classes=int(num_classes))
        return model, DualModelSpec(name=name, kwargs={"feat_dim": int(feat_dim)})

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
        vit_pool_s = str(vit_pool).strip().lower()
        if vit_pool_s == "cls":
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
        else:
            vit = nets.ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=vit_patch_size,
                hidden_size=int(vit_hidden_size),
                mlp_dim=int(vit_mlp_dim),
                num_layers=int(vit_num_layers),
                num_heads=int(vit_num_heads),
                classification=False,
                spatial_dims=3,
            )
            base = _ViTTokenPoolClassifier(vit, hidden_size=int(vit_hidden_size), num_classes=int(num_classes), pool=vit_pool_s)
        return (
            DualInputWrapper(base),
            DualModelSpec(
                name=name,
                kwargs={
                    "img_size": tuple(int(x) for x in img_size),
                    "vit_patch_size": tuple(int(x) for x in vit_patch_size),
                    "vit_pool": str(vit_pool_s),
                    "vit_hidden_size": int(vit_hidden_size),
                    "vit_mlp_dim": int(vit_mlp_dim),
                    "vit_num_layers": int(vit_num_layers),
                    "vit_num_heads": int(vit_num_heads),
                },
            ),
        )

    if name == "dual_vit_3d_xattn":
        if any(s <= 0 for s in img_size):
            raise ValueError(f"invalid img_size: {img_size}")
        if any(p <= 0 for p in vit_patch_size):
            raise ValueError(f"invalid vit_patch_size: {vit_patch_size}")
        for dim, p in zip(img_size, vit_patch_size):
            if int(dim) % int(p) != 0:
                raise ValueError(f"ViT requires img_size {img_size} divisible by patch_size {vit_patch_size}")
        vit_pool_s = str(vit_pool).strip().lower()
        vit = nets.ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=vit_patch_size,
            hidden_size=int(vit_hidden_size),
            mlp_dim=int(vit_mlp_dim),
            num_layers=int(vit_num_layers),
            num_heads=int(vit_num_heads),
            classification=False,
            spatial_dims=3,
        )
        extractor = _ViTTokenPool(vit, pool=vit_pool_s)
        model = DualCrossEarAttention(extractor, feat_dim=int(vit_hidden_size), num_classes=int(num_classes))
        return (
            model,
            DualModelSpec(
                name=name,
                kwargs={
                    "img_size": tuple(int(x) for x in img_size),
                    "vit_patch_size": tuple(int(x) for x in vit_patch_size),
                    "vit_pool": str(vit_pool_s),
                    "vit_hidden_size": int(vit_hidden_size),
                    "vit_mlp_dim": int(vit_mlp_dim),
                    "vit_num_layers": int(vit_num_layers),
                    "vit_num_heads": int(vit_num_heads),
                },
            ),
        )

    raise ValueError(f"unknown model: {name}")
