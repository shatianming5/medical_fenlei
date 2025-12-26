from __future__ import annotations

from typing import Any

import torch.nn as nn

from medical_fenlei.models.slice_attention_resnet import SliceAttentionResNet
from medical_fenlei.models.slice_attention_unet import SliceAttentionUNet
from medical_fenlei.models.slice_attention_vit import SliceAttentionViT


def normalize_ear2d_model_type(model_type: str) -> str:
    t = str(model_type).strip()
    t_low = t.lower()
    if t_low in {"resnet", "sliceattentionresnet", "slice_attention_resnet"}:
        return "SliceAttentionResNet"
    if t_low in {"vit", "sliceattentionvit", "slice_attention_vit"}:
        return "SliceAttentionViT"
    if t_low in {"unet", "sliceattentionunet", "slice_attention_unet"}:
        return "SliceAttentionUNet"
    return t


def make_ear2d_model(*, model_type: str, model_spec: dict[str, Any], in_channels: int = 1) -> nn.Module:
    mt = normalize_ear2d_model_type(model_type)
    if mt == "SliceAttentionResNet":
        return SliceAttentionResNet.from_spec(model_spec, in_channels=int(in_channels))
    if mt == "SliceAttentionViT":
        return SliceAttentionViT.from_spec(model_spec, in_channels=int(in_channels))
    if mt == "SliceAttentionUNet":
        return SliceAttentionUNet.from_spec(model_spec, in_channels=int(in_channels))
    raise ValueError(f"unknown ear2d model_type: {model_type!r}")


def model_type_from_checkpoint(ckpt: dict[str, Any]) -> str:
    cfg = ckpt.get("config") or {}
    mt = ckpt.get("model_type") or (cfg.get("model") or {}).get("type") or "SliceAttentionResNet"
    return normalize_ear2d_model_type(str(mt))


def model_spec_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    cfg = ckpt.get("config") or {}
    return ckpt.get("model_spec") or (cfg.get("model") or {}).get("spec") or {}


def make_ear2d_model_from_checkpoint(*, ckpt: dict[str, Any], in_channels: int = 1) -> nn.Module:
    return make_ear2d_model(model_type=model_type_from_checkpoint(ckpt), model_spec=model_spec_from_checkpoint(ckpt), in_channels=int(in_channels))

