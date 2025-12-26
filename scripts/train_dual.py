from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
import gc
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dual_dataset import DualPreprocessSpec, EarCTDualDataset
from medical_fenlei.metrics import classification_report_from_confusion
from medical_fenlei.models.dual_factory import make_dual_model
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    # logits: (B,2,C)  labels: (B,2)  mask: (B,2)
    if logits.ndim != 3:
        raise ValueError(f"expected logits (B,2,C), got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels shape {tuple(labels.shape)} != {tuple(logits.shape[:2])}")
    if label_mask.shape != labels.shape:
        raise ValueError(f"mask shape {tuple(label_mask.shape)} != {tuple(labels.shape)}")

    total = torch.tensor(0.0, device=logits.device)
    n = 0
    for side in (0, 1):
        m = label_mask[:, side].bool()
        if m.any():
            total = total + loss_fn(logits[m, side], labels[m, side])
            n += 1
    if n <= 0:
        return total
    return total / float(n)


@torch.no_grad()
def _masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)  # (B,2)
    mask = label_mask.bool()
    total_n = int(mask.sum().item())
    if total_n <= 0:
        out = {"acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "n": 0, "left_n": 0, "right_n": 0}
        if logits.shape[-1] == 2:
            out.update(
                {
                    "pred_pos": 0,
                    "pred_pos_rate": 0.0,
                    "left_pred_pos": 0,
                    "left_pred_pos_rate": 0.0,
                    "right_pred_pos": 0,
                    "right_pred_pos_rate": 0.0,
                }
            )
        return out

    total_correct = int(((pred == labels) & mask).sum().item())

    left_mask = mask[:, 0]
    right_mask = mask[:, 1]
    left_n = int(left_mask.sum().item())
    right_n = int(right_mask.sum().item())
    left_correct = int(((pred[:, 0] == labels[:, 0]) & left_mask).sum().item())
    right_correct = int(((pred[:, 1] == labels[:, 1]) & right_mask).sum().item())

    left_acc = float(left_correct / left_n) if left_n > 0 else 0.0
    right_acc = float(right_correct / right_n) if right_n > 0 else 0.0

    out = {
        "acc": float(total_correct / total_n),
        "left_acc": left_acc,
        "right_acc": right_acc,
        "n": total_n,
        "left_n": left_n,
        "right_n": right_n,
    }
    if logits.shape[-1] == 2:
        pred_pos = int(((pred == 1) & mask).sum().item())
        left_pred_pos = int(((pred[:, 0] == 1) & left_mask).sum().item())
        right_pred_pos = int(((pred[:, 1] == 1) & right_mask).sum().item())
        out.update(
            {
                "pred_pos": pred_pos,
                "pred_pos_rate": float(pred_pos / total_n),
                "left_pred_pos": left_pred_pos,
                "left_pred_pos_rate": float(left_pred_pos / left_n) if left_n > 0 else 0.0,
                "right_pred_pos": right_pred_pos,
                "right_pred_pos_rate": float(right_pred_pos / right_n) if right_n > 0 else 0.0,
            }
        )
    return out


def _make_adamw(model: torch.nn.Module, *, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """
    AdamW with standard no-decay filtering (bias/Norm/positional/cls params).

    This tends to be important for transformer-like backbones (ViT3D), and is
    generally safe for ResNet/UNet as well.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = str(name).lower()
        if (
            p.ndim == 1
            or n.endswith(".bias")
            or "bias" in n
            or "norm" in n
            or "bn" in n
            or "layernorm" in n
            or "position" in n
            or "pos_embed" in n
            or "cls_token" in n
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=float(lr))


def _parse_int_tuple(value: str, *, n: int | None = None) -> tuple[int, ...]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out = tuple(int(x) for x in parts)
    if n is not None and len(out) != n:
        raise ValueError(f"expected {n} ints, got {len(out)}: {value!r}")
    return out


def _filter_df_for_codes(df: pd.DataFrame, *, codes: set[int]) -> pd.DataFrame:
    if not codes:
        return df
    if "left_code" not in df.columns or "right_code" not in df.columns:
        return df
    mask = df["left_code"].isin(codes) | df["right_code"].isin(codes)
    return df.loc[mask].reset_index(drop=True)


def _count_codes(df: pd.DataFrame, *, codes: set[int]) -> int:
    if not codes:
        return 0
    if "left_code" not in df.columns or "right_code" not in df.columns:
        return 0
    return int(df["left_code"].isin(codes).sum() + df["right_code"].isin(codes).sum())


def _count_nonempty_report_text(df: pd.DataFrame) -> int:
    if "report_text" not in df.columns:
        return 0
    s = df["report_text"].fillna("").astype(str).str.strip()
    return int(s.ne("").sum())


def _ear_label_ids_from_df(df: pd.DataFrame) -> pd.Series:
    if "left_code" not in df.columns or "right_code" not in df.columns:
        return pd.Series([], dtype=np.int64)
    codes = pd.concat([df["left_code"], df["right_code"]], ignore_index=True)
    codes = pd.to_numeric(codes, errors="coerce").dropna()
    codes = codes[(codes >= 1) & (codes <= len(CLASS_ID_TO_NAME))].astype(np.int64)
    return (codes - 1).astype(np.int64)


def _make_class_weight(
    *,
    train_df: pd.DataFrame,
    num_classes: int,
    task_kind: str,
    task_pos_label_ids: tuple[int, ...] | None,
    task_neg_label_ids: tuple[int, ...] | None,
) -> torch.Tensor | None:
    ids = _ear_label_ids_from_df(train_df)
    if ids.empty:
        return None

    if str(task_kind) == "binary":
        if not task_pos_label_ids or not task_neg_label_ids:
            return None
        pos_set = set(int(x) for x in task_pos_label_ids)
        neg_set = set(int(x) for x in task_neg_label_ids)
        pos = int(ids.isin(list(pos_set)).sum())
        neg = int(ids.isin(list(neg_set)).sum())
        total = pos + neg
        if total <= 0:
            return None
        # class 0=neg, 1=pos
        w0 = float(total) / (2.0 * float(max(1, neg)))
        w1 = float(total) / (2.0 * float(max(1, pos)))
        return torch.tensor([w0, w1], dtype=torch.float32)

    # multi-class: inverse frequency
    vc = ids.value_counts().to_dict()
    total = float(len(ids))
    if total <= 0:
        return None
    weights = []
    for c in range(int(num_classes)):
        cnt = float(vc.get(int(c), 0.0))
        if cnt <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (float(num_classes) * cnt))
    return torch.tensor(weights, dtype=torch.float32)


def _make_exam_sampling_weights(
    *,
    train_df: pd.DataFrame,
    task_kind: str,
    task_pos_label_ids: tuple[int, ...] | None,
    task_neg_label_ids: tuple[int, ...] | None,
    class_weight: torch.Tensor | None,
    mode: str,
) -> torch.Tensor | None:
    if class_weight is None:
        return None
    if "left_code" not in train_df.columns or "right_code" not in train_df.columns:
        return None

    mode = str(mode or "mean").strip().lower()
    if mode not in {"mean", "max", "sum"}:
        raise ValueError(f"unknown balanced_sampler_mode: {mode!r} (expected mean|max|sum)")

    w = []
    pos_set = set(int(x) for x in (task_pos_label_ids or ()))
    neg_set = set(int(x) for x in (task_neg_label_ids or ()))

    for _, row in train_df.iterrows():
        ear_ws: list[float] = []
        for col in ("left_code", "right_code"):
            v = row.get(col)
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                code = int(v)
            except Exception:
                continue
            if not (1 <= int(code) <= len(CLASS_ID_TO_NAME)):
                continue
            lid = int(code) - 1

            if str(task_kind) == "binary":
                if lid in pos_set:
                    ear_ws.append(float(class_weight[1].item()))
                elif lid in neg_set:
                    ear_ws.append(float(class_weight[0].item()))
                else:
                    continue
            else:
                if 0 <= lid < int(class_weight.numel()):
                    ear_ws.append(float(class_weight[lid].item()))

        if not ear_ws:
            w.append(1.0)
        elif mode == "max":
            w.append(float(np.max(np.asarray(ear_ws, dtype=np.float64))))
        elif mode == "sum":
            w.append(float(np.sum(np.asarray(ear_ws, dtype=np.float64))))
        else:
            w.append(float(np.mean(np.asarray(ear_ws, dtype=np.float64))))

    if not w:
        return None
    out = torch.tensor(w, dtype=torch.double)
    if not torch.isfinite(out).all():
        out = torch.ones_like(out)
    return out


def _apply_binary_task(
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_label_ids: tuple[int, ...],
    neg_label_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map 6-class labels (0..5) to binary labels (0/1) and update mask.

    Any label not in pos/neg sets will be masked out (ignored).
    """
    if labels.ndim != 2:
        raise ValueError(f"expected labels (B,2), got {tuple(labels.shape)}")
    if mask.shape != labels.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != labels shape {tuple(labels.shape)}")

    pos_m = torch.zeros_like(mask, dtype=torch.bool)
    for v in pos_label_ids:
        pos_m |= labels == int(v)
    neg_m = torch.zeros_like(mask, dtype=torch.bool)
    for v in neg_label_ids:
        neg_m |= labels == int(v)

    keep = mask.bool() & (pos_m | neg_m)
    out = torch.zeros_like(labels)
    out[pos_m] = 1
    return out, keep


def _autotune_batch_size(
    *,
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    in_channels: int,
    num_slices: int,
    image_size: int,
    amp: bool,
    max_batch_size: int,
) -> int:
    if device.type != "cuda":
        return 1

    loss_fn = torch.nn.CrossEntropyLoss()
    model.train(True)

    def _try(bs: int) -> bool:
        torch.cuda.empty_cache()
        try:
            # IMPORTANT: include AdamW optimizer state allocation in the probe, otherwise
            # auto_batch can be overly optimistic for larger models (e.g. ResNet34+),
            # leading to OOM at the first real optimizer step.
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
            x = torch.randn(bs, 2, int(in_channels), num_slices, image_size, image_size, device=device)
            y = torch.randint(0, num_classes, (bs, 2), device=device, dtype=torch.long).view(-1)

            amp_enabled = amp and device.type == "cuda"
            if amp_enabled:
                if hasattr(torch, "amp"):
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                else:
                    autocast_ctx = torch.cuda.amp.autocast()
            else:
                autocast_ctx = nullcontext()

            model.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x).reshape(-1, num_classes)
                loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            del optimizer, x, y, logits, loss
            model.zero_grad(set_to_none=True)
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "oom" in msg:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return False
            raise

    lo = 1
    hi = 1
    while hi <= max_batch_size and _try(hi):
        lo = hi
        hi *= 2
    hi = min(hi, max_batch_size + 1)

    # binary search in (lo, hi)
    left, right = lo, hi
    while left + 1 < right:
        mid = (left + right) // 2
        if _try(mid):
            left = mid
        else:
            right = mid
    return int(left)


def _is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or ("cuda" in msg and "oom" in msg)


class FocalLoss(nn.Module):
    """
    Multi-class focal loss for integer targets.

    loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is softmax(logits)[target].
    """

    def __init__(self, *, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"expected logits (N,C), got {tuple(logits.shape)}")
        if target.ndim != 1:
            raise ValueError(f"expected target (N,), got {tuple(target.shape)}")
        if target.numel() <= 0:
            return logits.sum() * 0.0

        logp = torch.log_softmax(logits, dim=-1)  # (N,C)
        p = logp.exp()
        t = target.to(torch.long)
        pt = p.gather(1, t.view(-1, 1)).squeeze(1).clamp(min=1e-8, max=1.0)  # (N,)
        logpt = logp.gather(1, t.view(-1, 1)).squeeze(1)  # (N,)

        if self.alpha.numel() > 0:
            a = self.alpha.to(logits.device, dtype=logits.dtype)
            at = a.gather(0, t).to(logits.dtype)
        else:
            at = 1.0

        loss = -at * ((1.0 - pt) ** float(self.gamma)) * logpt
        return loss.mean()


def _init_classifier_bias_from_prior(model: torch.nn.Module, *, num_classes: int, priors: list[float]) -> None:
    if int(num_classes) <= 1:
        return
    if len(priors) != int(num_classes):
        return
    if any((not math.isfinite(float(p))) or float(p) <= 0 for p in priors):
        return
    s = float(sum(float(p) for p in priors))
    if not math.isfinite(s) or s <= 0:
        return
    priors = [float(p) / s for p in priors]

    # Find the last classifier-like layer.
    base = getattr(model, "base", model)
    cand: list[nn.Module] = []
    for _, m in base.named_modules():
        if isinstance(m, nn.Linear) and int(m.out_features) == int(num_classes) and m.bias is not None:
            cand.append(m)
        elif isinstance(m, nn.Conv3d) and int(m.out_channels) == int(num_classes) and m.bias is not None:
            cand.append(m)
    if not cand:
        return
    head = cand[-1]

    with torch.no_grad():
        if int(num_classes) == 2:
            p1 = float(min(1.0 - 1e-4, max(1e-4, priors[1])))
            p0 = float(1.0 - p1)
            delta = float(math.log(p1 / p0))
            b0 = -0.5 * delta
            b1 = 0.5 * delta
            head.bias.data[0] = float(b0)
            head.bias.data[1] = float(b1)
            return

        # Multi-class: set bias = log(prior) (up to an additive constant).
        v = torch.tensor([math.log(float(min(1.0 - 1e-8, max(1e-8, p)))) for p in priors], device=head.bias.device, dtype=head.bias.dtype)
        v = v - v.mean()
        head.bias.data.copy_(v)


def _rand_uniform(shape: tuple[int, ...], *, device: torch.device, low: float, high: float, dtype: torch.dtype) -> torch.Tensor:
    if high < low:
        low, high = high, low
    return (low + (high - low) * torch.rand(shape, device=device, dtype=dtype)).to(dtype)


def _augment_batch(
    x: torch.Tensor,
    *,
    flip_prob: float,
    intensity_prob: float,
    noise_prob: float,
    gamma_prob: float,
) -> torch.Tensor:
    # x: (B,2,1,D,H,W) in [0,1]
    if x.ndim != 6:
        return x
    if max(flip_prob, intensity_prob, noise_prob, gamma_prob) <= 0:
        return x

    b = int(x.shape[0])
    s = int(x.shape[1])
    n = b * s
    x2 = x.reshape(n, *x.shape[2:]).contiguous()  # (N,1,D,H,W)

    orig_dtype = x2.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        x2 = x2.float()

    device = x2.device

    # Random flips (per sample) on H/W.
    if flip_prob > 0:
        do_h = torch.rand((n,), device=device) < float(flip_prob)
        do_w = torch.rand((n,), device=device) < float(flip_prob)
        if do_h.any():
            x2[do_h] = x2[do_h].flip(-2)
        if do_w.any():
            x2[do_w] = x2[do_w].flip(-1)

    # Random intensity scale/shift.
    if intensity_prob > 0:
        do_i = torch.rand((n,), device=device) < float(intensity_prob)
        if do_i.any():
            scale = _rand_uniform((int(do_i.sum()), 1, 1, 1, 1), device=device, low=0.90, high=1.10, dtype=x2.dtype)
            shift = _rand_uniform((int(do_i.sum()), 1, 1, 1, 1), device=device, low=-0.10, high=0.10, dtype=x2.dtype)
            x2_do = x2[do_i]
            x2_do = x2_do * scale + shift
            x2[do_i] = x2_do

    # Random gamma.
    if gamma_prob > 0:
        do_g = torch.rand((n,), device=device) < float(gamma_prob)
        if do_g.any():
            gamma = _rand_uniform((int(do_g.sum()), 1, 1, 1, 1), device=device, low=0.70, high=1.50, dtype=x2.dtype)
            x2_do = x2[do_g].clamp(min=1e-6, max=1.0)
            x2[do_g] = x2_do**gamma

    # Random gaussian noise.
    if noise_prob > 0:
        do_n = torch.rand((n,), device=device) < float(noise_prob)
        if do_n.any():
            std = _rand_uniform((int(do_n.sum()), 1, 1, 1, 1), device=device, low=0.0, high=0.03, dtype=x2.dtype)
            x2[do_n] = x2[do_n] + torch.randn_like(x2[do_n]) * std

    x2 = x2.clamp(0.0, 1.0)
    if x2.dtype != orig_dtype:
        x2 = x2.to(orig_dtype)
    return x2.reshape(b, s, *x.shape[2:]).contiguous()


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 0.0,
    proto_reg_lambda: float = 0.0,
    vl_contrastive_lambda: float = 0.0,
    vl_temperature: float = 0.07,
    vl_text_vectorizer=None,
    vl_local_lambda: float = 0.0,
    vl_local_temperature: float = 0.07,
    vl_local_max_triplets_per_ear: int = 8,
    vl_local_drop_negated: bool = True,
    bilat_loss_lambda: float = 0.0,
    bilat_unilateral_sim_max: float = 0.0,
    bilat_normal_class_id: int | None = None,
    augment: bool = False,
    aug_flip_prob: float = 0.0,
    aug_intensity_prob: float = 0.0,
    aug_noise_prob: float = 0.0,
    aug_gamma_prob: float = 0.0,
    task_pos_label_ids: tuple[int, ...] | None = None,
    task_neg_label_ids: tuple[int, ...] | None = None,
    num_classes: int | None = None,
    collect_cm: bool = False,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_proto_reg = 0.0
    total_vl_loss = 0.0
    total_vl_pairs = 0
    vl_batches = 0
    total_vl_local_loss = 0.0
    total_vl_local_queries = 0
    vl_local_batches = 0
    total_bilat_loss = 0.0
    bilat_normal_pairs = 0
    bilat_unilateral_pairs = 0
    bilat_batches = 0
    total_acc = 0.0
    total_left = 0.0
    total_right = 0.0
    total_n = 0
    total_left_n = 0
    total_right_n = 0
    total_pred_pos = 0
    total_left_pred_pos = 0
    total_right_pred_pos = 0
    n_batches = 0

    cm = None
    if collect_cm:
        if num_classes is None:
            raise ValueError("num_classes is required when collect_cm=True")
        cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.int64)

    if is_train:
        grad_ctx = nullcontext()
    else:
        grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()

    with grad_ctx:
        accum_steps = max(1, int(grad_accum_steps))
        accum_i = 0
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["label_mask"].to(device, non_blocking=True)

            if task_pos_label_ids is not None and task_neg_label_ids is not None:
                y, m = _apply_binary_task(y, m, pos_label_ids=task_pos_label_ids, neg_label_ids=task_neg_label_ids)

            if is_train and augment:
                x = _augment_batch(
                    x,
                    flip_prob=float(aug_flip_prob),
                    intensity_prob=float(aug_intensity_prob),
                    noise_prob=float(aug_noise_prob),
                    gamma_prob=float(aug_gamma_prob),
                )

            amp_enabled = amp and device.type == "cuda"
            if amp_enabled:
                if hasattr(torch, "amp"):
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                else:
                    autocast_ctx = torch.cuda.amp.autocast()
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                feat = None
                feat_map = None
                need_map = bool(is_train) and float(vl_local_lambda) > 0
                need_feat = bool(is_train) and (
                    (float(vl_contrastive_lambda) > 0 and vl_text_vectorizer is not None) or (float(bilat_loss_lambda) > 0) or need_map
                )
                if need_feat:
                    if need_map:
                        fwm = getattr(model, "forward_with_features_and_maps", None)
                        if fwm is None and hasattr(model, "_orig_mod"):
                            fwm = getattr(model._orig_mod, "forward_with_features_and_maps", None)
                        if callable(fwm):
                            logits, feat, feat_map = fwm(x)
                        else:
                            fwf = getattr(model, "forward_with_features", None)
                            if fwf is None and hasattr(model, "_orig_mod"):
                                fwf = getattr(model._orig_mod, "forward_with_features", None)
                            if callable(fwf):
                                logits, feat = fwf(x)
                            else:
                                logits = model(x)
                    else:
                        fwf = getattr(model, "forward_with_features", None)
                        if fwf is None and hasattr(model, "_orig_mod"):
                            fwf = getattr(model._orig_mod, "forward_with_features", None)
                        if callable(fwf):
                            logits, feat = fwf(x)
                        else:
                            logits = model(x)
                else:
                    logits = model(x)  # (B,2,C)
                loss = _masked_ce_loss(logits, y, m, loss_fn=loss_fn)

            proto_reg = None
            if is_train and float(proto_reg_lambda) > 0:
                proto_fn = getattr(model, "prototype_reg_loss", None)
                if proto_fn is None and hasattr(model, "_orig_mod"):
                    proto_fn = getattr(model._orig_mod, "prototype_reg_loss", None)
                if callable(proto_fn):
                    proto_reg = proto_fn()
                    loss = loss + float(proto_reg_lambda) * proto_reg

            vl_loss = None
            if (
                bool(is_train)
                and float(vl_contrastive_lambda) > 0
                and vl_text_vectorizer is not None
                and feat is not None
                and "report_text" in batch
            ):
                texts = batch.get("report_text")
                if isinstance(texts, (list, tuple)):
                    texts_list = [str(t) for t in texts]
                else:
                    texts_list = [str(texts)]

                valid_idx = [i for i, t in enumerate(texts_list) if str(t).strip()]
                if len(valid_idx) >= 2:
                    img_emb = feat.mean(dim=1)  # (B,D)
                    img_emb = img_emb[valid_idx]
                    texts_valid = [texts_list[i] for i in valid_idx]
                    try:
                        mat = vl_text_vectorizer.transform(texts_valid)
                        if hasattr(mat, "toarray"):
                            mat = mat.toarray()
                        mat = np.asarray(mat, dtype=np.float32)
                    except Exception:
                        mat = None

                    if mat is not None:
                        txt_emb = torch.from_numpy(mat).to(device=device)
                        img_emb_f = F.normalize(img_emb.float(), dim=-1)
                        txt_emb_f = F.normalize(txt_emb.float(), dim=-1)
                        temp = float(vl_temperature)
                        if not math.isfinite(temp) or temp <= 0:
                            raise ValueError(f"vl_temperature must be > 0, got {vl_temperature!r}")
                        logits_it = (img_emb_f @ txt_emb_f.t()) / temp
                        targets = torch.arange(int(logits_it.shape[0]), device=device, dtype=torch.long)
                        vl_loss = (F.cross_entropy(logits_it, targets) + F.cross_entropy(logits_it.t(), targets)) * 0.5
                        loss = loss + float(vl_contrastive_lambda) * vl_loss

            vl_local_loss = None
            vl_local_queries = 0
            if (
                bool(is_train)
                and float(vl_local_lambda) > 0
                and vl_text_vectorizer is not None
                and feat_map is not None
                and "report_text" in batch
                and int(vl_local_max_triplets_per_ear) > 0
            ):
                from medical_fenlei.text_triplets import extract_entity_attribute_triplets

                texts = batch.get("report_text")
                if isinstance(texts, (list, tuple)):
                    texts_list = [str(t) for t in texts]
                else:
                    texts_list = [str(texts)]

                # Build per-ear triplet queries and classify them to the correct ear in the batch.
                max_k = int(vl_local_max_triplets_per_ear)
                drop_neg = bool(vl_local_drop_negated)
                q_texts: list[str] = []
                q_targets: list[int] = []
                per_counts = [[0, 0] for _ in range(int(len(texts_list)))]

                for i, t in enumerate(texts_list):
                    text = str(t).strip()
                    if not text:
                        continue
                    triplets = extract_entity_attribute_triplets(text)
                    if not triplets:
                        continue

                    for tr in triplets:
                        if drop_neg and bool(tr.negated):
                            continue
                        loc = str(getattr(tr, "location", "U") or "U").strip().upper()
                        if loc == "L":
                            ears = (0,)
                        elif loc == "R":
                            ears = (1,)
                        else:
                            ears = (0, 1)

                        for ear in ears:
                            if per_counts[i][int(ear)] >= max_k:
                                continue
                            prefix = "左侧" if int(ear) == 0 else "右侧"
                            neg = "无" if bool(tr.negated) else ""
                            q_texts.append(f"{prefix} {tr.entity} {neg}{tr.attribute}".strip())
                            q_targets.append(int(i) * 2 + int(ear))
                            per_counts[i][int(ear)] += 1

                        if per_counts[i][0] >= max_k and per_counts[i][1] >= max_k:
                            break

                if q_texts:
                    try:
                        mat = vl_text_vectorizer.transform(q_texts)
                        if hasattr(mat, "toarray"):
                            mat = mat.toarray()
                        mat = np.asarray(mat, dtype=np.float32)
                    except Exception:
                        mat = None

                    if mat is not None:
                        txt = torch.from_numpy(mat).to(device=device)
                        txt = F.normalize(txt.float(), dim=-1)  # (T,D)

                        fmap = feat_map
                        if fmap.ndim != 6 or int(fmap.shape[1]) != 2:
                            raise ValueError(f"vl_local expects feat_map (B,2,C,D,H,W), got {tuple(fmap.shape)}")
                        bsz = int(fmap.shape[0])
                        ch = int(fmap.shape[2])
                        fmap2 = fmap.reshape(bsz * 2, ch, int(fmap.shape[3]), int(fmap.shape[4]), int(fmap.shape[5]))
                        tokens = fmap2.flatten(2).transpose(1, 2).contiguous()  # (E,N,C)
                        tokens = F.normalize(tokens.float(), dim=-1)

                        e = int(tokens.shape[0])
                        n_tok = int(tokens.shape[1])
                        tokens_flat = tokens.reshape(e * n_tok, int(tokens.shape[2]))

                        sim = txt @ tokens_flat.t()  # (T, E*N)
                        sim = sim.view(int(txt.shape[0]), e, n_tok).amax(dim=-1)  # (T,E)
                        temp = float(vl_local_temperature)
                        if not math.isfinite(temp) or temp <= 0:
                            raise ValueError(f"vl_local_temperature must be > 0, got {vl_local_temperature!r}")
                        logits_te = sim / temp
                        targets = torch.tensor(q_targets, device=device, dtype=torch.long)
                        vl_local_loss = F.cross_entropy(logits_te, targets)
                        vl_local_queries = int(len(q_texts))
                        loss = loss + float(vl_local_lambda) * vl_local_loss

            bilat_loss = None
            if bool(is_train) and float(bilat_loss_lambda) > 0 and feat is not None and bilat_normal_class_id is not None:
                if feat.ndim != 3 or int(feat.shape[1]) != 2:
                    raise ValueError(f"bilat loss expects feat (B,2,D), got {tuple(feat.shape)}")
                both_present = m[:, 0].bool() & m[:, 1].bool()
                if both_present.any():
                    normal_id = int(bilat_normal_class_id)
                    y0 = y[:, 0]
                    y1 = y[:, 1]
                    is_both_normal = both_present & (y0 == normal_id) & (y1 == normal_id)
                    is_unilateral = both_present & ((y0 == normal_id) ^ (y1 == normal_id))

                    f0 = F.normalize(feat[:, 0].float(), dim=-1)
                    f1 = F.normalize(feat[:, 1].float(), dim=-1)
                    sim = (f0 * f1).sum(dim=-1)  # (B,) cosine similarity

                    terms: list[torch.Tensor] = []
                    if is_both_normal.any():
                        loss_n = (1.0 - sim[is_both_normal]).mean()
                        terms.append(loss_n)
                        bilat_normal_pairs = bilat_normal_pairs + int(is_both_normal.sum().item())
                    if is_unilateral.any():
                        sim_max = float(bilat_unilateral_sim_max)
                        loss_u = F.relu(sim[is_unilateral] - sim_max).mean()
                        terms.append(loss_u)
                        bilat_unilateral_pairs = bilat_unilateral_pairs + int(is_unilateral.sum().item())
                    if terms:
                        bilat_loss = sum(terms) / float(len(terms))
                        loss = loss + float(bilat_loss_lambda) * bilat_loss

            if is_train:
                loss_step = loss / float(accum_steps)
                if scaler is not None:
                    scaler.scale(loss_step).backward()
                else:
                    loss_step.backward()

                accum_i += 1
                if accum_i >= accum_steps:
                    if scaler is not None:
                        if float(grad_clip_norm) > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if float(grad_clip_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_i = 0

            metrics = _masked_accuracy(logits.detach(), y.detach(), m.detach())
            left_n = int(m[:, 0].sum().item())
            right_n = int(m[:, 1].sum().item())

            total_loss += float(loss.detach().cpu().item())
            if proto_reg is not None:
                total_proto_reg += float(proto_reg.detach().cpu().item())
            if vl_loss is not None:
                total_vl_loss += float(vl_loss.detach().cpu().item())
                total_vl_pairs += int(len(valid_idx))
                vl_batches += 1
            if vl_local_loss is not None:
                total_vl_local_loss += float(vl_local_loss.detach().cpu().item())
                total_vl_local_queries += int(vl_local_queries)
                vl_local_batches += 1
            if bilat_loss is not None:
                total_bilat_loss += float(bilat_loss.detach().cpu().item())
                bilat_batches += 1
            total_acc += float(metrics["acc"]) * int(metrics["n"])
            total_left += float(metrics["left_acc"]) * left_n
            total_right += float(metrics["right_acc"]) * right_n
            total_n += int(metrics["n"])
            total_left_n += left_n
            total_right_n += right_n
            total_pred_pos += int(metrics.get("pred_pos", 0))
            total_left_pred_pos += int(metrics.get("left_pred_pos", 0))
            total_right_pred_pos += int(metrics.get("right_pred_pos", 0))
            n_batches += 1

            if cm is not None:
                pred = logits.detach().argmax(dim=-1).cpu()
                y_cpu = y.detach().cpu()
                m_cpu = m.detach().cpu().bool()
                y_flat = y_cpu[m_cpu].view(-1).to(torch.int64)
                p_flat = pred[m_cpu].view(-1).to(torch.int64)
                if y_flat.numel() > 0:
                    idx = y_flat * int(num_classes) + p_flat
                    binc = torch.bincount(idx, minlength=int(num_classes) * int(num_classes))
                    cm += binc.view(int(num_classes), int(num_classes))

        # Flush last partial accumulation.
        if is_train and accum_i > 0:
            if scaler is not None:
                if float(grad_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                if float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if total_n <= 0:
        out = {
            "loss": total_loss / max(n_batches, 1),
            "proto_reg": (total_proto_reg / max(n_batches, 1)) if total_proto_reg > 0 else 0.0,
            "vl_loss": (total_vl_loss / max(vl_batches, 1)) if vl_batches > 0 else 0.0,
            "vl_pairs": int(total_vl_pairs),
            "vl_local_loss": (total_vl_local_loss / max(vl_local_batches, 1)) if vl_local_batches > 0 else 0.0,
            "vl_local_queries": int(total_vl_local_queries),
            "bilat_loss": (total_bilat_loss / max(bilat_batches, 1)) if bilat_batches > 0 else 0.0,
            "bilat_normal_pairs": int(bilat_normal_pairs),
            "bilat_unilateral_pairs": int(bilat_unilateral_pairs),
            "acc": 0.0,
            "left_acc": 0.0,
            "right_acc": 0.0,
            "n": 0,
        }
        if cm is not None:
            out["confusion_matrix"] = cm
        return out

    out = {
        "loss": total_loss / max(n_batches, 1),
        "proto_reg": (total_proto_reg / max(n_batches, 1)) if total_proto_reg > 0 else 0.0,
        "vl_loss": (total_vl_loss / max(vl_batches, 1)) if vl_batches > 0 else 0.0,
        "vl_pairs": int(total_vl_pairs),
        "vl_local_loss": (total_vl_local_loss / max(vl_local_batches, 1)) if vl_local_batches > 0 else 0.0,
        "vl_local_queries": int(total_vl_local_queries),
        "bilat_loss": (total_bilat_loss / max(bilat_batches, 1)) if bilat_batches > 0 else 0.0,
        "bilat_normal_pairs": int(bilat_normal_pairs),
        "bilat_unilateral_pairs": int(bilat_unilateral_pairs),
        "acc": total_acc / total_n,
        "left_acc": (total_left / max(total_left_n, 1)) if total_left_n > 0 else 0.0,
        "right_acc": (total_right / max(total_right_n, 1)) if total_right_n > 0 else 0.0,
        "n": total_n,
    }
    if logits.shape[-1] == 2:
        out.update(
            {
                "pred_pos_rate": float(total_pred_pos / total_n) if total_n > 0 else 0.0,
                "left_pred_pos_rate": float(total_left_pred_pos / total_left_n) if total_left_n > 0 else 0.0,
                "right_pred_pos_rate": float(total_right_pred_pos / total_right_n) if total_right_n > 0 else 0.0,
            }
        )
    if cm is not None:
        out["confusion_matrix"] = cm
    return out


def _resolve_split_paths(splits_root: Path, pct: int) -> tuple[Path, Path]:
    split_dir = splits_root / f"{pct}pct"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    return train_csv, val_csv


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    lr = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr


def _get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return float("nan")


def _compute_epoch_lr(
    epoch: int,
    *,
    base_lr: float,
    min_lr: float,
    schedule: str,
    warmup_epochs: int,
    warmup_ratio: float,
    total_epochs: int,
) -> float:
    schedule = str(schedule or "constant").strip().lower()
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    if not math.isfinite(min_lr) or min_lr < 0:
        min_lr = 0.0
    if min_lr > base_lr:
        min_lr = base_lr

    if schedule in {"none", "constant", "const", "fixed"} or total_epochs <= 1:
        return base_lr
    if schedule not in {"cosine"}:
        raise ValueError(f"unknown lr schedule: {schedule!r}")

    warmup_epochs = max(0, int(warmup_epochs))
    warmup_ratio = float(warmup_ratio)
    warmup_ratio = max(0.0, min(1.0, warmup_ratio))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return base_lr
        start_lr = base_lr * warmup_ratio
        t = float(epoch - 1) / float(warmup_epochs - 1)
        return start_lr + t * (base_lr - start_lr)

    remain = int(total_epochs) - warmup_epochs
    if remain <= 1:
        return min_lr
    t = float(epoch - warmup_epochs - 1) / float(remain - 1)
    t = max(0.0, min(1.0, t))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (base_lr - min_lr) * cos


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="由 scripts/make_splits_dual.py 生成"),
    pct: int = typer.Option(100, help="训练数据比例：1 / 20 / 100"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/<timestamp>）"),
    model: str = typer.Option(
        "dual_resnet10_3d",
        help=(
            "dual_resnet{10,18,34,50,101,152,200}_3d | dual_resnet{...}_3d_xattn | dual_resnet{...}_3d_a3b | dual_resnet{...}_3d_a3b4 | "
            "dual_resnet{...}_3d_proto | dual_resnet{...}_3d_a3b4_proto | "
            "dual_unet_3d | dual_vit_3d | dual_vit_3d_xattn"
        ),
    ),
    label_task: str = typer.Option(
        "six_class",
        help=(
            "标签任务：six_class | normal_vs_diseased | normal_vs_csoma | normal_vs_cholesteatoma | "
            "normal_vs_cholesterol_granuloma | normal_vs_ome | ome_vs_cholesterol_granuloma | cholesteatoma_vs_csoma"
        ),
    ),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(1),
    auto_batch: bool = typer.Option(False, help="自动寻找最大 batch_size 以榨干显存（OOM 探测）"),
    max_batch_size: int = typer.Option(32, help="auto_batch 的上限"),
    num_workers: int = typer.Option(8),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="缓存预处理后的体数据到 cache/，提高吞吐并榨干 GPU"),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes"), help="缓存目录（不入库）"),
    cache_dtype: str = typer.Option("float16", help="缓存数据类型：float16 | float32"),
    vit_patch_size: str = typer.Option("4,16,16", help="仅 dual_vit_3d / dual_vit_3d_xattn 生效，例如 4,16,16"),
    vit_pool: str = typer.Option("cls", help="仅 dual_vit_3d / dual_vit_3d_xattn 生效：token pooling = cls | mean"),
    vit_hidden_size: int = typer.Option(768),
    vit_mlp_dim: int = typer.Option(3072),
    vit_num_layers: int = typer.Option(12),
    vit_num_heads: int = typer.Option(12),
    unet_channels: str = typer.Option("16,32,64,128,256", help="仅 dual_unet_3d 生效"),
    unet_strides: str = typer.Option("2,2,2,2", help="仅 dual_unet_3d 生效"),
    unet_num_res_units: int = typer.Option(2, help="仅 dual_unet_3d 生效"),
    lr: float = typer.Option(1e-4),
    lr_schedule: str = typer.Option("constant", help="学习率策略：constant | cosine"),
    min_lr: float = typer.Option(1e-6, help="cosine 最小 lr（constant 无视）"),
    warmup_epochs: int = typer.Option(0, help="warmup epoch 数（0=关闭）"),
    warmup_ratio: float = typer.Option(0.1, help="warmup 起始 lr = lr * ratio"),
    weight_decay: float = typer.Option(0.05, help="AdamW weight decay（更强正则，默认比 PyTorch 的 0.01 更大）"),
    seed: int = typer.Option(42),
    label_smoothing: float = typer.Option(0.10, help="CrossEntropy label smoothing（更强正则）"),
    grad_clip_norm: float = typer.Option(0.0, help="梯度裁剪 max_norm（0=关闭；建议 1.0 用于更大 lr）"),
    grad_accum_steps: int = typer.Option(1, help="梯度累计步数（>1 可用小 batch + 大等效 batch，改善泛化/稳定）"),
    augment: bool = typer.Option(True, "--augment/--no-augment", help="训练时启用数据增强（不会写入 cache）"),
    aug_flip_prob: float = typer.Option(0.5, help="随机翻转概率（H/W）"),
    aug_intensity_prob: float = typer.Option(0.7, help="随机强度缩放/平移概率"),
    aug_noise_prob: float = typer.Option(0.2, help="随机高斯噪声概率"),
    aug_gamma_prob: float = typer.Option(0.2, help="随机 gamma 概率"),
    amp: bool = typer.Option(True),
    tf32: bool = typer.Option(True, help="CUDA: 允许 TF32 加速 matmul/conv"),
    cudnn_benchmark: bool = typer.Option(True, help="CUDA: cudnn benchmark 以提高吞吐"),
    compile: bool = typer.Option(False, help="PyTorch 2: torch.compile 以提高吞吐"),
    empty_cache: bool = typer.Option(True, "--empty-cache/--no-empty-cache", help="每个 epoch 的 train/val 之间调用 torch.cuda.empty_cache，避免碎片导致 OOM"),
    eval_every: int = typer.Option(1, help="每 N 个 epoch 做一次 val（>=1；总会在最后一个 epoch 做 val）"),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="上传训练指标到 Weights & Biases"),
    wandb_project: str = typer.Option("medical_fenlei"),
    wandb_entity: str | None = typer.Option(None),
    wandb_name: str | None = typer.Option(None),
    wandb_group: str | None = typer.Option(None),
    wandb_tags: str = typer.Option("", help="逗号分隔 tags"),
    wandb_mode: str = typer.Option("online", help="online | offline | disabled"),
    wandb_dir: Path = typer.Option(Path("wandb"), help="wandb 本地目录（不入库）"),
    early_stop_patience: int = typer.Option(0, help="早停 patience（0=关闭）"),
    early_stop_metric: str = typer.Option("val_acc", help="val_acc | val_loss | macro_f1 | macro_recall | macro_specificity | weighted_f1"),
    early_stop_min_delta: float = typer.Option(0.0, help="最小提升幅度（避免抖动）"),
    class_weights: bool = typer.Option(True, "--class-weights/--no-class-weights", help="按耳级别频次自动计算 loss class weight"),
    balanced_sampler: bool = typer.Option(False, "--balanced-sampler/--no-balanced-sampler", help="训练集使用 WeightedRandomSampler 近似平衡采样"),
    balanced_sampler_mode: str = typer.Option("max", help="balanced sampler 权重聚合：mean | max | sum（更推荐 max）"),
    loss: str = typer.Option("ce", help="loss：ce | focal（focal 会忽略 label_smoothing）"),
    focal_gamma: float = typer.Option(2.0, help="focal loss gamma（loss=focal 生效）"),
    init_bias: bool = typer.Option(True, "--init-bias/--no-init-bias", help="binary: 初始化 classifier bias 为训练集先验，避免早期塌缩"),
    proto_reg_lambda: float | None = typer.Option(None, help="*_proto: prototype anchor L2 lambda（None=auto；0=关闭）"),
    proto_init: str = typer.Option(
        "auto",
        help="*_proto: prototype 初始化：auto | prompt_hash（占位） | prompt_hf（HuggingFace/CMBERT） | rand",
    ),
    proto_text_model: str = typer.Option("hfl/chinese-roberta-wwm-ext", help="*_proto: prompt_hf 的 HF 模型 id/路径（可替换为 CMBERT）"),
    cmbert_model: str = typer.Option(
        os.environ.get("MEDICAL_FENLEI_CMBERT_MODEL", ""),
        help="CMBERT HF 模型 id/路径（非空则覆盖 proto_text_model/vl_text_model，并自动切换 vl_text_encoder=hf）",
    ),
    proto_text_pool: str = typer.Option("cls", help="*_proto: prompt_hf pooling：cls | mean"),
    proto_text_max_length: int = typer.Option(256, help="*_proto: prompt_hf tokenizer max_length"),
    proto_text_proj_seed: int = typer.Option(42, help="*_proto: prompt_hf hidden_size->feat_dim 随机投影 seed"),
    proto_freeze_missing: bool = typer.Option(True, "--proto-freeze-missing/--no-proto-freeze-missing", help="*_proto 6-class: 冻结训练集中缺失类别的 prototype（保持文本原型固定）"),
    proto_freeze_ids: str = typer.Option("", help="*_proto: 逗号分隔冻结的 class id（0-based）"),
    vl_contrastive_lambda: float = typer.Option(0.0, help="*_proto: 视觉-文本 InfoNCE loss lambda（0=关闭）"),
    vl_temperature: float = typer.Option(0.07, help="*_proto: InfoNCE temperature（>0）"),
    vl_text_encoder: str = typer.Option("hash", help="*_proto: 视觉-文本对比学习的文本编码：hash（baseline） | hf（HuggingFace/CMBERT）"),
    vl_text_model: str = typer.Option("", help="hf: HF 模型 id/路径（空=使用 --proto-text-model）"),
    vl_text_pool: str = typer.Option("", help="hf: pooling=cls|mean（空=使用 --proto-text-pool）"),
    vl_text_max_length: int = typer.Option(0, help="hf: tokenizer max_length（0=使用 --proto-text-max-length）"),
    vl_text_proj_seed: int = typer.Option(0, help="hf: hidden_size->feat_dim 随机投影 seed（0=使用 --proto-text-proj-seed）"),
    vl_text_device: str = typer.Option("cpu", help="hf: 文本编码 device：cpu | cuda（建议 cpu，避免与训练抢 GPU）"),
    vl_text_batch_size: int = typer.Option(16, help="hf: 文本编码 batch_size（仅影响文本编码吞吐）"),
    vl_text_cache_size: int = typer.Option(10000, help="hf: 文本 embedding cache 大小（0=禁用；<0=不限）"),
    vl_local_lambda: float = typer.Option(0.0, help="*_proto: 基于报告 triplet 的局部对齐 InfoNCE lambda（0=关闭）"),
    vl_local_temperature: float = typer.Option(0.07, help="*_proto: 局部对齐 temperature（>0）"),
    vl_local_max_triplets_per_ear: int = typer.Option(8, help="局部对齐：每个 ear 最多使用多少个 triplet 作为 query"),
    vl_local_drop_negated: bool = typer.Option(True, "--vl-local-drop-negated/--vl-local-keep-negated", help="局部对齐：是否丢弃否定 triplet（如 未见骨质破坏）"),
    bilat_loss_lambda: float = typer.Option(0.0, help="*_proto: 双侧约束 loss lambda（0=关闭）"),
    bilat_unilateral_sim_max: float = typer.Option(0.0, help="*_proto: 单侧病变时左右耳 cosine 相似度上限（越小越推开）"),
    crop_size: int = typer.Option(192, help="每侧颞骨 ROI patch 大小（像素；会再 resize 到 image_size）"),
    window_wl: float = typer.Option(700.0, help="CT 窗位（HU）"),
    window_ww: float = typer.Option(4000.0, help="CT 窗宽（HU）"),
    window2_wl: float = typer.Option(0.0, help="第二个窗位（HU；window2_ww<=0 关闭）"),
    window2_ww: float = typer.Option(0.0, help="第二个窗宽（HU；<=0 关闭）"),
    pair_features: str = typer.Option("none", help="双侧对比特征：none | self_other_diff"),
    sampling: str = typer.Option("air_block", help="z 采样：even | air_block（先定位耳区 block 再采样）"),
    block_len: int = typer.Option(64, help="sampling=air_block 时的 block 长度（切片数）"),
    target_spacing: float = typer.Option(0.7, help="统一 in-plane spacing（mm；<=0 关闭）"),
    target_z_spacing: float = typer.Option(0.8, help="统一 z spacing（mm；<=0 关闭）"),
) -> None:
    train_csv, val_csv = _resolve_split_paths(splits_root, pct)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if train_df.empty or val_df.empty:
        raise typer.Exit(code=2)

    task_spec = resolve_task(label_task)
    class_id_to_name = dict(task_spec.class_id_to_name)
    num_classes = int(task_spec.num_classes)
    task_pos_label_ids: tuple[int, ...] | None = None
    task_neg_label_ids: tuple[int, ...] | None = None

    if task_spec.kind == "binary":
        codes = task_spec.relevant_codes()
        train_df = _filter_df_for_codes(train_df, codes=codes)
        val_df = _filter_df_for_codes(val_df, codes=codes)
        if train_df.empty or val_df.empty:
            typer.echo(f"no data for task={task_spec.name} after filtering by codes={sorted(codes)}")
            raise typer.Exit(code=2)

        task_pos_label_ids = task_spec.pos_label_ids()
        task_neg_label_ids = task_spec.neg_label_ids()
        if set(task_pos_label_ids) & set(task_neg_label_ids):
            raise ValueError(f"binary task pos/neg overlap: pos={task_pos_label_ids} neg={task_neg_label_ids}")

    train_report_nonempty = _count_nonempty_report_text(train_df)
    val_report_nonempty = _count_nonempty_report_text(val_df)

    dicom_root = infer_dicom_root(dicom_base)
    _seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}"
    if not cache:
        used_cache_dir = None

    if str(sampling) not in {"even", "air_block"}:
        raise ValueError(f"unknown sampling: {sampling!r} (expected even|air_block)")

    pair_features_s = str(pair_features).strip().lower() if pair_features is not None else "none"
    if pair_features_s not in {"none", "self_other_diff"}:
        raise ValueError(f"unknown pair_features: {pair_features_s!r} (expected none|self_other_diff)")

    w2_ww = float(window2_ww)
    w2_wl = float(window2_wl)
    window2_wl_v = w2_wl if w2_ww > 0 else None
    window2_ww_v = w2_ww if w2_ww > 0 else None
    base_channels = 2 if window2_ww_v is not None else 1
    pair_factor = 3 if pair_features_s == "self_other_diff" else 1
    in_channels = int(base_channels) * int(pair_factor)
    preprocess_spec = DualPreprocessSpec(
        num_slices=int(num_slices),
        image_size=int(image_size),
        crop_size=int(crop_size),
        window_wl=float(window_wl),
        window_ww=float(window_ww),
        window2_wl=window2_wl_v,
        window2_ww=window2_ww_v,
        pair_features=pair_features_s,
        sampling=str(sampling),
        block_len=int(block_len),
        flip_right=True,
        target_spacing=float(target_spacing) if float(target_spacing) > 0 else None,
        target_z_spacing=float(target_z_spacing) if float(target_z_spacing) > 0 else None,
    )

    train_ds = EarCTDualDataset(
        index_df=train_df,
        dicom_root=dicom_root,
        spec=preprocess_spec,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
    )
    val_ds = EarCTDualDataset(
        index_df=val_df,
        dicom_root=dicom_root,
        spec=preprocess_spec,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
    )

    # `class_weight` can be used by loss and/or sampler (they're controlled separately).
    class_weight = None
    if bool(class_weights) or bool(balanced_sampler):
        class_weight = _make_class_weight(
            train_df=train_df,
            num_classes=int(num_classes),
            task_kind=str(task_spec.kind),
            task_pos_label_ids=task_pos_label_ids,
            task_neg_label_ids=task_neg_label_ids,
        )

    if str(model) in {"dual_vit_3d", "dual_vit_3d_xattn"} and str(task_spec.kind) == "binary" and bool(class_weights) and not bool(balanced_sampler):
        typer.echo("note: dual_vit_3d(binary) may stall with --class-weights; consider --balanced-sampler and --no-class-weights")

    train_sampler = None
    if bool(balanced_sampler):
        weights = _make_exam_sampling_weights(
            train_df=train_df,
            task_kind=str(task_spec.kind),
            task_pos_label_ids=task_pos_label_ids,
            task_neg_label_ids=task_neg_label_ids,
            class_weight=class_weight,
            mode=str(balanced_sampler_mode),
        )
        if weights is not None:
            train_sampler = WeightedRandomSampler(weights=weights, num_samples=int(len(train_ds)), replacement=True)

    def _make_train_loader(bs: int) -> DataLoader:
        use_sampler = train_sampler is not None
        return DataLoader(
            train_ds,
            batch_size=int(bs),
            shuffle=not use_sampler,
            sampler=train_sampler if use_sampler else None,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def _make_val_loader(bs: int) -> DataLoader:
        return DataLoader(
            val_ds,
            batch_size=int(bs),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    train_loader = _make_train_loader(int(batch_size))
    val_loader = _make_val_loader(int(batch_size))

    img_size = (int(num_slices), int(image_size), int(image_size))
    vit_patch = _parse_int_tuple(vit_patch_size, n=3)
    unet_ch = _parse_int_tuple(unet_channels)
    unet_st = _parse_int_tuple(unet_strides)

    model_s = str(model).strip()
    is_proto_model = model_s.lower().endswith("_proto")

    cmbert_s = str(cmbert_model).strip()
    if cmbert_s:
        proto_text_model = str(cmbert_s)
        if not str(vl_text_model).strip():
            vl_text_model = str(cmbert_s)
        if str(vl_text_encoder).strip().lower() in {"", "hash"}:
            vl_text_encoder = "hf"

    proto_prompts_zh = None
    if is_proto_model and str(task_spec.kind) == "binary":
        from medical_fenlei.text_prompts import get_task_prompts_zh

        proto_prompts_zh = get_task_prompts_zh(task_spec)

    proto_init_s = str(proto_init or "").strip().lower()
    if is_proto_model and str(task_spec.kind) == "binary" and proto_init_s in {"", "auto", "default"}:
        # Align with check.md: even for binary tasks, initialize prototypes from text prompts.
        proto_init = "prompt_hf"

    net, spec = make_dual_model(
        model,
        num_classes=num_classes,
        in_channels=int(in_channels),
        img_size=img_size,
        vit_patch_size=vit_patch,
        vit_pool=str(vit_pool),
        vit_hidden_size=vit_hidden_size,
        vit_mlp_dim=vit_mlp_dim,
        vit_num_layers=vit_num_layers,
        vit_num_heads=vit_num_heads,
        unet_channels=unet_ch,
        unet_strides=unet_st,
        unet_num_res_units=unet_num_res_units,
        proto_init=str(proto_init),
        proto_text_model=str(proto_text_model),
        proto_text_pool=str(proto_text_pool),
        proto_text_max_length=int(proto_text_max_length),
        proto_text_proj_seed=int(proto_text_proj_seed),
        proto_prompts_zh=proto_prompts_zh,
    )
    if proto_reg_lambda is None:
        proto_reg_lambda_v = 0.05 if (is_proto_model and int(num_classes) == len(CLASS_ID_TO_NAME)) else 0.0
    else:
        proto_reg_lambda_v = float(proto_reg_lambda)
    if proto_reg_lambda_v < 0:
        raise ValueError(f"proto_reg_lambda must be >= 0, got {proto_reg_lambda_v:g}")

    proto_frozen_ids: list[int] = []
    if is_proto_model:
        freeze_set: set[int] = set(int(x) for x in _parse_int_tuple(proto_freeze_ids))
        if bool(proto_freeze_missing) and int(num_classes) == len(CLASS_ID_TO_NAME):
            ids = _ear_label_ids_from_df(train_df)
            seen = set(int(x) for x in ids.unique().tolist()) if not ids.empty else set()
            missing = {int(i) for i in range(int(num_classes)) if int(i) not in seen}
            freeze_set |= missing
        proto_frozen_ids = sorted(freeze_set)
        if proto_frozen_ids:
            fn = getattr(net, "set_frozen_class_ids", None)
            if callable(fn):
                fn(proto_frozen_ids)
            else:
                typer.echo("warning: proto freezing requested but model has no set_frozen_class_ids(); skip")

    vl_contrastive_lambda_v = float(vl_contrastive_lambda)
    if vl_contrastive_lambda_v < 0:
        raise ValueError(f"vl_contrastive_lambda must be >= 0, got {vl_contrastive_lambda_v:g}")
    vl_temperature_v = float(vl_temperature)
    if not math.isfinite(vl_temperature_v) or vl_temperature_v <= 0:
        raise ValueError(f"vl_temperature must be > 0, got {vl_temperature!r}")

    vl_local_lambda_v = float(vl_local_lambda)
    if vl_local_lambda_v < 0:
        raise ValueError(f"vl_local_lambda must be >= 0, got {vl_local_lambda_v:g}")
    vl_local_temperature_v = float(vl_local_temperature)
    if not math.isfinite(vl_local_temperature_v) or vl_local_temperature_v <= 0:
        raise ValueError(f"vl_local_temperature must be > 0, got {vl_local_temperature!r}")
    vl_local_max_triplets_per_ear_v = max(0, int(vl_local_max_triplets_per_ear))
    vl_local_drop_negated_v = bool(vl_local_drop_negated)

    vl_text_vectorizer = None
    if vl_contrastive_lambda_v > 0 or vl_local_lambda_v > 0:
        if not is_proto_model:
            typer.echo("warning: vl contrastive/local alignment is only supported for *_proto models; disabling")
            vl_contrastive_lambda_v = 0.0
            vl_local_lambda_v = 0.0
        else:
            if vl_contrastive_lambda_v > 0 and int(train_report_nonempty) < 2:
                typer.echo(
                    "warning: vl contrastive enabled but train.csv has no usable report_text "
                    f"(nonempty={int(train_report_nonempty)}; need >=2); disabling"
                )
                vl_contrastive_lambda_v = 0.0
            if vl_local_lambda_v > 0 and int(train_report_nonempty) < 1:
                typer.echo(
                    "warning: vl local alignment enabled but train.csv has no usable report_text "
                    f"(nonempty={int(train_report_nonempty)}; need >=1); disabling"
                )
                vl_local_lambda_v = 0.0

            if vl_local_lambda_v > 0:
                fwm = getattr(net, "forward_with_features_and_maps", None)
                if fwm is None and hasattr(net, "_orig_mod"):
                    fwm = getattr(net._orig_mod, "forward_with_features_and_maps", None)
                if not callable(fwm):
                    typer.echo("warning: vl local alignment requires model.forward_with_features_and_maps(); disabling")
                    vl_local_lambda_v = 0.0

            if vl_contrastive_lambda_v > 0 or vl_local_lambda_v > 0:
                feat_dim = int(getattr(net, "feat_dim", 0) or int(spec.kwargs.get("feat_dim", 0) or 0))
                if feat_dim <= 0:
                    raise ValueError("vl contrastive/local: cannot infer feat_dim for text encoding")
                enc = str(vl_text_encoder).strip().lower()
                if enc in {"", "hash"}:
                    try:
                        from sklearn.feature_extraction.text import HashingVectorizer
                    except Exception as e:
                        raise RuntimeError("vl(text_encoder=hash) requires scikit-learn (HashingVectorizer)") from e

                    vl_text_vectorizer = HashingVectorizer(
                        n_features=int(feat_dim),
                        analyzer="char",
                        ngram_range=(1, 2),
                        lowercase=False,
                        alternate_sign=False,
                        norm=None,
                    )
                elif enc == "hf":
                    from medical_fenlei.text_encoder import HFTextVectorizer

                    model_name = str(vl_text_model).strip() or str(proto_text_model)
                    pool = str(vl_text_pool).strip() or str(proto_text_pool)
                    max_len = int(vl_text_max_length) if int(vl_text_max_length) > 0 else int(proto_text_max_length)
                    proj_seed = int(vl_text_proj_seed) if int(vl_text_proj_seed) > 0 else int(proto_text_proj_seed)

                    device_s = str(vl_text_device).strip().lower()
                    if device_s == "cuda" and not torch.cuda.is_available():
                        typer.echo("warning: vl_text_device=cuda requested but CUDA is unavailable; fallback to cpu")
                        device_s = "cpu"

                    vl_text_vectorizer = HFTextVectorizer(
                        model_name_or_path=model_name,
                        out_dim=int(feat_dim),
                        pool=str(pool),
                        max_length=int(max_len),
                        batch_size=int(vl_text_batch_size),
                        device=str(device_s),
                        proj_seed=int(proj_seed),
                        cache_size=int(vl_text_cache_size),
                    )
                else:
                    raise ValueError(f"unknown vl_text_encoder: {vl_text_encoder!r} (expected hash|hf)")

    bilat_loss_lambda_v = float(bilat_loss_lambda)
    if bilat_loss_lambda_v < 0:
        raise ValueError(f"bilat_loss_lambda must be >= 0, got {bilat_loss_lambda_v:g}")
    bilat_unilateral_sim_max_v = float(bilat_unilateral_sim_max)
    if not math.isfinite(bilat_unilateral_sim_max_v):
        raise ValueError(f"bilat_unilateral_sim_max must be finite, got {bilat_unilateral_sim_max!r}")

    bilat_normal_class_id_v: int | None = None
    if bilat_loss_lambda_v > 0:
        if not is_proto_model:
            typer.echo("warning: bilat loss is only supported for *_proto models; disabling")
            bilat_loss_lambda_v = 0.0
        elif int(num_classes) != len(CLASS_ID_TO_NAME):
            typer.echo("warning: bilat loss currently expects six_class (num_classes=6); disabling")
            bilat_loss_lambda_v = 0.0
        else:
            # CLASS_ID_TO_NAME defines "正常" as class id 4.
            bilat_normal_class_id_v = 4

    net = net.to(device)
    if compile and hasattr(torch, "compile"):
        net = torch.compile(net)

    if bool(init_bias) and str(task_spec.kind) == "binary" and int(num_classes) == 2:
        ids = _ear_label_ids_from_df(train_df)
        pos_set = set(int(x) for x in (task_pos_label_ids or ()))
        neg_set = set(int(x) for x in (task_neg_label_ids or ()))
        pos = int(ids.isin(list(pos_set)).sum()) if not ids.empty else 0
        neg = int(ids.isin(list(neg_set)).sum()) if not ids.empty else 0
        total = int(pos + neg)
        if total > 0:
            priors = [float(neg) / float(total), float(pos) / float(total)]
            _init_classifier_bias_from_prior(net, num_classes=int(num_classes), priors=priors)

    if auto_batch:
        batch_size = _autotune_batch_size(
            model=net,
            device=device,
            num_classes=num_classes,
            in_channels=int(in_channels),
            num_slices=num_slices,
            image_size=image_size,
            amp=amp,
            max_batch_size=max_batch_size,
        )

        # Rebuild loaders with the chosen batch size.
        train_loader = _make_train_loader(int(batch_size))
        val_loader = _make_val_loader(int(batch_size))

    base_lr = float(lr)
    optimizer = _make_adamw(net, lr=base_lr, weight_decay=float(weight_decay))

    loss_name = str(loss or "ce").strip().lower()
    loss_weight = class_weight if bool(class_weights) else None
    if loss_name in {"ce", "cross_entropy", "crossentropy"}:
        try:
            loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=float(label_smoothing))
        except TypeError:
            # Older torch may not support label_smoothing.
            loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight) if loss_weight is not None else torch.nn.CrossEntropyLoss()
    elif loss_name in {"focal", "focal_loss"}:
        if float(label_smoothing) > 0:
            typer.echo("loss=focal: label_smoothing is ignored (set to 0 internally)")
        alpha = loss_weight if loss_weight is not None else None
        loss_fn = FocalLoss(gamma=float(focal_gamma), alpha=alpha)
    else:
        raise ValueError(f"unknown loss: {loss!r} (expected ce|focal)")

    loss_fn = loss_fn.to(device)

    amp_enabled = amp and device.type == "cuda"
    if not amp_enabled:
        scaler = None
    elif hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_\\-]+", "_", model)
    safe_task = re.sub(r"[^A-Za-z0-9_\\-]+", "_", str(task_spec.name))
    out = output_dir or Path("outputs") / f"{safe_model}__{safe_task}_{pct}pct_{ts}"
    ckpt_dir = out / "checkpoints"
    report_dir = out / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"

    if task_spec.kind == "binary":
        train_pos = _count_codes(train_df, codes=set(task_spec.pos_codes))
        train_neg = _count_codes(train_df, codes=set(task_spec.neg_codes))
        val_pos = _count_codes(val_df, codes=set(task_spec.pos_codes))
        val_neg = _count_codes(val_df, codes=set(task_spec.neg_codes))
        typer.echo(
            f"task: 一次检查 -> 左/右双输出 二分类({task_spec.name})  classes={num_classes}  "
            f"train(pos={train_pos},neg={train_neg}) val(pos={val_pos},neg={val_neg})"
        )
    else:
        typer.echo(f"task: 一次检查 -> 左/右双输出 6 分类  classes={num_classes}")
    typer.echo(
        f"model: {model}  pct={pct}%  batch_size={batch_size}  amp={amp}  "
        f"wd={float(weight_decay)}  ls={float(label_smoothing)}  augment={bool(augment)}"
    )
    typer.echo(
        f"loss: {str(loss_name)} focal_gamma={float(focal_gamma):g} class_weights={bool(class_weights)} "
        f"balanced_sampler={bool(balanced_sampler)} sampler_mode={str(balanced_sampler_mode)} init_bias={bool(init_bias)}"
    )
    if is_proto_model:
        vl_text_desc = str(vl_text_encoder).strip().lower()
        if vl_text_desc == "hf":
            model_name = str(vl_text_model).strip() or str(proto_text_model)
            pool = str(vl_text_pool).strip() or str(proto_text_pool)
            max_len = int(vl_text_max_length) if int(vl_text_max_length) > 0 else int(proto_text_max_length)
            vl_text_desc = f"hf({model_name},pool={pool},max_len={max_len})"
        typer.echo(
            f"proto: reg_lambda={float(proto_reg_lambda_v):g} freeze_missing={bool(proto_freeze_missing)} frozen_ids={proto_frozen_ids} "
            f"vl_lambda={float(vl_contrastive_lambda_v):g} vl_temp={float(vl_temperature_v):g} "
            f"vl_local_lambda={float(vl_local_lambda_v):g} vl_local_temp={float(vl_local_temperature_v):g} "
            f"vl_local_maxT={int(vl_local_max_triplets_per_ear_v)} vl_local_drop_neg={int(bool(vl_local_drop_negated_v))} "
            f"vl_text={vl_text_desc} "
            f"bilat_lambda={float(bilat_loss_lambda_v):g} bilat_unilateral_sim_max={float(bilat_unilateral_sim_max_v):g}"
        )
    typer.echo(
        f"lr: base={base_lr:g} schedule={str(lr_schedule)} warmup_epochs={int(warmup_epochs)} "
        f"warmup_ratio={float(warmup_ratio):g} min_lr={float(min_lr):g} grad_clip_norm={float(grad_clip_norm):g} "
        f"grad_accum_steps={int(grad_accum_steps)} eval_every={int(eval_every)}"
    )
    typer.echo(f"dicom_root: {dicom_root}")
    if used_cache_dir is not None:
        typer.echo(f"cache_dir: {used_cache_dir} ({cache_dtype})")
    typer.echo(
        f"preprocess: crop_size={int(preprocess_spec.crop_size)} sampling={str(preprocess_spec.sampling)} "
        f"block_len={int(preprocess_spec.block_len)} target_spacing={float(preprocess_spec.target_spacing or 0.0):g} "
        f"target_z_spacing={float(preprocess_spec.target_z_spacing or 0.0):g} "
        f"wl={float(preprocess_spec.window_wl):g} ww={float(preprocess_spec.window_ww):g} "
        f"wl2={float(preprocess_spec.window2_wl or 0.0):g} ww2={float(preprocess_spec.window2_ww or 0.0):g} "
        f"pair={str(preprocess_spec.pair_features)} in_channels={int(in_channels)}"
    )
    if class_weight is not None:
        try:
            typer.echo(f"class_weight: {[float(x) for x in class_weight.detach().cpu().tolist()]}")
        except Exception:
            pass
    if int(train_report_nonempty) > 0 or int(val_report_nonempty) > 0:
        typer.echo(f"report_text: train_nonempty={int(train_report_nonempty)} val_nonempty={int(val_report_nonempty)}")
    if train_sampler is not None:
        typer.echo("train_sampler: WeightedRandomSampler (balanced)")
    typer.echo(f"train_exams: {len(train_ds)}  val_exams: {len(val_ds)}")
    typer.echo(f"output: {out}")

    wb = None
    wb_run = None
    if bool(wandb) and str(wandb_mode).lower() != "disabled":
        try:
            import wandb as _wandb

            wb = _wandb
            if str(wandb_mode).lower() == "online" and not os.environ.get("WANDB_API_KEY"):
                typer.echo("wandb: WANDB_API_KEY 未设置；请先 export WANDB_API_KEY=...（或设置 WANDB_MODE=offline）")

            name = str(wandb_name) if wandb_name else out.name
            tags = [t.strip() for t in str(wandb_tags).split(",") if t.strip()]
            wb_run = wb.init(
                project=str(wandb_project),
                entity=str(wandb_entity) if wandb_entity else None,
                name=name,
                group=str(wandb_group) if wandb_group else None,
                tags=tags or None,
                dir=str(wandb_dir),
                mode=str(wandb_mode).lower(),
                config={
                    "task": {"name": str(task_spec.name), "kind": str(task_spec.kind), "num_classes": int(num_classes)},
                    "data": {
                        "splits_root": str(splits_root),
                        "pct": int(pct),
                        "dicom_root": str(dicom_root),
                        "cache_dir": str(used_cache_dir) if used_cache_dir is not None else None,
                        "cache_dtype": str(cache_dtype),
                        "num_slices": int(num_slices),
                        "image_size": int(image_size),
                        "crop_size": int(preprocess_spec.crop_size),
                        "sampling": str(preprocess_spec.sampling),
                        "block_len": int(preprocess_spec.block_len),
                        "target_spacing": float(preprocess_spec.target_spacing) if preprocess_spec.target_spacing is not None else 0.0,
                        "target_z_spacing": float(preprocess_spec.target_z_spacing) if preprocess_spec.target_z_spacing is not None else 0.0,
                    },
                    "model": {"name": spec.name, "kwargs": spec.kwargs},
                    "train": {
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "grad_accum_steps": int(grad_accum_steps),
                        "amp": bool(amp),
                        "augment": bool(augment),
                        "class_weights": bool(class_weights),
                        "balanced_sampler": bool(balanced_sampler),
                        "balanced_sampler_mode": str(balanced_sampler_mode),
                        "loss": str(loss_name),
                        "focal_gamma": float(focal_gamma),
                        "init_bias": bool(init_bias),
                        "proto_reg_lambda": float(proto_reg_lambda_v),
                        "proto_freeze_missing": bool(proto_freeze_missing),
                        "proto_frozen_ids": [int(x) for x in proto_frozen_ids],
                        "vl_contrastive_lambda": float(vl_contrastive_lambda_v),
                        "vl_temperature": float(vl_temperature_v),
                        "bilat_loss_lambda": float(bilat_loss_lambda_v),
                        "bilat_unilateral_sim_max": float(bilat_unilateral_sim_max_v),
                        "lr": float(base_lr),
                        "lr_schedule": str(lr_schedule),
                        "min_lr": float(min_lr),
                        "warmup_epochs": int(warmup_epochs),
                        "warmup_ratio": float(warmup_ratio),
                        "grad_clip_norm": float(grad_clip_norm),
                        "weight_decay": float(weight_decay),
                        "label_smoothing": float(label_smoothing),
                        "eval_every": int(eval_every),
                    },
                },
            )
        except Exception as e:
            typer.echo(f"wandb: init failed ({type(e).__name__}: {e}); continue without wandb")
            wb = None
            wb_run = None

    metric_mode = "min" if early_stop_metric == "val_loss" else "max"
    best_score = float("inf") if metric_mode == "min" else float("-inf")
    bad_epochs = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    val_amp = bool(amp)
    if device.type == "cuda" and bool(amp) and str(model) in {"dual_vit_3d", "dual_vit_3d_xattn"}:
        # ViT3D can output near-equal logits early (or when imbalance handling is strong).
        # Under fp16 autocast, tiny logit deltas may quantize to ties and make val_acc look \"stuck\".
        val_amp = False
        typer.echo("note: dual_vit_3d*: disable AMP for validation to avoid fp16 tie artifacts")

    def _rebuild_loaders(bs: int) -> None:
        nonlocal train_loader, val_loader
        train_loader = _make_train_loader(int(bs))
        val_loader = _make_val_loader(int(bs))

    for epoch in range(1, epochs + 1):
        epoch_lr = _compute_epoch_lr(
            epoch,
            base_lr=base_lr,
            min_lr=float(min_lr),
            schedule=str(lr_schedule),
            warmup_epochs=int(warmup_epochs),
            warmup_ratio=float(warmup_ratio),
            total_epochs=int(epochs),
        )
        _set_optimizer_lr(optimizer, epoch_lr)

        oom_retries = 0
        while True:
            try:
                train_m = _run_epoch(
                    model=net,
                    loader=train_loader,
                    device=device,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scaler=scaler,
                    amp=amp,
                    grad_accum_steps=int(grad_accum_steps),
                    grad_clip_norm=float(grad_clip_norm),
                    proto_reg_lambda=float(proto_reg_lambda_v),
                    vl_contrastive_lambda=float(vl_contrastive_lambda_v),
                    vl_temperature=float(vl_temperature_v),
                    vl_text_vectorizer=vl_text_vectorizer,
                    vl_local_lambda=float(vl_local_lambda_v),
                    vl_local_temperature=float(vl_local_temperature_v),
                    vl_local_max_triplets_per_ear=int(vl_local_max_triplets_per_ear_v),
                    vl_local_drop_negated=bool(vl_local_drop_negated_v),
                    bilat_loss_lambda=float(bilat_loss_lambda_v),
                    bilat_unilateral_sim_max=float(bilat_unilateral_sim_max_v),
                    bilat_normal_class_id=bilat_normal_class_id_v,
                    augment=bool(augment),
                    aug_flip_prob=float(aug_flip_prob),
                    aug_intensity_prob=float(aug_intensity_prob),
                    aug_noise_prob=float(aug_noise_prob),
                    aug_gamma_prob=float(aug_gamma_prob),
                    task_pos_label_ids=task_pos_label_ids,
                    task_neg_label_ids=task_neg_label_ids,
                )
                break
            except RuntimeError as e:
                if device.type == "cuda" and _is_cuda_oom(e) and int(batch_size) > 1 and oom_retries < 16:
                    oom_retries += 1
                    old_bs = int(batch_size)
                    batch_size = old_bs - 1
                    typer.echo(f"OOM(train): epoch={epoch} batch_size {old_bs} -> {int(batch_size)} (retry {oom_retries}/16)")
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    _rebuild_loaders(int(batch_size))
                    continue
                raise

        if empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()

        do_eval = int(eval_every) <= 1 or (epoch % int(eval_every) == 0) or (epoch == int(epochs))
        val_m: dict
        report: dict | None = None
        score: float | None = None

        if do_eval:
            oom_retries = 0
            while True:
                try:
                    val_m = _run_epoch(
                        model=net,
                        loader=val_loader,
                        device=device,
                        loss_fn=loss_fn,
                        optimizer=None,
                        scaler=None,
                        amp=val_amp,
                        grad_accum_steps=1,
                        augment=False,
                        task_pos_label_ids=task_pos_label_ids,
                        task_neg_label_ids=task_neg_label_ids,
                        num_classes=num_classes,
                        collect_cm=True,
                    )
                    break
                except RuntimeError as e:
                    if device.type == "cuda" and _is_cuda_oom(e) and int(batch_size) > 1 and oom_retries < 16:
                        oom_retries += 1
                        old_bs = int(batch_size)
                        batch_size = old_bs - 1
                        typer.echo(f"OOM(val): epoch={epoch} batch_size {old_bs} -> {int(batch_size)} (retry {oom_retries}/16)")
                        optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        _rebuild_loaders(int(batch_size))
                        continue
                    raise

            cm = val_m.pop("confusion_matrix")
            report = classification_report_from_confusion(cm, class_id_to_name=class_id_to_name)
            report_path = report_dir / f"epoch_{epoch}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            monitor_map = {
                "val_loss": float(val_m["loss"]),
                "val_acc": float(val_m["acc"]),
                "macro_recall": float(report["macro_recall"]),
                "macro_specificity": float(report["macro_specificity"]),
                "macro_f1": float(report["macro_f1"]),
                "weighted_f1": float(report["weighted_f1"]),
            }
            if early_stop_metric not in monitor_map:
                raise ValueError(f"unknown early_stop_metric: {early_stop_metric}")
            score = float(monitor_map[early_stop_metric])
        else:
            val_m = {"loss": None, "acc": None, "left_acc": None, "right_acc": None, "n": 0}

        rec = {
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
            "val_metrics": (
                {k: report[k] for k in ("accuracy", "macro_recall", "macro_specificity", "macro_f1", "weighted_f1", "total")} if report else None
            ),
            "model": {"name": spec.name, "kwargs": spec.kwargs},
            "hparams": {
                "label_task": str(task_spec.name),
                "lr": float(base_lr),
                "lr_schedule": str(lr_schedule),
                "min_lr": float(min_lr),
                "warmup_epochs": int(warmup_epochs),
                "warmup_ratio": float(warmup_ratio),
                "grad_clip_norm": float(grad_clip_norm),
                "grad_accum_steps": int(grad_accum_steps),
                "weight_decay": float(weight_decay),
                "label_smoothing": float(label_smoothing),
                "augment": bool(augment),
                "aug_flip_prob": float(aug_flip_prob),
                "aug_intensity_prob": float(aug_intensity_prob),
                "aug_noise_prob": float(aug_noise_prob),
                "aug_gamma_prob": float(aug_gamma_prob),
                "class_weights": bool(class_weights),
                "balanced_sampler": bool(balanced_sampler),
                "balanced_sampler_mode": str(balanced_sampler_mode),
                "loss": str(loss_name),
                "focal_gamma": float(focal_gamma),
                "init_bias": bool(init_bias),
                "proto_reg_lambda": float(proto_reg_lambda_v),
                "proto_freeze_missing": bool(proto_freeze_missing),
                "proto_frozen_ids": [int(x) for x in proto_frozen_ids],
                "vl_contrastive_lambda": float(vl_contrastive_lambda_v),
                "vl_temperature": float(vl_temperature_v),
                "vl_text_encoder": str(vl_text_encoder),
                "vl_text_model": (str(vl_text_model).strip() or str(proto_text_model)),
                "vl_local_lambda": float(vl_local_lambda_v),
                "vl_local_temperature": float(vl_local_temperature_v),
                "vl_local_max_triplets_per_ear": int(vl_local_max_triplets_per_ear_v),
                "vl_local_drop_negated": bool(vl_local_drop_negated_v),
                "bilat_loss_lambda": float(bilat_loss_lambda_v),
                "bilat_unilateral_sim_max": float(bilat_unilateral_sim_max_v),
                "crop_size": int(preprocess_spec.crop_size),
                "sampling": str(preprocess_spec.sampling),
                "block_len": int(preprocess_spec.block_len),
                "target_spacing": float(preprocess_spec.target_spacing) if preprocess_spec.target_spacing is not None else 0.0,
                "target_z_spacing": float(preprocess_spec.target_z_spacing) if preprocess_spec.target_z_spacing is not None else 0.0,
            },
            "lr_current": float(_get_optimizer_lr(optimizer)),
            "batch_size": int(batch_size),
            "early_stop": {
                "metric": early_stop_metric,
                "mode": metric_mode,
                "score": score,
                "patience": int(early_stop_patience),
                "min_delta": float(early_stop_min_delta),
            },
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if wb_run is not None:
            try:
                wandb_metrics = {
                    "epoch": int(epoch),
                    "lr": float(_get_optimizer_lr(optimizer)),
                    "batch_size": int(batch_size),
                    "train/loss": float(train_m["loss"]),
                    "train/proto_reg": float(train_m.get("proto_reg", 0.0)),
                    "train/vl_loss": float(train_m.get("vl_loss", 0.0)),
                    "train/vl_pairs": int(train_m.get("vl_pairs", 0)),
                    "train/vl_local_loss": float(train_m.get("vl_local_loss", 0.0)),
                    "train/vl_local_queries": int(train_m.get("vl_local_queries", 0)),
                    "train/bilat_loss": float(train_m.get("bilat_loss", 0.0)),
                    "train/bilat_normal_pairs": int(train_m.get("bilat_normal_pairs", 0)),
                    "train/bilat_unilateral_pairs": int(train_m.get("bilat_unilateral_pairs", 0)),
                    "train/acc": float(train_m["acc"]),
                    "train/left_acc": float(train_m["left_acc"]),
                    "train/right_acc": float(train_m["right_acc"]),
                    "train/pred_pos_rate": float(train_m.get("pred_pos_rate")) if train_m.get("pred_pos_rate") is not None else None,
                    "train/left_pred_pos_rate": float(train_m.get("left_pred_pos_rate")) if train_m.get("left_pred_pos_rate") is not None else None,
                    "train/right_pred_pos_rate": float(train_m.get("right_pred_pos_rate")) if train_m.get("right_pred_pos_rate") is not None else None,
                    "val/do_eval": bool(do_eval),
                    "val/loss": float(val_m["loss"]) if do_eval else None,
                    "val/proto_reg": float(val_m.get("proto_reg", 0.0)) if do_eval else None,
                    "val/vl_loss": float(val_m.get("vl_loss", 0.0)) if do_eval else None,
                    "val/vl_pairs": int(val_m.get("vl_pairs", 0)) if do_eval else None,
                    "val/vl_local_loss": float(val_m.get("vl_local_loss", 0.0)) if do_eval else None,
                    "val/vl_local_queries": int(val_m.get("vl_local_queries", 0)) if do_eval else None,
                    "val/bilat_loss": float(val_m.get("bilat_loss", 0.0)) if do_eval else None,
                    "val/bilat_normal_pairs": int(val_m.get("bilat_normal_pairs", 0)) if do_eval else None,
                    "val/bilat_unilateral_pairs": int(val_m.get("bilat_unilateral_pairs", 0)) if do_eval else None,
                    "val/acc": float(val_m["acc"]) if do_eval else None,
                    "val/left_acc": float(val_m["left_acc"]) if do_eval else None,
                    "val/right_acc": float(val_m["right_acc"]) if do_eval else None,
                    "val/pred_pos_rate": float(val_m.get("pred_pos_rate")) if do_eval and val_m.get("pred_pos_rate") is not None else None,
                    "val/left_pred_pos_rate": float(val_m.get("left_pred_pos_rate")) if do_eval and val_m.get("left_pred_pos_rate") is not None else None,
                    "val/right_pred_pos_rate": float(val_m.get("right_pred_pos_rate")) if do_eval and val_m.get("right_pred_pos_rate") is not None else None,
                    "val/macro_recall": float(report["macro_recall"]) if report is not None else None,
                    "val/macro_specificity": float(report["macro_specificity"]) if report is not None else None,
                    "val/macro_f1": float(report["macro_f1"]) if report is not None else None,
                    "val/weighted_f1": float(report["weighted_f1"]) if report is not None else None,
                    "val/accuracy": float(report["accuracy"]) if report is not None else None,
                }
                wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
                wb_run.log(wandb_metrics, step=int(epoch))
            except Exception:
                pass

        if report is not None:
            pred_pos_str = ""
            if train_m.get("pred_pos_rate") is not None and val_m.get("pred_pos_rate") is not None:
                pred_pos_str = f"  pred_pos={float(train_m['pred_pos_rate']):.3f}/{float(val_m['pred_pos_rate']):.3f}"
            typer.echo(
                "epoch "
                f"{epoch}: lr={_get_optimizer_lr(optimizer):g} train_loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} "
                f"(L={train_m['left_acc']:.3f} R={train_m['right_acc']:.3f})  "
                f"val_loss={float(val_m['loss']):.4f} acc={float(val_m['acc']):.3f} "
                f"(L={float(val_m['left_acc']):.3f} R={float(val_m['right_acc']):.3f})  "
                f"macro_recall={float(report['macro_recall']):.3f} macro_f1={float(report['macro_f1']):.3f}"
                f"{pred_pos_str}"
            )
        else:
            typer.echo(
                "epoch "
                f"{epoch}: lr={_get_optimizer_lr(optimizer):g} train_loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} "
                f"(L={train_m['left_acc']:.3f} R={train_m['right_acc']:.3f})  val=SKIP"
            )

        ckpt = {
            "epoch": epoch,
            "task": "dual",
            "label_task": str(task_spec.name),
            "task_kind": str(task_spec.kind),
            "class_id_to_name": {int(k): str(v) for k, v in class_id_to_name.items()},
            "pos_codes": tuple(int(x) for x in getattr(task_spec, "pos_codes", ()) or ()),
            "neg_codes": tuple(int(x) for x in getattr(task_spec, "neg_codes", ()) or ()),
            "model_name": spec.name,
            "model_kwargs": spec.kwargs,
            "num_classes": num_classes,
            "num_slices": num_slices,
            "image_size": image_size,
            "batch_size": int(batch_size),
            "lr": float(base_lr),
            "lr_schedule": str(lr_schedule),
            "min_lr": float(min_lr),
            "warmup_epochs": int(warmup_epochs),
            "warmup_ratio": float(warmup_ratio),
            "grad_clip_norm": float(grad_clip_norm),
            "grad_accum_steps": int(grad_accum_steps),
            "weight_decay": float(weight_decay),
            "label_smoothing": float(label_smoothing),
            "loss": str(loss_name),
            "focal_gamma": float(focal_gamma),
            "augment": bool(augment),
            "aug_flip_prob": float(aug_flip_prob),
            "aug_intensity_prob": float(aug_intensity_prob),
            "aug_noise_prob": float(aug_noise_prob),
            "aug_gamma_prob": float(aug_gamma_prob),
            "class_weights": bool(class_weights),
            "balanced_sampler": bool(balanced_sampler),
            "balanced_sampler_mode": str(balanced_sampler_mode),
            "init_bias": bool(init_bias),
            "proto_reg_lambda": float(proto_reg_lambda_v),
            "proto_freeze_missing": bool(proto_freeze_missing),
            "proto_frozen_ids": [int(x) for x in proto_frozen_ids],
            "vl_contrastive_lambda": float(vl_contrastive_lambda_v),
            "vl_temperature": float(vl_temperature_v),
            "vl_text_encoder": str(vl_text_encoder),
            "vl_text_model": (str(vl_text_model).strip() or str(proto_text_model)),
            "vl_local_lambda": float(vl_local_lambda_v),
            "vl_local_temperature": float(vl_local_temperature_v),
            "vl_local_max_triplets_per_ear": int(vl_local_max_triplets_per_ear_v),
            "vl_local_drop_negated": bool(vl_local_drop_negated_v),
            "bilat_loss_lambda": float(bilat_loss_lambda_v),
            "bilat_unilateral_sim_max": float(bilat_unilateral_sim_max_v),
            "preprocess": {
                "crop_size": int(preprocess_spec.crop_size),
                "sampling": str(preprocess_spec.sampling),
                "block_len": int(preprocess_spec.block_len),
                "target_spacing": float(preprocess_spec.target_spacing) if preprocess_spec.target_spacing is not None else 0.0,
                "target_z_spacing": float(preprocess_spec.target_z_spacing) if preprocess_spec.target_z_spacing is not None else 0.0,
                "window_wl": float(preprocess_spec.window_wl),
                "window_ww": float(preprocess_spec.window_ww),
                "window2_wl": float(preprocess_spec.window2_wl) if preprocess_spec.window2_wl is not None else 0.0,
                "window2_ww": float(preprocess_spec.window2_ww) if preprocess_spec.window2_ww is not None else 0.0,
                "pair_features": str(preprocess_spec.pair_features),
                "version": str(preprocess_spec.version),
            },
            "in_channels": int(in_channels),
            "state_dict": net.state_dict(),
            "val_loss": val_m["loss"],
            "early_stop_metric": early_stop_metric,
            "early_stop_score": score,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        improved = False
        if score is not None:
            if metric_mode == "min":
                improved = (best_score - score) > float(early_stop_min_delta)
            else:
                improved = (score - best_score) > float(early_stop_min_delta)

        if improved:
            best_score = score
            bad_epochs = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
        elif score is not None:
            bad_epochs += 1

        if early_stop_patience > 0 and bad_epochs >= int(early_stop_patience):
            typer.echo(f"early stopping: no improvement on {early_stop_metric} for {early_stop_patience} epochs")
            break

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    app()
