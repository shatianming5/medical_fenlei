from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, WeightedRandomSampler

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.data.ear_dataset import EarCTHUEarDataset, EarPreprocessSpec
from medical_fenlei.metrics import binary_metrics
from medical_fenlei.models.slice_attention_resnet import SliceAttentionResNet
from medical_fenlei.models.slice_attention_unet import SliceAttentionUNet
from medical_fenlei.models.slice_attention_vit import SliceAttentionViT
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _git_sha() -> str | None:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def _window_hu_to_unit(x: torch.Tensor, *, wl: float, ww: float) -> torch.Tensor:
    # x: HU
    lower = float(wl) - float(ww) / 2.0
    upper = float(wl) + float(ww) / 2.0
    x = x.clamp(min=lower, max=upper)
    return (x - lower) / (upper - lower + 1e-6)


def _rand_uniform(shape: tuple[int, ...], *, device: torch.device, low: float, high: float, dtype: torch.dtype) -> torch.Tensor:
    if high < low:
        low, high = high, low
    return (low + (high - low) * torch.rand(shape, device=device, dtype=dtype)).to(dtype)


def _augment_unit_volume(
    x: torch.Tensor,
    *,
    flip_prob: float,
    intensity_prob: float,
    noise_prob: float,
    gamma_prob: float,
    affine_prob: float,
    affine_degrees: float,
    affine_translate: float,
    affine_scale_low: float,
    affine_scale_high: float,
) -> torch.Tensor:
    # x: (B,K,1,H,W) in [0,1]
    if x.ndim != 5:
        return x
    if max(flip_prob, intensity_prob, noise_prob, gamma_prob, affine_prob) <= 0:
        return x

    b, k = int(x.shape[0]), int(x.shape[1])

    orig_dtype = x.dtype
    x2 = x
    if orig_dtype in (torch.float16, torch.bfloat16):
        x2 = x2.float()

    device = x2.device

    # Random affine per ear-volume (apply same transform to all slices).
    affine_prob = float(affine_prob)
    if affine_prob > 0:
        do_a = torch.rand((b,), device=device) < float(affine_prob)
        if do_a.any():
            theta = torch.zeros((b, 2, 3), device=device, dtype=x2.dtype)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0

            idx = do_a.nonzero(as_tuple=False).view(-1)
            n_a = int(idx.numel())
            deg = float(affine_degrees)
            ang = _rand_uniform((n_a,), device=device, low=-deg, high=deg, dtype=x2.dtype) * (math.pi / 180.0)
            s = _rand_uniform(
                (n_a,),
                device=device,
                low=float(affine_scale_low),
                high=float(affine_scale_high),
                dtype=x2.dtype,
            )
            tmax = float(max(0.0, float(affine_translate))) * 2.0  # normalized [-1..1] coords
            tx = _rand_uniform((n_a,), device=device, low=-tmax, high=tmax, dtype=x2.dtype)
            ty = _rand_uniform((n_a,), device=device, low=-tmax, high=tmax, dtype=x2.dtype)

            ca = torch.cos(ang) * s
            sa = torch.sin(ang) * s
            theta[idx, 0, 0] = ca
            theta[idx, 0, 1] = -sa
            theta[idx, 0, 2] = tx
            theta[idx, 1, 0] = sa
            theta[idx, 1, 1] = ca
            theta[idx, 1, 2] = ty

            x_flat = x2.reshape(b * k, *x2.shape[2:]).contiguous()
            theta_rep = theta.repeat_interleave(k, dim=0)
            grid = F.affine_grid(theta_rep, size=x_flat.shape, align_corners=False)
            x_flat = F.grid_sample(x_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            x2 = x_flat.reshape(b, k, *x2.shape[2:]).contiguous()

    # Random flips per ear-volume on H/W.
    flip_prob = float(flip_prob)
    if flip_prob > 0:
        do_h = torch.rand((b,), device=device) < float(flip_prob)
        do_w = torch.rand((b,), device=device) < float(flip_prob)
        if do_h.any():
            x2[do_h] = x2[do_h].flip(-2)
        if do_w.any():
            x2[do_w] = x2[do_w].flip(-1)

    # Random intensity scale/shift per ear-volume.
    intensity_prob = float(intensity_prob)
    if intensity_prob > 0:
        do_i = torch.rand((b,), device=device) < float(intensity_prob)
        if do_i.any():
            n_i = int(do_i.sum().item())
            scale = _rand_uniform((n_i, 1, 1, 1, 1), device=device, low=0.85, high=1.15, dtype=x2.dtype)
            shift = _rand_uniform((n_i, 1, 1, 1, 1), device=device, low=-0.15, high=0.15, dtype=x2.dtype)
            x2_do = x2[do_i]
            x2[do_i] = x2_do * scale + shift

    # Random gamma per ear-volume.
    gamma_prob = float(gamma_prob)
    if gamma_prob > 0:
        do_g = torch.rand((b,), device=device) < float(gamma_prob)
        if do_g.any():
            n_g = int(do_g.sum().item())
            gamma = _rand_uniform((n_g, 1, 1, 1, 1), device=device, low=0.60, high=1.60, dtype=x2.dtype)
            x2_do = x2[do_g].clamp(min=1e-6, max=1.0)
            x2[do_g] = x2_do**gamma

    # Random gaussian noise per ear-volume.
    noise_prob = float(noise_prob)
    if noise_prob > 0:
        do_n = torch.rand((b,), device=device) < float(noise_prob)
        if do_n.any():
            n_n = int(do_n.sum().item())
            std = _rand_uniform((n_n, 1, 1, 1, 1), device=device, low=0.0, high=0.05, dtype=x2.dtype)
            x2_do = x2[do_n]
            x2[do_n] = x2_do + torch.randn_like(x2_do) * std

    x2 = x2.clamp(0.0, 1.0)
    if x2.dtype != orig_dtype:
        x2 = x2.to(orig_dtype)
    return x2.contiguous()


def _make_balanced_sampler(y: np.ndarray, *, seed: int) -> WeightedRandomSampler:
    y = np.asarray(y).astype(int)
    n = int(y.size)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0 or neg <= 0:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.zeros(n, dtype=np.float64)
        w[y == 1] = 0.5 / float(pos)
        w[y == 0] = 0.5 / float(neg)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=int(n), replacement=True, generator=g)


def _bce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    pos_weight: float | None,
    label_smoothing: float,
) -> torch.Tensor:
    y = y.view(-1).to(logits.dtype)
    if float(label_smoothing) > 0:
        eps = float(label_smoothing)
        y = y * (1.0 - eps) + 0.5 * eps
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), y, pos_weight=pw)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), y)


def _focal_loss_vec(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float,
    alpha: float | None,
    pos_weight: float | None,
    label_smoothing: float,
) -> torch.Tensor:
    y = y.view(-1).to(logits.dtype)
    if float(label_smoothing) > 0:
        eps = float(label_smoothing)
        y = y * (1.0 - eps) + 0.5 * eps

    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype) if pos_weight is not None else None
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), y, reduction="none", pos_weight=pw)

    p = torch.sigmoid(logits.view(-1))
    p_t = p * y + (1.0 - p) * (1.0 - y)
    mod = (1.0 - p_t).clamp(min=0.0).pow(float(gamma))
    loss = mod * bce

    if alpha is not None:
        a = float(alpha)
        alpha_t = (a * y + (1.0 - a) * (1.0 - y)).to(loss.dtype)
        loss = alpha_t * loss

    return loss


def _bce_loss_vec(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    pos_weight: float | None,
    label_smoothing: float,
) -> torch.Tensor:
    y = y.view(-1).to(logits.dtype)
    if float(label_smoothing) > 0:
        eps = float(label_smoothing)
        y = y * (1.0 - eps) + 0.5 * eps
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype) if pos_weight is not None else None
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), y, reduction="none", pos_weight=pw)


def _robust_drop_high_loss_mean(loss_vec: torch.Tensor, y_hard: torch.Tensor, *, drop_frac: float) -> torch.Tensor:
    drop_frac = float(drop_frac)
    if drop_frac <= 0:
        return loss_vec.mean()

    y_hard = y_hard.view(-1).to(dtype=torch.long)
    loss_vec = loss_vec.view(-1)
    kept: list[torch.Tensor] = []
    for cls in (0, 1):
        m = y_hard == int(cls)
        n = int(m.sum().item())
        if n <= 0:
            continue
        keep = max(1, int(math.ceil((1.0 - drop_frac) * n)))
        subset = loss_vec[m]
        idx = torch.topk(subset.detach(), k=int(keep), largest=False).indices
        kept.append(subset.gather(0, idx))
    if not kept:
        return loss_vec.mean()
    return torch.cat(kept, dim=0).mean()


@torch.no_grad()
def _eval_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    wl: float,
    ww: float,
    amp: bool,
    specificity_target: float,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float | None,
) -> dict[str, Any]:
    model.eval()
    ys: list[float] = []
    ps: list[float] = []
    losses: list[float] = []

    grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            hu = batch["hu"].to(device, non_blocking=True)  # (B,K,H,W)
            y = batch["y"].to(device, non_blocking=True).view(-1, 1)

            x = _window_hu_to_unit(hu, wl=float(wl), ww=float(ww)).unsqueeze(2)  # (B,K,1,H,W)

            amp_enabled = amp and device.type == "cuda"
            if amp_enabled:
                if hasattr(torch, "amp"):
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                else:
                    autocast_ctx = torch.cuda.amp.autocast()
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                logits = model(x)  # (B,1)
                if loss_name == "focal":
                    lv = _focal_loss_vec(
                        logits,
                        y,
                        gamma=float(focal_gamma),
                        alpha=float(focal_alpha) if focal_alpha is not None else None,
                        pos_weight=None,
                        label_smoothing=0.0,
                    )
                    loss = lv.mean()
                else:
                    loss = _bce_loss(logits, y, pos_weight=None, label_smoothing=0.0)

            prob = torch.sigmoid(logits).detach().float().view(-1).cpu().numpy()
            ys.extend(y.detach().float().view(-1).cpu().numpy().tolist())
            ps.extend(prob.tolist())
            losses.append(float(loss.detach().cpu().item()))

    y_np = np.asarray(ys, dtype=np.int64)
    p_np = np.asarray(ps, dtype=np.float64)
    m = binary_metrics(y_np, p_np, threshold=0.5, specificity_target=float(specificity_target))
    m["loss"] = float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else 0.0
    return m


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="split 根目录（建议使用 --patient-split 生成的目录）"),
    pct: int = typer.Option(20, help="训练数据比例：1 / 20 / 100"),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), help="耳朵级 manifest（不入库）"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/...；不入库）"),
    label_task: str = typer.Option("normal_vs_diseased", help="二分类任务名（见 src/medical_fenlei/tasks.py）"),
    model: str = typer.Option("resnet", help="resnet | vit | unet"),
    backbone: str = typer.Option("resnet18", help="resnet18 | resnet34 | resnet50"),
    vit_patch_size: int = typer.Option(16, help="仅当 model=vit 时：patch_size"),
    vit_hidden_size: int = typer.Option(512, help="仅当 model=vit 时：hidden size"),
    vit_mlp_dim: int = typer.Option(2048, help="仅当 model=vit 时：MLP dim（0 表示自动=4*hidden）"),
    vit_num_layers: int = typer.Option(8, help="仅当 model=vit 时：transformer layers"),
    vit_num_heads: int = typer.Option(8, help="仅当 model=vit 时：attention heads"),
    vit_dropout: float = typer.Option(0.1, help="仅当 model=vit 时：dropout"),
    unet_embed_dim: int = typer.Option(128, help="仅当 model=unet 时：每个 slice 的 embedding dim（UNet out_channels）"),
    unet_channels: str = typer.Option("16,32,64,128,256", help="仅当 model=unet 时：channels（逗号分隔）"),
    unet_strides: str = typer.Option("2,2,2,2", help="仅当 model=unet 时：strides（逗号分隔）"),
    unet_num_res_units: int = typer.Option(2, help="仅当 model=unet 时：num_res_units"),
    aggregator: str = typer.Option("attention", help="z 聚合方式：attention | transformer | mean"),
    attn_hidden: int = typer.Option(128),
    dropout: float = typer.Option(0.2),
    transformer_layers: int = typer.Option(0, help="仅当 aggregator=transformer 时启用（建议 1~2）"),
    transformer_heads: int = typer.Option(8),
    transformer_ff_dim: int = typer.Option(0, help="Transformer FFN 维度（0 表示自动=4*embed_dim）"),
    transformer_dropout: float = typer.Option(0.1),
    transformer_max_len: int = typer.Option(256),
    epochs: int = typer.Option(50),
    batch_size: int = typer.Option(16),
    grad_accum: int = typer.Option(1, help="梯度累积步数（显存不够时用）"),
    num_workers: int = typer.Option(8),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    crop_size: int = typer.Option(192),
    crop_mode: str = typer.Option("temporal_patch", help="crop：bbox_bias | temporal_patch（颞骨 patch）"),
    crop_lateral_band_frac: float = typer.Option(0.6, help="temporal_patch：外侧区域 band 宽度（0~1，越小越靠边）"),
    crop_lateral_bias: float = typer.Option(0.25, help="x-center bias（越小越靠外侧）"),
    crop_min_area: int = typer.Option(300, help="temporal_patch：连通域最小面积阈值（像素）"),
    sampling: str = typer.Option("even", help="even | air_block"),
    block_len: int = typer.Option(64),
    target_spacing: float = typer.Option(0.0, help=">0 时启用：in-plane 重采样到统一 spacing（mm/px），会写入新的 cache key"),
    target_z_spacing: float = typer.Option(0.0, help=">0 时启用：z 方向重采样到统一 spacing（mm），输出固定 num_slices 的物理窗口"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="使用 cache/ears_hu 缓存 HU 体数据"),
    cache_dir: Path = typer.Option(Path("cache/ears_hu"), help="缓存目录（不入库）"),
    lr: float = typer.Option(3e-4),
    weight_decay: float = typer.Option(1e-4),
    label_smoothing: float = typer.Option(0.05, help="二分类 label smoothing（抗噪）"),
    amp: bool = typer.Option(True),
    tf32: bool = typer.Option(True),
    cudnn_benchmark: bool = typer.Option(True),
    max_pos_weight: float = typer.Option(20.0, help="BCE pos_weight 上限（避免极端爆炸）"),
    use_pos_weight: bool = typer.Option(True, "--pos-weight/--no-pos-weight"),
    augment: bool = typer.Option(True, "--augment/--no-augment"),
    aug_flip_prob: float = typer.Option(0.2),
    aug_intensity_prob: float = typer.Option(0.6),
    aug_noise_prob: float = typer.Option(0.2),
    aug_gamma_prob: float = typer.Option(0.2),
    aug_affine_prob: float = typer.Option(0.0, help="几何增强：随机仿射（按 ear-volume 同步应用到所有 slice）"),
    aug_affine_degrees: float = typer.Option(12.0, help="随机旋转角度范围（±deg）"),
    aug_affine_translate: float = typer.Option(0.08, help="随机平移范围（相对图像尺寸比例，如 0.08 表示 8%）"),
    aug_affine_scale_low: float = typer.Option(0.85, help="随机缩放下限"),
    aug_affine_scale_high: float = typer.Option(1.15, help="随机缩放上限"),
    base_wl: float = typer.Option(500.0),
    base_ww: float = typer.Option(3000.0),
    wl_jitter: float = typer.Option(200.0),
    ww_scale_low: float = typer.Option(0.8),
    ww_scale_high: float = typer.Option(1.2),
    loss: str = typer.Option("bce", help="loss：bce | focal"),
    focal_gamma: float = typer.Option(2.0),
    focal_alpha: float | None = typer.Option(None, help="focal alpha（None 表示不启用 alpha 平衡项）"),
    drop_high_loss_frac: float = typer.Option(0.0, help="robust：每个 batch 内按类丢弃 top-loss 比例（0 表示禁用）"),
    drop_high_loss_warmup_epochs: int = typer.Option(0, help="robust：在前 N 个 epoch 线性 warmup 到 drop_high_loss_frac"),
    early_stop_patience: int = typer.Option(10),
    early_stop_metric: str = typer.Option("acc", help="acc | auprc | auroc | sensitivity_at_spec | f1 | loss"),
    early_stop_min_delta: float = typer.Option(0.0),
    specificity_target: float = typer.Option(0.95, help="用于 sensitivity_at_spec 的目标 specificity"),
    freeze_backbone_epochs: int = typer.Option(0, help="前 N 个 epoch 冻结 backbone（linear probe）"),
    train_limit: int | None = typer.Option(None, help="仅使用前 N 个训练耳朵样本（用于 smoke）"),
    val_limit: int | None = typer.Option(None, help="仅使用前 N 个验证耳朵样本（用于 smoke）"),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="上传训练指标到 Weights & Biases"),
    wandb_project: str = typer.Option("medical_fenlei"),
    wandb_entity: str | None = typer.Option(None),
    wandb_name: str | None = typer.Option(None),
    wandb_group: str | None = typer.Option(None),
    wandb_tags: str = typer.Option("", help="逗号分隔 tags"),
    wandb_mode: str = typer.Option("online", help="online | offline | disabled"),
    wandb_dir: Path = typer.Option(Path("wandb"), help="wandb 本地目录（不入库）"),
    seed: int = typer.Option(42),
) -> None:
    crop_mode = str(crop_mode).strip()
    if crop_mode not in ("bbox_bias", "temporal_patch"):
        raise ValueError("crop_mode must be one of: bbox_bias, temporal_patch")

    model = str(model).strip().lower()
    if model not in ("resnet", "vit", "unet"):
        raise ValueError("model must be one of: resnet, vit, unet")

    loss_name = str(loss).strip().lower()
    if loss_name not in ("bce", "focal"):
        raise ValueError("loss must be one of: bce, focal")

    if drop_high_loss_frac < 0 or drop_high_loss_frac >= 1:
        raise ValueError("drop_high_loss_frac must be in [0, 1)")

    train_csv = splits_root / f"{pct}pct" / "train.csv"
    val_csv = splits_root / f"{pct}pct" / "val.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    task = resolve_task(label_task)
    if task.kind != "binary":
        raise ValueError(f"train_ear2d currently supports binary tasks only; got: {task.kind}")

    man = pd.read_csv(manifest_csv)
    if man.empty:
        raise typer.Exit(code=2)

    train_exam = set(pd.read_csv(train_csv)["exam_id"].astype(int).tolist())
    val_exam = set(pd.read_csv(val_csv)["exam_id"].astype(int).tolist())

    df = man.loc[man["has_label"].fillna(False)].copy()
    df["label_code"] = pd.to_numeric(df["label_code"], errors="coerce")
    df = df[np.isfinite(df["label_code"])].copy()
    df["label_code"] = df["label_code"].astype(int)

    rel_codes = task.relevant_codes()
    pos_codes = set(task.pos_codes)
    neg_codes = set(task.neg_codes)

    df = df[df["label_code"].isin(sorted(rel_codes))].copy()
    if df.empty:
        typer.echo(f"no ears after filtering by task={task.name} codes={sorted(rel_codes)}")
        raise typer.Exit(code=2)

    df["y"] = df["label_code"].map(lambda c: 1.0 if int(c) in pos_codes else 0.0)

    train_df = df[df["exam_id"].astype(int).isin(train_exam)].copy()
    val_df = df[df["exam_id"].astype(int).isin(val_exam)].copy()
    if train_df.empty or val_df.empty:
        typer.echo("empty train/val after split filtering")
        raise typer.Exit(code=2)

    train_df = train_df[["exam_id", "series_relpath", "side", "y", "label_code"]].reset_index(drop=True)
    val_df = val_df[["exam_id", "series_relpath", "side", "y", "label_code"]].reset_index(drop=True)

    if train_limit is not None and int(train_limit) > 0 and len(train_df) > int(train_limit):
        train_df = train_df.sample(n=int(train_limit), random_state=int(seed)).reset_index(drop=True)
    if val_limit is not None and int(val_limit) > 0 and len(val_df) > int(val_limit):
        val_df = val_df.sample(n=int(val_limit), random_state=int(seed)).reset_index(drop=True)

    dicom_root = infer_dicom_root(dicom_base)
    spec = EarPreprocessSpec(
        num_slices=int(num_slices),
        image_size=int(image_size),
        crop_size=int(crop_size),
        crop_mode=str(crop_mode),
        crop_lateral_band_frac=float(crop_lateral_band_frac),
        crop_lateral_bias=float(crop_lateral_bias),
        crop_min_area=int(crop_min_area),
        sampling=str(sampling),
        block_len=int(block_len),
        target_spacing=float(target_spacing) if float(target_spacing) > 0 else None,
        target_z_spacing=float(target_z_spacing) if float(target_z_spacing) > 0 else None,
        version="v3",
    )

    used_cache_dir = None
    if cache:
        ts_tag = f"_ts{float(target_spacing):.6g}" if float(target_spacing) > 0 else ""
        tz_tag = f"_tz{float(target_z_spacing):.6g}" if float(target_z_spacing) > 0 else ""
        used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}_c{int(crop_size)}_{str(sampling)}{ts_tag}{tz_tag}_crop{crop_mode}"

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_ds = EarCTHUEarDataset(index_df=train_df, dicom_root=dicom_root, spec=spec, cache_dir=used_cache_dir, return_meta=True)
    val_ds = EarCTHUEarDataset(index_df=val_df, dicom_root=dicom_root, spec=spec, cache_dir=used_cache_dir, return_meta=True)

    train_y = train_df["y"].astype(float).to_numpy()
    sampler = _make_balanced_sampler(train_y, seed=int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(max(1, num_workers // 2)),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    pos_n = int((train_y == 1).sum())
    neg_n = int((train_y == 0).sum())
    pos_weight = None
    if use_pos_weight and pos_n > 0 and neg_n > 0:
        pos_weight = min(float(max_pos_weight), float(neg_n) / float(pos_n))

    model_type: str
    if model == "resnet":
        model_type = "SliceAttentionResNet"
        net: torch.nn.Module = SliceAttentionResNet(
            backbone=str(backbone),
            in_channels=1,
            aggregator=str(aggregator),
            attn_hidden=int(attn_hidden),
            dropout=float(dropout),
            out_dim=1,
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            transformer_ff_dim=int(transformer_ff_dim),
            transformer_dropout=float(transformer_dropout),
            transformer_max_len=int(transformer_max_len),
        )
    elif model == "vit":
        model_type = "SliceAttentionViT"
        net = SliceAttentionViT(
            in_channels=1,
            image_size=int(image_size),
            patch_size=int(vit_patch_size),
            vit_hidden_size=int(vit_hidden_size),
            vit_mlp_dim=int(vit_mlp_dim),
            vit_num_layers=int(vit_num_layers),
            vit_num_heads=int(vit_num_heads),
            vit_dropout=float(vit_dropout),
            aggregator=str(aggregator),
            attn_hidden=int(attn_hidden),
            dropout=float(dropout),
            out_dim=1,
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            transformer_ff_dim=int(transformer_ff_dim),
            transformer_dropout=float(transformer_dropout),
            transformer_max_len=int(transformer_max_len),
        )
    else:
        model_type = "SliceAttentionUNet"

        def _parse_ints(s: str, *, name: str) -> tuple[int, ...]:
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            if not parts:
                raise ValueError(f"{name} must be a non-empty comma-separated list")
            return tuple(int(p) for p in parts)

        net = SliceAttentionUNet(
            in_channels=1,
            image_size=int(image_size),
            unet_channels=_parse_ints(unet_channels, name="unet_channels"),
            unet_strides=_parse_ints(unet_strides, name="unet_strides"),
            unet_num_res_units=int(unet_num_res_units),
            unet_embed_dim=int(unet_embed_dim),
            aggregator=str(aggregator),
            attn_hidden=int(attn_hidden),
            dropout=float(dropout),
            out_dim=1,
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            transformer_ff_dim=int(transformer_ff_dim),
            transformer_dropout=float(transformer_dropout),
            transformer_max_len=int(transformer_max_len),
        )

    model_any = net.to(device)
    # keep the old variable name for minimal diff
    model = model_any

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    amp_enabled = amp and device.type == "cuda"
    if not amp_enabled:
        scaler = None
    elif hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_task = str(task.name).replace("/", "_")
    out = output_dir or Path("outputs") / f"ear2d_{str(backbone)}__{safe_task}_{pct}pct_{ts}"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"
    config_path = out / "run_config.json"

    config = {
        "task": {"name": task.name, "pos_codes": list(task.pos_codes), "neg_codes": list(task.neg_codes)},
        "data": {
            "splits_root": str(splits_root),
            "pct": int(pct),
            "manifest_csv": str(manifest_csv),
            "dicom_root": str(dicom_root),
            "cache_dir": str(used_cache_dir) if used_cache_dir is not None else None,
            "spec": spec.__dict__,
        },
        "model": {"type": str(model_type), "spec": model.spec.__dict__},
        "train": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "pos_weight": pos_weight,
            "label_smoothing": float(label_smoothing),
            "amp": bool(amp),
            "augment": bool(augment),
            "base_wl": float(base_wl),
            "base_ww": float(base_ww),
            "wl_jitter": float(wl_jitter),
            "ww_scale_low": float(ww_scale_low),
            "ww_scale_high": float(ww_scale_high),
            "aug_flip_prob": float(aug_flip_prob),
            "aug_intensity_prob": float(aug_intensity_prob),
            "aug_noise_prob": float(aug_noise_prob),
            "aug_gamma_prob": float(aug_gamma_prob),
            "aug_affine_prob": float(aug_affine_prob),
            "aug_affine_degrees": float(aug_affine_degrees),
            "aug_affine_translate": float(aug_affine_translate),
            "aug_affine_scale_low": float(aug_affine_scale_low),
            "aug_affine_scale_high": float(aug_affine_scale_high),
            "loss": str(loss_name),
            "focal_gamma": float(focal_gamma),
            "focal_alpha": float(focal_alpha) if focal_alpha is not None else None,
            "drop_high_loss_frac": float(drop_high_loss_frac),
            "drop_high_loss_warmup_epochs": int(drop_high_loss_warmup_epochs),
            "early_stop": {
                "metric": str(early_stop_metric),
                "patience": int(early_stop_patience),
                "min_delta": float(early_stop_min_delta),
                "specificity_target": float(specificity_target),
            },
            "freeze_backbone_epochs": int(freeze_backbone_epochs),
            "seed": int(seed),
        },
        "git_sha": _git_sha(),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"task: ear-level binary ({task.name})  train_ears={len(train_ds)} val_ears={len(val_ds)}  pos={pos_n} neg={neg_n} pos_weight={pos_weight}")
    typer.echo(f"model: {model_type}  spec={model.spec.__dict__}")
    typer.echo(f"dicom_root: {dicom_root}")
    if used_cache_dir is not None:
        typer.echo(f"cache_dir: {used_cache_dir}")
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
                config=config,
            )
        except Exception as e:
            typer.echo(f"wandb: init failed ({type(e).__name__}: {e}); continue without wandb")
            wb = None
            wb_run = None

    early_stop_metric = str(early_stop_metric).strip().lower()
    metric_alias = {
        "acc": "accuracy",
    }
    early_stop_metric = metric_alias.get(early_stop_metric, early_stop_metric)

    metric_mode = "min" if early_stop_metric == "loss" else "max"
    best = float("inf") if metric_mode == "min" else float("-inf")
    bad = 0

    global_step = 0
    total_steps = int(epochs) * max(1, len(train_loader))
    warmup_steps = int(math.ceil(total_steps * 0.05))

    def _lr_for_step(step: int) -> float:
        if total_steps <= 0:
            return float(lr)
        if step < warmup_steps and warmup_steps > 0:
            return float(lr) * float(step + 1) / float(warmup_steps)
        # cosine decay to 0
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return float(lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _get_backbone_module(m: torch.nn.Module) -> torch.nn.Module | None:
        for name in ("backbone", "encoder", "unet"):
            sub = getattr(m, name, None)
            if isinstance(sub, torch.nn.Module):
                return sub
        return None

    backbone_mod = _get_backbone_module(model)
    if backbone_mod is None and int(freeze_backbone_epochs) > 0:
        typer.echo("warning: freeze_backbone_epochs set but model has no recognized backbone/encoder/unet; skipping freeze")

    for epoch in range(1, int(epochs) + 1):
        # optional linear probe
        if backbone_mod is not None:
            freeze_now = int(freeze_backbone_epochs) > 0 and int(epoch) <= int(freeze_backbone_epochs)
            for p in backbone_mod.parameters():
                p.requires_grad = not bool(freeze_now)

        model.train(True)
        train_losses: list[float] = []
        ys: list[float] = []
        ps: list[float] = []

        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            hu = batch["hu"].to(device, non_blocking=True)  # (B,K,H,W)
            y = batch["y"].to(device, non_blocking=True).view(-1, 1)
            y_hard = (y.view(-1) >= 0.5).to(dtype=torch.long)

            # WL/WW jitter
            if augment:
                wl = float(base_wl) + float(_rand_uniform((), device=device, low=-float(wl_jitter), high=float(wl_jitter), dtype=torch.float32).item())
                ww = float(base_ww) * float(_rand_uniform((), device=device, low=float(ww_scale_low), high=float(ww_scale_high), dtype=torch.float32).item())
            else:
                wl, ww = float(base_wl), float(base_ww)

            x = _window_hu_to_unit(hu, wl=wl, ww=ww).unsqueeze(2)  # (B,K,1,H,W)

            if augment:
                x = _augment_unit_volume(
                    x,
                    flip_prob=float(aug_flip_prob),
                    intensity_prob=float(aug_intensity_prob),
                    noise_prob=float(aug_noise_prob),
                    gamma_prob=float(aug_gamma_prob),
                    affine_prob=float(aug_affine_prob),
                    affine_degrees=float(aug_affine_degrees),
                    affine_translate=float(aug_affine_translate),
                    affine_scale_low=float(aug_affine_scale_low),
                    affine_scale_high=float(aug_affine_scale_high),
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
                logits = model(x)  # (B,1)
                if loss_name == "focal":
                    loss_vec = _focal_loss_vec(
                        logits,
                        y,
                        gamma=float(focal_gamma),
                        alpha=float(focal_alpha) if focal_alpha is not None else None,
                        pos_weight=pos_weight,
                        label_smoothing=float(label_smoothing),
                    )
                else:
                    loss_vec = _bce_loss_vec(logits, y, pos_weight=pos_weight, label_smoothing=float(label_smoothing))

                drop_frac_now = float(drop_high_loss_frac)
                if int(drop_high_loss_warmup_epochs) > 0:
                    t = float(max(0, int(epoch) - 1)) / float(max(1, int(drop_high_loss_warmup_epochs)))
                    drop_frac_now = float(drop_frac_now) * float(min(1.0, t))

                loss = _robust_drop_high_loss_mean(loss_vec, y_hard, drop_frac=drop_frac_now) / float(max(1, int(grad_accum)))

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % int(max(1, grad_accum)) == 0:
                # cosine+warmup
                lr_now = _lr_for_step(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = float(lr_now)

                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            prob = torch.sigmoid(logits.detach()).float().view(-1).cpu().numpy()
            ys.extend(y.detach().float().view(-1).cpu().numpy().tolist())
            ps.extend(prob.tolist())
            train_losses.append(float(loss.detach().cpu().item()) * float(max(1, int(grad_accum))))

        train_m = binary_metrics(np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float64), threshold=0.5, specificity_target=float(specificity_target))
        train_m["loss"] = float(np.mean(np.asarray(train_losses, dtype=np.float64))) if train_losses else 0.0

        val_m = _eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            wl=float(base_wl),
            ww=float(base_ww),
            amp=bool(amp),
            specificity_target=float(specificity_target),
            loss_name=str(loss_name),
            focal_gamma=float(focal_gamma),
            focal_alpha=float(focal_alpha) if focal_alpha is not None else None,
        )

        rec = {"epoch": int(epoch), "train": train_m, "val": val_m}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if wb_run is not None:
            try:
                lr_now = None
                try:
                    lr_now = float(optimizer.param_groups[0]["lr"])
                except Exception:
                    lr_now = None

                wandb_metrics = {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "lr": lr_now,
                    "train/loss": float(train_m.get("loss")),
                    "train/acc": train_m.get("accuracy"),
                    "train/auroc": train_m.get("auroc"),
                    "train/auprc": train_m.get("auprc"),
                    "train/f1": train_m.get("f1"),
                    "val/loss": float(val_m.get("loss")),
                    "val/acc": val_m.get("accuracy"),
                    "val/auroc": val_m.get("auroc"),
                    "val/auprc": val_m.get("auprc"),
                    "val/f1": val_m.get("f1"),
                }
                wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
                wb_run.log(wandb_metrics, step=int(epoch))
            except Exception:
                pass

        typer.echo(
            f"epoch {epoch}: train_loss={train_m['loss']:.4f} acc={train_m.get('accuracy'):.3f}  "
            f"val_loss={val_m['loss']:.4f} acc={val_m.get('accuracy'):.3f}"
        )

        ckpt = {
            "epoch": int(epoch),
            "task": "ear2d",
            "label_task": str(task.name),
            "pos_codes": tuple(int(x) for x in task.pos_codes),
            "neg_codes": tuple(int(x) for x in task.neg_codes),
            "model_type": str(model_type),
            "model_spec": model.spec.__dict__,
            "state_dict": model.state_dict(),
            "config": config,
            "val": val_m,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        score = val_m.get(early_stop_metric)
        if score is None:
            score = val_m.get("loss") if early_stop_metric == "loss" else None
        if score is None:
            typer.echo(f"warning: early_stop_metric {early_stop_metric} missing; disabling early-stop")
            continue

        improved = False
        if metric_mode == "min":
            improved = (best - float(score)) > float(early_stop_min_delta)
        else:
            improved = (float(score) - best) > float(early_stop_min_delta)

        if improved:
            best = float(score)
            bad = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
        else:
            bad += 1

        if int(early_stop_patience) > 0 and bad >= int(early_stop_patience):
            typer.echo(f"early stopping: no improvement on {early_stop_metric} for {early_stop_patience} epochs")
            break

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    app()
