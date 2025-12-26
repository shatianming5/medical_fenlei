from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader, WeightedRandomSampler

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.ear_dataset import EarCTHUEarDataset, EarPreprocessSpec
from medical_fenlei.metrics import classification_report_from_confusion
from medical_fenlei.models.slice_attention_resnet import SliceAttentionResNet
from medical_fenlei.paths import infer_dicom_root

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
) -> torch.Tensor:
    if x.ndim != 5:
        return x
    if max(flip_prob, intensity_prob, noise_prob, gamma_prob) <= 0:
        return x

    b, k = int(x.shape[0]), int(x.shape[1])
    n = b * k
    x2 = x.reshape(n, *x.shape[2:]).contiguous()  # (N,1,H,W)

    orig_dtype = x2.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        x2 = x2.float()

    device = x2.device

    if flip_prob > 0:
        do_h = torch.rand((n,), device=device) < float(flip_prob)
        do_w = torch.rand((n,), device=device) < float(flip_prob)
        if do_h.any():
            x2[do_h] = x2[do_h].flip(-2)
        if do_w.any():
            x2[do_w] = x2[do_w].flip(-1)

    if intensity_prob > 0:
        do_i = torch.rand((n,), device=device) < float(intensity_prob)
        if do_i.any():
            scale = _rand_uniform((int(do_i.sum()), 1, 1, 1), device=device, low=0.90, high=1.10, dtype=x2.dtype)
            shift = _rand_uniform((int(do_i.sum()), 1, 1, 1), device=device, low=-0.10, high=0.10, dtype=x2.dtype)
            x2_do = x2[do_i]
            x2_do = x2_do * scale + shift
            x2[do_i] = x2_do

    if gamma_prob > 0:
        do_g = torch.rand((n,), device=device) < float(gamma_prob)
        if do_g.any():
            gamma = _rand_uniform((int(do_g.sum()), 1, 1, 1), device=device, low=0.70, high=1.50, dtype=x2.dtype)
            x2_do = x2[do_g].clamp(min=1e-6, max=1.0)
            x2[do_g] = x2_do**gamma

    if noise_prob > 0:
        do_n = torch.rand((n,), device=device) < float(noise_prob)
        if do_n.any():
            std = _rand_uniform((int(do_n.sum()), 1, 1, 1), device=device, low=0.0, high=0.03, dtype=x2.dtype)
            x2[do_n] = x2[do_n] + torch.randn_like(x2[do_n]) * std

    x2 = x2.clamp(0.0, 1.0)
    if x2.dtype != orig_dtype:
        x2 = x2.to(orig_dtype)
    return x2.reshape(b, k, *x.shape[2:]).contiguous()


def _make_class_balanced_sampler(y: np.ndarray, *, seed: int) -> WeightedRandomSampler:
    y = np.asarray(y).astype(int)
    n = int(y.size)
    if n <= 0:
        w = np.ones(0, dtype=np.float64)
    else:
        counts = np.bincount(y)
        w = np.zeros(n, dtype=np.float64)
        for cls in range(int(counts.size)):
            c = int(counts[cls])
            if c <= 0:
                continue
            w[y == cls] = 1.0 / float(c)
        if float(w.sum()) > 0:
            w = w / float(w.sum())
        else:
            w = np.ones(n, dtype=np.float64) / float(n)

    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=int(n), replacement=True, generator=g)


def _cross_entropy_loss_vec(logits: torch.Tensor, y: torch.Tensor, *, label_smoothing: float) -> torch.Tensor:
    y = y.view(-1).to(dtype=torch.long)
    logits = logits.view(y.shape[0], -1)

    logp = torch.nn.functional.log_softmax(logits, dim=1)
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    if float(label_smoothing) > 0:
        eps = float(label_smoothing)
        smooth = -logp.mean(dim=1)
        return (1.0 - eps) * nll + eps * smooth
    return nll


@torch.no_grad()
def _eval_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    wl: float,
    ww: float,
    amp: bool,
    num_classes: int,
    label_smoothing: float,
    class_id_to_name: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    ys: list[int] = []
    preds: list[int] = []

    grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            hu = batch["hu"].to(device, non_blocking=True)  # (B,K,H,W)
            y = batch["y"].to(device, non_blocking=True).view(-1).long()

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
                logits = model(x)  # (B,C)
                lv = _cross_entropy_loss_vec(logits, y, label_smoothing=float(label_smoothing))
                loss = lv.mean()

            pred = logits.detach().float().argmax(dim=1).view(-1)
            ys.extend(y.detach().cpu().numpy().astype(int).tolist())
            preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
            losses.append(float(loss.detach().cpu().item()))

    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for yt, yp in zip(ys, preds):
        if 0 <= int(yt) < int(num_classes) and 0 <= int(yp) < int(num_classes):
            cm[int(yt), int(yp)] += 1

    rep = classification_report_from_confusion(cm, class_id_to_name=class_id_to_name)
    rep["loss"] = float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else 0.0
    return rep


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="split 根目录（建议使用 --patient-split 生成的目录）"),
    pct: int = typer.Option(20, help="训练数据比例：1 / 20 / 100"),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), help="耳朵级 manifest（不入库）"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/...；不入库）"),
    task_name: str = typer.Option("abnormal_subtype", help="仅用于命名输出目录（stage2 训练）"),
    target_codes: str = typer.Option("1,2,3,6", help="异常细分的目标 code 集合（逗号分隔）"),
    merge_code4_into_other: bool = typer.Option(True, "--merge-code4/--no-merge-code4", help="将 code4(4) 合并进 other(6)"),
    backbone: str = typer.Option("resnet18", help="resnet18 | resnet34 | resnet50"),
    aggregator: str = typer.Option("attention", help="z 聚合方式：attention | transformer | mean"),
    attn_hidden: int = typer.Option(128),
    dropout: float = typer.Option(0.2),
    transformer_layers: int = typer.Option(0, help="仅当 aggregator=transformer 时启用（建议 1~2）"),
    transformer_heads: int = typer.Option(8),
    transformer_ff_dim: int = typer.Option(0),
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
    label_smoothing: float = typer.Option(0.0, help="多类 CE smoothing（抗噪）"),
    amp: bool = typer.Option(True),
    tf32: bool = typer.Option(True),
    cudnn_benchmark: bool = typer.Option(True),
    augment: bool = typer.Option(True, "--augment/--no-augment"),
    aug_flip_prob: float = typer.Option(0.2),
    aug_intensity_prob: float = typer.Option(0.6),
    aug_noise_prob: float = typer.Option(0.2),
    aug_gamma_prob: float = typer.Option(0.2),
    base_wl: float = typer.Option(500.0),
    base_ww: float = typer.Option(3000.0),
    wl_jitter: float = typer.Option(200.0),
    ww_scale_low: float = typer.Option(0.8),
    ww_scale_high: float = typer.Option(1.2),
    early_stop_patience: int = typer.Option(10),
    early_stop_metric: str = typer.Option("macro_f1", help="macro_f1 | macro_recall | weighted_f1 | loss"),
    early_stop_min_delta: float = typer.Option(0.0),
    seed: int = typer.Option(42),
) -> None:
    crop_mode = str(crop_mode).strip()
    if crop_mode not in ("bbox_bias", "temporal_patch"):
        raise ValueError("crop_mode must be one of: bbox_bias, temporal_patch")

    code_list = [int(x.strip()) for x in str(target_codes).split(",") if x.strip()]
    if not code_list:
        raise typer.Exit(code=2)
    if merge_code4_into_other and 4 not in code_list:
        code_list = list(code_list) + [4]

    if 6 not in code_list:
        raise ValueError("stage2 expects other(6) included in target_codes")

    # Define mapping: codes -> class_id in the provided order (after merge).
    base_codes = [int(x) for x in code_list if int(x) != 4]
    base_codes = list(dict.fromkeys(base_codes))  # stable unique
    code_to_class_id = {int(c): int(i) for i, c in enumerate(base_codes)}
    num_classes = int(len(base_codes))
    class_id_to_name = {int(i): str(CLASS_ID_TO_NAME[int(c) - 1]) for c, i in code_to_class_id.items()}

    train_csv = splits_root / f"{pct}pct" / "train.csv"
    val_csv = splits_root / f"{pct}pct" / "val.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    man = pd.read_csv(manifest_csv)
    if man.empty:
        raise typer.Exit(code=2)

    train_exam = set(pd.read_csv(train_csv)["exam_id"].astype(int).tolist())
    val_exam = set(pd.read_csv(val_csv)["exam_id"].astype(int).tolist())

    df = man.loc[man["has_label"].fillna(False)].copy()
    df["label_code"] = pd.to_numeric(df["label_code"], errors="coerce")
    df = df[np.isfinite(df["label_code"])].copy()
    df["label_code"] = df["label_code"].astype(int)
    df = df[df["label_code"].isin(sorted(code_list))].copy()
    if df.empty:
        typer.echo("no ears after label_code filtering")
        raise typer.Exit(code=2)

    def _map_code(c: int) -> int:
        if merge_code4_into_other and int(c) == 4:
            c = 6
        return int(code_to_class_id[int(c)])

    df["y"] = df["label_code"].map(_map_code).astype(int)

    train_df = df[df["exam_id"].astype(int).isin(train_exam)].copy()
    val_df = df[df["exam_id"].astype(int).isin(val_exam)].copy()
    if train_df.empty or val_df.empty:
        typer.echo("empty train/val after split filtering")
        raise typer.Exit(code=2)

    train_df = train_df[["exam_id", "series_relpath", "side", "y", "label_code"]].reset_index(drop=True)
    val_df = val_df[["exam_id", "series_relpath", "side", "y", "label_code"]].reset_index(drop=True)

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

    train_ds = EarCTHUEarDataset(
        index_df=train_df,
        dicom_root=dicom_root,
        spec=spec,
        cache_dir=used_cache_dir,
        return_meta=True,
        y_dtype=torch.long,
    )
    val_ds = EarCTHUEarDataset(
        index_df=val_df,
        dicom_root=dicom_root,
        spec=spec,
        cache_dir=used_cache_dir,
        return_meta=True,
        y_dtype=torch.long,
    )

    train_y = train_df["y"].astype(int).to_numpy()
    sampler = _make_class_balanced_sampler(train_y, seed=int(seed))

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

    model = SliceAttentionResNet(
        backbone=str(backbone),
        in_channels=1,
        aggregator=str(aggregator),
        attn_hidden=int(attn_hidden),
        dropout=float(dropout),
        out_dim=int(num_classes),
        transformer_layers=int(transformer_layers),
        transformer_heads=int(transformer_heads),
        transformer_ff_dim=int(transformer_ff_dim),
        transformer_dropout=float(transformer_dropout),
        transformer_max_len=int(transformer_max_len),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    amp_enabled = amp and device.type == "cuda"
    if not amp_enabled:
        scaler = None
    elif hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_task = str(task_name).replace("/", "_")
    out = output_dir or Path("outputs") / f"ear2d_{str(backbone)}__{safe_task}_{pct}pct_{ts}"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"
    config_path = out / "run_config.json"

    config = {
        "task": {"name": str(task_name), "kind": "multiclass", "code_to_class_id": code_to_class_id, "merge_code4_into_other": bool(merge_code4_into_other)},
        "data": {
            "splits_root": str(splits_root),
            "pct": int(pct),
            "manifest_csv": str(manifest_csv),
            "dicom_root": str(dicom_root),
            "cache_dir": str(used_cache_dir) if used_cache_dir is not None else None,
            "spec": spec.__dict__,
        },
        "model": {"type": "SliceAttentionResNet", "spec": model.spec.__dict__},
        "train": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
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
            "early_stop": {"metric": str(early_stop_metric), "patience": int(early_stop_patience), "min_delta": float(early_stop_min_delta)},
            "seed": int(seed),
        },
        "class_id_to_name": {int(k): str(v) for k, v in class_id_to_name.items()},
        "git_sha": _git_sha(),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"task: ear2d-multiclass ({task_name})  train_ears={len(train_ds)} val_ears={len(val_ds)} classes={num_classes}")
    typer.echo(f"codes: {sorted(code_list)}  base_codes: {base_codes}  merge_code4_into_other={merge_code4_into_other}")
    typer.echo(f"dicom_root: {dicom_root}")
    if used_cache_dir is not None:
        typer.echo(f"cache_dir: {used_cache_dir}")
    typer.echo(f"output: {out}")

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
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return float(lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    for epoch in range(1, int(epochs) + 1):
        model.train(True)
        train_losses: list[float] = []
        ys: list[int] = []
        preds: list[int] = []

        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            hu = batch["hu"].to(device, non_blocking=True)  # (B,K,H,W)
            y = batch["y"].to(device, non_blocking=True).view(-1).long()

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
                logits = model(x)  # (B,C)
                lv = _cross_entropy_loss_vec(logits, y, label_smoothing=float(label_smoothing))
                loss = lv.mean() / float(max(1, int(grad_accum)))

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % int(max(1, grad_accum)) == 0:
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

            pred = logits.detach().float().argmax(dim=1).view(-1)
            ys.extend(y.detach().cpu().numpy().astype(int).tolist())
            preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
            train_losses.append(float(loss.detach().cpu().item()) * float(max(1, int(grad_accum))))

        train_cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
        for yt, yp in zip(ys, preds):
            if 0 <= int(yt) < int(num_classes) and 0 <= int(yp) < int(num_classes):
                train_cm[int(yt), int(yp)] += 1
        train_rep = classification_report_from_confusion(train_cm, class_id_to_name=class_id_to_name)
        train_rep["loss"] = float(np.mean(np.asarray(train_losses, dtype=np.float64))) if train_losses else 0.0

        val_rep = _eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            wl=float(base_wl),
            ww=float(base_ww),
            amp=bool(amp),
            num_classes=int(num_classes),
            label_smoothing=0.0,
            class_id_to_name=class_id_to_name,
        )

        report_path = rep_dir / f"epoch_{epoch}.json"
        report_path.write_text(json.dumps({"train": train_rep, "val": val_rep}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        monitor_map = {
            "loss": float(val_rep["loss"]),
            "macro_recall": float(val_rep["macro_recall"]),
            "macro_f1": float(val_rep["macro_f1"]),
            "weighted_f1": float(val_rep["weighted_f1"]),
        }
        if early_stop_metric not in monitor_map:
            raise ValueError(f"unknown early_stop_metric: {early_stop_metric}")
        score = float(monitor_map[early_stop_metric])

        rec = {"epoch": int(epoch), "train": train_rep, "val": val_rep, "early_stop": {"metric": str(early_stop_metric), "score": float(score)}}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        typer.echo(
            f"epoch {epoch}: train_loss={train_rep['loss']:.4f} acc={train_rep['accuracy']:.3f} "
            f"val_loss={val_rep['loss']:.4f} acc={val_rep['accuracy']:.3f} macro_f1={val_rep['macro_f1']:.3f}"
        )

        ckpt = {
            "epoch": int(epoch),
            "task": "ear2d_multiclass",
            "task_name": str(task_name),
            "class_id_to_name": {int(k): str(v) for k, v in class_id_to_name.items()},
            "code_to_class_id": {int(k): int(v) for k, v in code_to_class_id.items()},
            "merge_code4_into_other": bool(merge_code4_into_other),
            "model_spec": model.spec.__dict__,
            "state_dict": model.state_dict(),
            "config": config,
            "val": val_rep,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

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


if __name__ == "__main__":
    app()
