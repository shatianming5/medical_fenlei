from __future__ import annotations

import json
import time
from contextlib import nullcontext
import gc
from pathlib import Path
import re

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dual_dataset import EarCTDualDataset
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
        return {"acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "n": 0, "left_n": 0, "right_n": 0}

    total_correct = int(((pred == labels) & mask).sum().item())

    left_mask = mask[:, 0]
    right_mask = mask[:, 1]
    left_n = int(left_mask.sum().item())
    right_n = int(right_mask.sum().item())
    left_correct = int(((pred[:, 0] == labels[:, 0]) & left_mask).sum().item())
    right_correct = int(((pred[:, 1] == labels[:, 1]) & right_mask).sum().item())

    left_acc = float(left_correct / left_n) if left_n > 0 else 0.0
    right_acc = float(right_correct / right_n) if right_n > 0 else 0.0

    return {
        "acc": float(total_correct / total_n),
        "left_acc": left_acc,
        "right_acc": right_acc,
        "n": total_n,
        "left_n": left_n,
        "right_n": right_n,
    }


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
            x = torch.randn(bs, 2, 1, num_slices, image_size, image_size, device=device)
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
    total_acc = 0.0
    total_left = 0.0
    total_right = 0.0
    total_n = 0
    total_left_n = 0
    total_right_n = 0
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
                logits = model(x)  # (B,2,C)
                loss = _masked_ce_loss(logits, y, m, loss_fn=loss_fn)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            metrics = _masked_accuracy(logits.detach(), y.detach(), m.detach())
            left_n = int(m[:, 0].sum().item())
            right_n = int(m[:, 1].sum().item())

            total_loss += float(loss.detach().cpu().item())
            total_acc += float(metrics["acc"]) * int(metrics["n"])
            total_left += float(metrics["left_acc"]) * left_n
            total_right += float(metrics["right_acc"]) * right_n
            total_n += int(metrics["n"])
            total_left_n += left_n
            total_right_n += right_n
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

    if total_n <= 0:
        out = {"loss": total_loss / max(n_batches, 1), "acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "n": 0}
        if cm is not None:
            out["confusion_matrix"] = cm
        return out

    out = {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / total_n,
        "left_acc": (total_left / max(total_left_n, 1)) if total_left_n > 0 else 0.0,
        "right_acc": (total_right / max(total_right_n, 1)) if total_right_n > 0 else 0.0,
        "n": total_n,
    }
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


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="由 scripts/make_splits_dual.py 生成"),
    pct: int = typer.Option(100, help="训练数据比例：1 / 20 / 100"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/<timestamp>）"),
    model: str = typer.Option(
        "dual_resnet10_3d",
        help="dual_resnet{10,18,34,50,101,152,200}_3d | dual_unet_3d | dual_vit_3d",
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
    vit_patch_size: str = typer.Option("4,16,16", help="仅 dual_vit_3d 生效，例如 4,16,16"),
    vit_hidden_size: int = typer.Option(768),
    vit_mlp_dim: int = typer.Option(3072),
    vit_num_layers: int = typer.Option(12),
    vit_num_heads: int = typer.Option(12),
    unet_channels: str = typer.Option("16,32,64,128,256", help="仅 dual_unet_3d 生效"),
    unet_strides: str = typer.Option("2,2,2,2", help="仅 dual_unet_3d 生效"),
    unet_num_res_units: int = typer.Option(2, help="仅 dual_unet_3d 生效"),
    lr: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(0.05, help="AdamW weight decay（更强正则，默认比 PyTorch 的 0.01 更大）"),
    seed: int = typer.Option(42),
    label_smoothing: float = typer.Option(0.10, help="CrossEntropy label smoothing（更强正则）"),
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
    early_stop_patience: int = typer.Option(0, help="早停 patience（0=关闭）"),
    early_stop_metric: str = typer.Option("val_loss", help="val_loss | macro_f1 | macro_recall | macro_specificity | weighted_f1"),
    early_stop_min_delta: float = typer.Option(0.0, help="最小提升幅度（避免抖动）"),
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

    train_ds = EarCTDualDataset(
        index_df=train_df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
    )
    val_ds = EarCTDualDataset(
        index_df=val_df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    img_size = (int(num_slices), int(image_size), int(image_size))
    vit_patch = _parse_int_tuple(vit_patch_size, n=3)
    unet_ch = _parse_int_tuple(unet_channels)
    unet_st = _parse_int_tuple(unet_strides)

    net, spec = make_dual_model(
        model,
        num_classes=num_classes,
        in_channels=1,
        img_size=img_size,
        vit_patch_size=vit_patch,
        vit_hidden_size=vit_hidden_size,
        vit_mlp_dim=vit_mlp_dim,
        vit_num_layers=vit_num_layers,
        vit_num_heads=vit_num_heads,
        unet_channels=unet_ch,
        unet_strides=unet_st,
        unet_num_res_units=unet_num_res_units,
    )
    net = net.to(device)
    if compile and hasattr(torch, "compile"):
        net = torch.compile(net)

    if auto_batch:
        batch_size = _autotune_batch_size(
            model=net,
            device=device,
            num_classes=num_classes,
            num_slices=num_slices,
            image_size=image_size,
            amp=amp,
            max_batch_size=max_batch_size,
        )

        # Rebuild loaders with the chosen batch size.
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(weight_decay))
    try:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    except TypeError:
        # Older torch may not support label_smoothing.
        loss_fn = torch.nn.CrossEntropyLoss()

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
    typer.echo(f"dicom_root: {dicom_root}")
    if used_cache_dir is not None:
        typer.echo(f"cache_dir: {used_cache_dir} ({cache_dtype})")
    typer.echo(f"train_exams: {len(train_ds)}  val_exams: {len(val_ds)}")
    typer.echo(f"output: {out}")

    metric_mode = "min" if early_stop_metric == "val_loss" else "max"
    best_score = float("inf") if metric_mode == "min" else float("-inf")
    bad_epochs = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    def _rebuild_loaders(bs: int) -> None:
        nonlocal train_loader, val_loader
        train_loader = DataLoader(
            train_ds,
            batch_size=int(bs),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(bs),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    for epoch in range(1, epochs + 1):
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
                    amp=amp,
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
            "macro_recall": float(report["macro_recall"]),
            "macro_specificity": float(report["macro_specificity"]),
            "macro_f1": float(report["macro_f1"]),
            "weighted_f1": float(report["weighted_f1"]),
        }
        if early_stop_metric not in monitor_map:
            raise ValueError(f"unknown early_stop_metric: {early_stop_metric}")
        score = float(monitor_map[early_stop_metric])

        rec = {
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
            "val_metrics": {k: report[k] for k in ("accuracy", "macro_recall", "macro_specificity", "macro_f1", "weighted_f1", "total")},
            "model": {"name": spec.name, "kwargs": spec.kwargs},
            "hparams": {
                "label_task": str(task_spec.name),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "label_smoothing": float(label_smoothing),
                "augment": bool(augment),
                "aug_flip_prob": float(aug_flip_prob),
                "aug_intensity_prob": float(aug_intensity_prob),
                "aug_noise_prob": float(aug_noise_prob),
                "aug_gamma_prob": float(aug_gamma_prob),
            },
            "batch_size": int(batch_size),
            "early_stop": {"metric": early_stop_metric, "mode": metric_mode, "score": score, "patience": int(early_stop_patience), "min_delta": float(early_stop_min_delta)},
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        typer.echo(
            "epoch "
            f"{epoch}: train_loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} "
            f"(L={train_m['left_acc']:.3f} R={train_m['right_acc']:.3f})  "
            f"val_loss={val_m['loss']:.4f} acc={val_m['acc']:.3f} "
            f"(L={val_m['left_acc']:.3f} R={val_m['right_acc']:.3f})  "
            f"macro_recall={report['macro_recall']:.3f} macro_f1={report['macro_f1']:.3f}"
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
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "label_smoothing": float(label_smoothing),
            "augment": bool(augment),
            "aug_flip_prob": float(aug_flip_prob),
            "aug_intensity_prob": float(aug_intensity_prob),
            "aug_noise_prob": float(aug_noise_prob),
            "aug_gamma_prob": float(aug_gamma_prob),
            "state_dict": net.state_dict(),
            "val_loss": val_m["loss"],
            "early_stop_metric": early_stop_metric,
            "early_stop_score": score,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        improved = False
        if metric_mode == "min":
            improved = (best_score - score) > float(early_stop_min_delta)
        else:
            improved = (score - best_score) > float(early_stop_min_delta)

        if improved:
            best_score = score
            bad_epochs = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
        else:
            bad_epochs += 1

        if early_stop_patience > 0 and bad_epochs >= int(early_stop_patience):
            typer.echo(f"early stopping: no improvement on {early_stop_metric} for {early_stop_patience} epochs")
            break


if __name__ == "__main__":
    app()
