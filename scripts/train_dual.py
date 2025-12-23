from __future__ import annotations

import json
import time
from contextlib import nullcontext
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
            torch.cuda.synchronize()
            del x, y, logits, loss
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "oom" in msg:
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


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
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

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        m = batch["label_mask"].to(device, non_blocking=True)

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
    seed: int = typer.Option(42),
    amp: bool = typer.Option(True),
    tf32: bool = typer.Option(True, help="CUDA: 允许 TF32 加速 matmul/conv"),
    cudnn_benchmark: bool = typer.Option(True, help="CUDA: cudnn benchmark 以提高吞吐"),
    compile: bool = typer.Option(False, help="PyTorch 2: torch.compile 以提高吞吐"),
) -> None:
    train_csv, val_csv = _resolve_split_paths(splits_root, pct)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if train_df.empty or val_df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)
    num_classes = len(CLASS_ID_TO_NAME)

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

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
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
    out = output_dir or Path("outputs") / f"{safe_model}_{pct}pct_{ts}"
    ckpt_dir = out / "checkpoints"
    report_dir = out / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"

    typer.echo(f"task: 一次检查 -> 左/右双输出 6 分类  classes={num_classes}")
    typer.echo(f"model: {model}  pct={pct}%  batch_size={batch_size}  amp={amp}")
    typer.echo(f"dicom_root: {dicom_root}")
    if used_cache_dir is not None:
        typer.echo(f"cache_dir: {used_cache_dir} ({cache_dtype})")
    typer.echo(f"train_exams: {len(train_ds)}  val_exams: {len(val_ds)}")
    typer.echo(f"output: {out}")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(
            model=net,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
        )
        val_m = _run_epoch(
            model=net,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
            scaler=None,
            amp=amp,
            num_classes=num_classes,
            collect_cm=True,
        )

        cm = val_m.pop("confusion_matrix")
        report = classification_report_from_confusion(cm, class_id_to_name=CLASS_ID_TO_NAME)
        report_path = report_dir / f"epoch_{epoch}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        rec = {
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
            "val_metrics": {k: report[k] for k in ("accuracy", "macro_recall", "macro_specificity", "macro_f1", "weighted_f1", "total")},
            "model": {"name": spec.name, "kwargs": spec.kwargs},
            "batch_size": int(batch_size),
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
            "model_name": spec.name,
            "model_kwargs": spec.kwargs,
            "num_classes": num_classes,
            "num_slices": num_slices,
            "image_size": image_size,
            "batch_size": int(batch_size),
            "state_dict": net.state_dict(),
            "val_loss": val_m["loss"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")


if __name__ == "__main__":
    app()
